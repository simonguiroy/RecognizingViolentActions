import numpy as np
from video_dataset import VideoDataset
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.i3d.i3d import I3D
from parser import get_args
import sys
import subprocess
import csv
import math
from video_dataset import my_collate, DeterministicSubsetRandomSampler


MAX_FRAMES_FOR_16GB_GPU = 500
NUM_MIB_FOR_16GB_GPU = 16119


def run(args):

    # Seeding random number generators to have determinism and reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    num_samples = len(VideoDataset(root_dir='datasets/' + args.dataset, split='train'))
    train_sampler = DeterministicSubsetRandomSampler(range(0, num_samples))

    # Getting Free Buffer free memory of the GPU
    subproc_out = subprocess.check_output(['nvidia-smi', '--id=' + str(args.gpu_id), '--query-gpu=memory.free', '--format=csv,nounits'])
    num_MiB = int(str(subproc_out).split('\\n')[1])

    # Avoiding exceeding the RAM limit when accumulating graphs for gradient descent. This should result in using
    # approximately 80% of the Frame Buffer free memory of the GPU.
    max_frames_per_batch = int(MAX_FRAMES_FOR_16GB_GPU * (num_MiB / NUM_MIB_FOR_16GB_GPU))
    max_frames_per_clip = int(max_frames_per_batch / args.batch_size)

    if args.max_frames_per_clip <= 0:
        print("Invalid value for max number of frames per clip!")
        sys.exit()

    elif args.max_frames_per_clip > max_frames_per_clip:
        print("You required too many frames per clip for your device memory!")
        print("Max frames required: " + str(args.max_frames_per_clip) + "  Max frames allowable: " +
              str(max_frames_per_clip))
        sys.exit()
    else:
        max_frames_per_clip = args.max_frames_per_clip

    # List of action class labels
    action_classes = [x.strip() for x in open('models/i3d/label_map.txt')]

    # Loading checkpoint for specified epoch
    checkpoint = torch.load('out/i3d/saved_models/i3d_' + args.stream + '_epoch-' + str(args.resume_epoch) + '.pkl')
    if args.resume_epoch != 0:
        train_sampler = checkpoint['train_sampler']

    # Initializing model
    model = I3D(num_classes=len(action_classes), modality=args.stream)
    model.load_state_dict(checkpoint['state_dict'])

    # Initializing optimizer
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, dampening=0, weight_decay=1e-7,
                          nesterov=False)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    if args.resume_epoch == 0:
        with open('out/i3d/logs/i3d_' + args.stream + '_train.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'iter', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'])

    with open('failed_videos.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['batch_idx', 'sample_idx'])


    # Datasets and dataloaders
    train_dataset = VideoDataset(root_dir='datasets/' + args.dataset, split='train', stream=args.stream,
                           max_frames_per_clip=max_frames_per_clip)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                           collate_fn=my_collate, sampler=train_sampler)
    num_train_iters = math.ceil(len(train_dataset) / args.batch_size)

    valid_dataset = VideoDataset(root_dir='datasets/' + args.dataset, split='valid', stream=args.stream,
                                 max_frames_per_clip=-1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=1,
                                  collate_fn=my_collate)

    # Loading model onto device
    if torch.cuda.is_available() and args.device_type == 'gpu':
        device = torch.device('cuda:' + str(args.gpu_id))
    else:
        device = torch.device('cpu')
    model.to(device)
    # NOTE: Calling model.train() seems to drastically affect the model prediction.
    model.eval()

    # Training for specified number of epochs
    for epoch in range(args.resume_epoch, args.num_epochs):
        print("Epoch: " + str(epoch))
        running_loss = 0.0
        running_corrects = 0

        # Iterating through the training dataloader
        for batch_idx, batch in enumerate(train_dataloader):
            print('Batch_idx: ' + str(batch_idx))

            # Computing batch loss with one forward pass at a time, since videos may have different lengths
            batch_loss = 0.0
            true_batch_size = 0
            for sample_idx in range(args.batch_size):

                label = batch[0][sample_idx]
                video = batch[1][sample_idx]
                video = torch.autograd.Variable(torch.from_numpy(video.transpose(0, 4, 1, 2, 3))).to(device)
                target = torch.tensor([action_classes.index(label)], dtype=torch.long, device=device)

                optimizer.zero_grad()
                try:
                    out_var, logits = model(video)
                    true_batch_size += 1
                except:
                    print('Failed to compute forward pass on video!')
                    with open('failed_videos.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([batch_idx, sample_idx])

                loss = criterion(out_var, target)
                batch_loss += loss
                running_loss += loss.item()
                running_corrects += target.data.cpu().item() == torch.argmax(out_var).cpu().item()
                print('')
                print("Stream: " + args.stream + " Target: " + label + " Prediction: " + action_classes[torch.argmax(out_var)])

            batch_loss /= true_batch_size
            batch_loss.backward()
            optimizer.step()

            # Printing running accuracy and loss for specified log frequency
            if batch_idx % args.log_frequency == 0 and args.log_frequency != -1 and batch_idx > 0:
                completion = ((batch_idx+1) * args.batch_size / len(train_dataset))
                current_loss = running_loss / ((batch_idx+1) * args.batch_size)
                current_acc = running_corrects / ((batch_idx+1) * args.batch_size)
                print("[Train] Epoch: {}/{} Epoch completion: {} Loss: {} Acc: {}".format(epoch, args.num_epochs,
                                                                                            completion, current_loss,
                                                                                            current_acc))
                # Writing training results in log file
                with open('out/i3d/logs/i3d_' + args.stream +
                          '_train.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, batch_idx + (epoch * num_train_iters), current_loss, current_acc, -1, -1])

                # Saving model
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'args': args,
                    'train_sampler': train_sampler
                }, 'out/i3d/saved_models/i3d_' + args.stream + '_epoch-' + str(epoch) + '_iter-' + str(batch_idx) + '.pkl')
                print("Saving model...")

        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects / len(train_dataset)
        print("[Train] Epoch: {}/{} Loss: {} Acc: {}".format(epoch, args.num_epochs, train_loss, train_acc))

        # At every end of epoch, saving model.
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'args': args,
            'train_sampler': train_sampler
        }, 'out/i3d/saved_models/i3d_' + args.stream + '_epoch-' + str(epoch + 1) + '.pkl')
        print("Saving model...")

        # Running validation, with batches of size 1
        running_loss = 0.0
        running_corrects = 0
        for batch_idx, batch in enumerate(valid_dataloader):

            label = batch[0][0]
            video = batch[1][0]
            video = torch.autograd.Variable(torch.from_numpy(video.transpose(0, 4, 1, 2, 3))).to(device)
            target = torch.tensor([action_classes.index(label)], dtype=torch.long, device=device)
            with torch.no_grad():
                out_var, logits = model(video)
            running_loss += criterion(out_var, target).item()
            running_corrects += target.data.cpu().item() == torch.argmax(out_var).cpu().item()

        valid_loss = running_loss / len(valid_dataset)
        valid_acc = running_corrects / len(valid_dataset)
        print("[Valid] Epoch: {}/{} Loss: {} Acc: {}".format(epoch, args.num_epochs, valid_loss, valid_acc))

        # Writing training and validation results in log file
        with open('out/i3d/logs/i3d_' + args.stream + '_train.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, (epoch + 1) * num_train_iters, train_loss, train_acc, valid_loss, valid_acc])


if __name__ == "__main__":
    args = get_args()
    run(args)
