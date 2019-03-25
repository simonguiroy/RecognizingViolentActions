import os
import numpy as np
from video_dataset import VideoDataset
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import *
from convert_parameters import parameters_to_vector
from parser import get_args
import sys
import gc
from time import time
import subprocess #debug
import tqdm
import csv


# Since different videos may have different number of frames, we can't pass a batch of videos as a single
# input Tensor to the model. We thus create the batch as a list, compute the forward pass one video at a time,
# accumulate the loss and average it, before computing the backward pass.
def my_collate(batch):
    """
    This custom collate function returns the batch where videos and labels are sorted in a list. This allows to have a
    batch with videos of variable number of frames. This function is used by the dataloader internally.
    :param batch: The batch returned by the dataset.
    :return: A single batch (list), where the first element is a list of labels, and the second is a list of videos
    """
    labels = [item[0] for item in batch]
    videos = [item[1] for item in batch]
    return [labels, videos]


def run(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # To avoid exceding the RAM limit when
    max_frames_for_ram = 500
    if args.batch_size == 1:
        max_frames_per_clip = -1
    else:
        max_frames_per_clip = int(max_frames_for_ram / args.batch_size)

    action_classes = [x.strip() for x in open('models/i3d/label_map.txt')]

    # Datasets and dataloaders
    train_dataset = VideoDataset(root_dir='datasets/' + args.dataset, split='train', stream=args.stream,
                           max_frames_per_clip=max_frames_per_clip)

    valid_dataset = VideoDataset(root_dir='datasets/' + args.dataset, split='valid', stream=args.stream,
                                 max_frames_per_clip=-1)

    test_dataset = VideoDataset(root_dir='datasets/' + args.dataset, split='test', stream=args.stream,
                                 max_frames_per_clip=-1)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                           collate_fn=my_collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=1,
                                  collate_fn=my_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1,
                                  collate_fn=my_collate)


    # Initializing model
    model = I3D(num_classes=len(action_classes), modality=args.stream)

    # Initializing optimizer
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, dampening=0, weight_decay=0,
                          nesterov=False)

    # Loading model and optimizer state
    if args.resume_epoch == 0:
        model.load_state_dict(torch.load('out/i3d/checkpoints/i3d_' + args.stream + '.pth'))
        with open('out/i3d/saved_models/i3d_' + args.stream + '_train.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'])

    else:
        checkpoint = torch.load('out/i3d/saved_models/i3d_' + args.stream + '_epoch_' + args.resume_epoch + '.pth')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])


    # Loading model onto device
    if torch.cuda.is_available() and args.device_type == 'gpu':
        device = torch.device('cuda:' + str(args.gpu_id))
    else:
        device = torch.device('cpu')
    model.to(device)

    # NOTE: Calling model.train() seems to drastically affect the model prediction.
    #model.train()

    for epoch in range(args.resume_epochs, args.num_epochs):
        print("Epoch: " + str(epoch))
        running_loss = 0.0
        running_corrects = 0

        for batch_idx, batch in enumerate(train_dataloader):

            for sample_idx in range(args.batch_size):

                label = batch[0][sample_idx]
                video = batch[1][sample_idx]
                video = torch.autograd.Variable(torch.from_numpy(video.transpose(0, 4, 1, 2, 3))).to(device)
                target = torch.tensor([action_classes.index(label)], dtype=torch.long, device=device)

                optimizer.zero_grad()
                out_var, logits = model(video)
                loss = F.cross_entropy(input=out_var, target=target)

                running_loss += loss.item()
                running_corrects += target.data.cpu().item() == torch.argmax(out_var).cpu().item()
                #print("Target: " + label + " Prediction: " + action_classes[torch.argmax(out_var)])

            loss /= args.batch_size
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                completion = 100 * int(batch_idx+1 * args.batch_size / len(train_dataset))
                current_loss = running_loss / (batch_idx+1 * args.batch_size / len(train_dataset))
                current_acc = running_corrects / (batch_idx+1 * args.batch_size / len(train_dataset))
                print("[Train] Epoch: {}/{} Epoch completion: {} % Loss: {} Acc: {}".format(epoch, args.num_epochs,
                                                                                            completion, current_loss,
                                                                                            current_acc))

        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects / len(train_dataset)

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'opt_dict': optimizer.state_dict(),
        }, 'out/i3d/saved_models/i3d_' + args.stream + '_epoch_' + epoch + '.pth')
        print("Saving model...")

        running_loss = 0.0
        running_corrects = 0

        print("[Train] Epoch: {}/{} Loss: {} Acc: {}".format(epoch, args.num_epochs, train_loss, train_acc))

        for batch_idx, batch in enumerate(valid_dataloader):

            label = batch[0][0]
            video = batch[1][0]
            video = torch.autograd.Variable(torch.from_numpy(video.transpose(0, 4, 1, 2, 3))).to(device)
            target = torch.tensor([action_classes.index(label)], dtype=torch.long, device=device)
            with torch.no_grad():
                out_var, logits = model(video)
            loss = F.cross_entropy(input=out_var, target=target)
            running_loss += loss.item()
            running_corrects += target.data.cpu().item() == torch.argmax(out_var).cpu().item()

        valid_loss = running_loss / len(train_dataset)
        valid_acc = running_corrects / len(train_dataset)
        print("[Valid] Epoch: {}/{} Loss: {} Acc: {}".format(epoch, args.num_epochs, valid_loss, valid_acc))

        with open('out/i3d/saved_models/i3d_' + args.stream + '_train.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])

    running_loss = 0.0
    running_corrects = 0

    for batch_idx, batch in enumerate(test_dataloader):
        label = batch[0][0]
        video = batch[1][0]
        video = torch.autograd.Variable(torch.from_numpy(video.transpose(0, 4, 1, 2, 3))).to(device)
        target = torch.tensor([action_classes.index(label)], dtype=torch.long, device=device)
        with torch.no_grad():
            out_var, logits = model(video)
        loss = F.cross_entropy(input=out_var, target=target)
        running_loss += loss.item()
        running_corrects += target.data.cpu().item() == torch.argmax(out_var).cpu().item()

    test_loss = running_loss / len(train_dataset)
    test_acc = running_corrects / len(train_dataset)

    print("*****************************************")
    print("[Test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch, args.num_epochs, test_loss, test_acc))

    with open('out/i3d/saved_models/i3d_' + args.stream + '_test.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'test_loss', 'test_acc'])
        writer.writerow([epoch, test_loss, test_acc])

    # Clearing GPU cache and clearing model from memory
    torch.cuda.empty_cache()
    model = None
    gc.collect()


if __name__ == "__main__":
    args = get_args()
    run(args)
