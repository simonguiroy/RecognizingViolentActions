import numpy as np
from video_dataset import VideoDataset
import torch
from torch.utils.data import DataLoader
from models.i3d.i3d import I3D
import sys
import csv
import argparse
from video_dataset import my_collate, DeterministicSubsetRandomSampler


def run(args):

    saved_model_name = test_args.saved_model_path.split('/')[-1].split('.pkl')[0]
    # Loading checkpoint of saved model
    checkpoint = torch.load(test_args.saved_model_path)
    # Retrieving training arguments
    args = checkpoint['args']

    # Seeding random number generators to have determinism and reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    num_samples = len(VideoDataset(root_dir='datasets/' + args.dataset, split='test'))
    test_sampler = DeterministicSubsetRandomSampler(range(0, num_samples))

    # List of action class labels
    action_classes = [x.strip() for x in open('models/i3d/label_map.txt')]

    # Initializing model
    model = I3D(num_classes=len(action_classes), modality=args.stream)
    model.load_state_dict(checkpoint['state_dict'])

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Dataset and dataloader
    test_dataset = VideoDataset(root_dir='datasets/' + args.dataset, split='test', stream=args.stream,
                                 max_frames_per_clip=-1)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                  collate_fn=my_collate, sampler=test_sampler)

    if torch.cuda.is_available() and args.device_type == 'gpu':
        device = torch.device('cuda:' + str(args.gpu_id))
    else:
        device = torch.device('cpu')

    model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    preds = []

    # Iterating through the testing dataloader
    for batch_idx, batch in enumerate(test_dataloader):
        label = batch[0][0]
        video = batch[1][0]
        video = torch.autograd.Variable(torch.from_numpy(video.transpose(0, 4, 1, 2, 3))).to(device)
        target = torch.tensor([action_classes.index(label)], dtype=torch.long, device=device)

        with torch.no_grad():
            out_var, logits = model(video)
        preds.append(out_var.cpu())
        torch.save({'preds': preds, 'shuffled_indices': test_sampler.shuffled_indices}, 'out/i3d/preds/preds_' +
                   saved_model_name + '.pkl')
        running_loss += criterion(out_var, target).item()
        running_corrects += target.data.cpu().item() == torch.argmax(out_var).cpu().item()
        print('')
        print("Stream: " + args.stream + " Target: " + label + " Prediction: " + action_classes[torch.argmax(out_var)])

        # Printing running accuracy and loss for every 10 batch
        if batch_idx % 10 == 0 and batch_idx > 0:
            completion = ((batch_idx + 1) / len(test_dataset))
            current_loss = running_loss / (batch_idx + 1)
            current_acc = running_corrects / (batch_idx + 1)
            print("[Test] Completion: {} Loss: {} Acc: {}".format(completion, current_loss, current_acc))

            # Overwriting log file every ten sample
            with open('out/i3d/logs/' + saved_model_name + '_test.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'test_loss', 'test_acc'])
                writer.writerow([args.resume_epoch, current_loss, current_acc])

    test_loss = running_loss / len(test_dataset)
    test_acc = running_corrects / len(test_dataset)

    print("*****************************************")
    print("[Test] Completed! Loss: {} Acc: {}".format(test_loss, test_acc))

    with open('out/i3d/logs/' + saved_model_name + '_test.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'test_loss', 'test_acc'])
        writer.writerow([args.resume_epoch, test_loss, test_acc])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recognizing Violent Human Actions')
    parser.add_argument('--saved_model_path', type=str, help='Path to saved model to test.')
    test_args = parser.parse_args()

    run(test_args)
