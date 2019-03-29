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
import random

#### Move initial model checkpoints to saved_models, rename for resume_iter = 0, change if statement to load checkpoint
# to make coherent, increment epoch number so that at first epoch completion, model is saved under epoch_1. For test script,
# apply same logic. In test script, when a testing run is executed, the log file must be created, thus there's no checking
# what is the resume iter. Also in test script, only create log file at the end of the testing run, then write in it. In testing
# script, perhaps also save other infos and outputs, like target class and predicted class, as labels.

# Make sure that when resuming at a given epoch, that the dataloaders have the right state! This is important if dataloaders
# don't go through the entire dataset within one epoch!

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


class DeterministicSubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement. To enable reproducibility, list
    of randomly shuffled indices if created during __init__().

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.shuffled_indices = torch.randperm(len(indices))

    def __iter__(self):
        return (self.shuffled_indices[i] for i in range(len(self.shuffled_indices)))

    def __len__(self):
        return len(self.shuffled_indices)


def run(args):

    # Avoiding exceeding the RAM limit when accumulating graphs for gradient descent
    max_frames_per_batch = 500
    if args.batch_size == 1:
        max_frames_per_clip = -1
    else:
        max_frames_per_clip = int(max_frames_per_batch / args.batch_size)

    # List of action class labels
    action_classes = [x.strip() for x in open('models/i3d/label_map.txt')]

    # Loading checkpoint for specified epoch
    checkpoint = torch.load('out/i3d/saved_models/i3d_' + args.stream + '_epoch-' + str(args.resume_epoch) + '.pkl')
    if args.resume_epoch != 0:
        args.seed = checkpoint['seed']

    # Seeding random number generators to have determinism and reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    num_samples = len(VideoDataset(root_dir='datasets/' + args.dataset, split='train'))
    train_sampler = DeterministicSubsetRandomSampler(range(0, num_samples))

    # Initializing model
    model = I3D(num_classes=len(action_classes), modality=args.stream)
    model.load_state_dict(checkpoint['state_dict'])

    # Initializing optimizer
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, dampening=0, weight_decay=1e-7,
                          nesterov=False)
    if args.resume_epoch != 0:
        optimizer.load_state_dict(checkpoint['opt_dict'])

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    if args.resume_epoch == 0:
        with open('out/i3d/logs/i3d_' + args.stream + '_train.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'])


    # Datasets and dataloaders
    train_dataset = VideoDataset(root_dir='datasets/' + args.dataset, split='train', stream=args.stream,
                           max_frames_per_clip=max_frames_per_clip)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                           collate_fn=my_collate, sampler=train_sampler)

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
    #model.train()
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
            for sample_idx in range(args.batch_size):

                label = batch[0][sample_idx]
                video = batch[1][sample_idx][args.stream]
                video = torch.autograd.Variable(torch.from_numpy(video.transpose(0, 4, 1, 2, 3))).to(device)
                target = torch.tensor([action_classes.index(label)], dtype=torch.long, device=device)

                optimizer.zero_grad()
                out_var, logits = model(video)
                out_var = out_var
                loss = criterion(out_var, target)

                batch_loss += loss
                running_loss += loss.item()
                running_corrects += target.data.cpu().item() == torch.argmax(out_var).cpu().item()
                print('')
                print("Stream: " + args.stream + " Target: " + label + " Prediction: " + action_classes[torch.argmax(out_var)])

            batch_loss /= args.batch_size
            batch_loss.backward()
            optimizer.step()

            # Printing running accuracy and loss for every 10 batch
            if batch_idx % 10 == 0 and batch_idx > 0:
                completion = ((batch_idx+1) * args.batch_size / len(train_dataset))
                current_loss = running_loss / ((batch_idx+1) * args.batch_size)
                current_acc = running_corrects / ((batch_idx+1) * args.batch_size)
                print("[Train] Epoch: {}/{} Epoch completion: {} Loss: {} Acc: {}".format(epoch, args.num_epochs,
                                                                                            completion, current_loss,
                                                                                            current_acc))

        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects / len(train_dataset)
        print("[Train] Epoch: {}/{} Loss: {} Acc: {}".format(epoch, args.num_epochs, train_loss, train_acc))

        # At every end of epoch, saving model and optimizer state to checkpoint
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'opt_dict': optimizer.state_dict(),
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

        valid_loss = running_loss / len(train_dataset)
        valid_acc = running_corrects / len(train_dataset)
        print("[Valid] Epoch: {}/{} Loss: {} Acc: {}".format(epoch, args.num_epochs, valid_loss, valid_acc))

        # Writing training and validation results in log file
        with open('out/i3d/logs/i3d_' + args.stream + '_train.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])

    # Clearing GPU cache and clearing model from memory
    torch.cuda.empty_cache()
    model = None
    gc.collect()


if __name__ == "__main__":
    args = get_args()
    if args.stream == 'dual':
        print("Can select \'dual\' can only be selected at test time! During training, select \'rgb\' or \'flow\'")
    run(args)
