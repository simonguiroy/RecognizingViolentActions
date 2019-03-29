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

    # List of action class labels
    action_classes = [x.strip() for x in open('models/i3d/label_map.txt')]

    # Loading checkpoint for specified epoch
    checkpoint = torch.load('out/i3d/saved_models/i3d_' + args.stream + '_epoch-' + str(args.resume_epoch) + '.pkl')
    if args.resume_epoch != 0:
        args.seed = checkpoint['seed']

    #TODO: Since we will save model predictions, seeding here must be based on previous testing runs.
    # Seeding random number generators to have determinism and reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    test_sampler_indices = torch.load('test_sampler.pkl')
    print(test_sampler_indices) #debug

    num_samples = len(VideoDataset(root_dir='datasets/' + args.dataset, split='test'))
    test_sampler = DeterministicSubsetRandomSampler(range(0, num_samples))
    test_sampler.shuffled_indices = test_sampler_indices
    print(test_sampler.shuffled_indices) #debug
    sys.exit() #debug

    #torch.save(test_sampler.shuffled_indices, 'test_sampler.pkl') #debug

    # Initializing model
    models = dict()
    if args.stream == 'dual':
        models['rgb'] = I3D(num_classes=len(action_classes), modality='rgb')
        models['flow'] = I3D(num_classes=len(action_classes), modality='flow')
    else:
        models[args.stream] = I3D(num_classes=len(action_classes), modality=args.stream)

    # Loading model state
    for key in models.keys():
        models[key].load_state_dict(checkpoint['state_dict'])

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # TODO: make datasets shuffle identically for both streams
    # Dataset and dataloader
    test_dataset = VideoDataset(root_dir='datasets/' + args.dataset, split='test', stream=args.stream,
                                 max_frames_per_clip=-1)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                  collate_fn=my_collate, sampler=test_sampler)

    # Loading model state
    """
    if args.resume_epoch == 0:
        logs_fields = []
        logs_fields.append('Epoch')
        for key in models.keys():
            logs_fields.append(key + '_loss')
            logs_fields.append(key + '_acc')
        if args.stream == 'dual':
            logs_fields.append('dual_acc')
        with open('out/i3d/logs/i3d_' + args.stream + '_test.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(logs_fields)
    """

    if torch.cuda.is_available() and args.device_type == 'gpu':
        device = torch.device('cuda:' + str(args.gpu_id))
    else:
        device = torch.device('cpu')

    # Loading model onto device, setting eval mode.
    for key in models.keys():
        models[key].to(device)
        models[key].eval()

    running_loss = 0.0
    running_corrects = 0

    preds = []

    #TODO: For now we are just collecting statistics and only testing one stream at a time
    # Iterating through the testing dataloader
    for batch_idx, batch in enumerate(test_dataloader):
        label = batch[0][0]
        video = batch[1][0][args.stream]
        video = torch.autograd.Variable(torch.from_numpy(video.transpose(0, 4, 1, 2, 3))).to(device)
        target = torch.tensor([action_classes.index(label)], dtype=torch.long, device=device)

        with torch.no_grad():
            out_var, logits = models[args.stream](video)
        preds.append(out_var.cpu())
        torch.save({'preds': preds, 'shuffled_indices': test_sampler.shuffled_indices}, 'out/i3d/preds/preds_' +
                   args.stream + '_epoch-' + str(args.resume_epoch) + '.pkl')
        running_loss += criterion(out_var, target).item()
        running_corrects += target.data.cpu().item() == torch.argmax(out_var).cpu().item()
        print('')
        print("Stream: " + args.stream + " Target: " + label + " Prediction: " + action_classes[torch.argmax(out_var)])

        # Printing running accuracy and loss for every 10 batch
        if batch_idx % 5 == 0 and batch_idx > 0:
            completion = ((batch_idx + 1) / len(test_dataset))
            current_loss = running_loss / (batch_idx + 1)
            current_acc = running_corrects / (batch_idx + 1)
            print("[Test] Completion: {} Loss: {} Acc: {}".format(completion, current_loss, current_acc))

            # Overwriting log file every ten sample
            with open('out/i3d/logs/i3d_' + args.stream + '_test.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'test_loss', 'test_acc'])
                writer.writerow([args.resume_epoch, current_loss, current_acc])

    test_loss = running_loss / len(test_dataset)
    test_acc = running_corrects / len(test_dataset)

    print("*****************************************")
    print("[Test] Completed! Loss: {} Acc: {}".format(test_loss, test_acc))

    with open('out/i3d/logs/i3d_' + args.stream + '_test.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'test_loss', 'test_acc'])
        writer.writerow([args.resume_epoch, test_loss, test_acc])

    # Clearing GPU cache and clearing model from memory
    torch.cuda.empty_cache()
    model = None
    gc.collect()


if __name__ == "__main__":
    args = get_args()
    run(args)
