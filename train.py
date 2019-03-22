import argparse
import os
import numpy as np
from video_dataset import VideoDataset
import torch
import torch.optim as optim
import torch.nn.functional as F
from models import *
from convert_parameters import parameters_to_vector
import sys
import gc

import subprocess #debug



# Note: for now we use a batch-size of 1.


def run_demo(args):
    action_classes = [x.strip() for x in open('models/' + args.model + '/label_map.txt')]
    num_samples = len(VideoDataset(root_dir='datasets/' + args.dataset, stream=args.stream, split=args.split))
    if args.model == 'i3d':
        model_class = I3D
    else:
        pass


    batch_size = 1 #debug

    out_logits = np.zeros([num_samples, len(action_classes)])
    truth_labels = np.zeros([num_samples,1])
    truth_labels = VideoDataset(root_dir='datasets/' + args.dataset, split=args.split, stream=args.stream).get_labels()

    np.savez('out/' + args.model + '/' + args.model + '-' + args.dataset + '-' + args.stream + args.suffix + '.npz',
             truth_labels=truth_labels,
             out_logits=out_logits)

    dataset = VideoDataset(root_dir='datasets/' + args.dataset, split=args.split, stream=args.stream, resize_frames=args.resize_frames)
    model = model_class(num_classes=len(action_classes), modality=args.stream)


    lr = 0.1 # debug, set in arg parser
    optimizer = optim.SGD(model.parameters(), lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)

    #param_vec = parameters_to_vector(model.parameters()) #debug
    #torch.save(param_vec, 'param_evc.pkl') #debug
    #print(param_vec.size())  #debug

    #model.eval()

    model.load_state_dict(torch.load('models/' + args.model + '/checkpoints/' + args.model + '_' + args.stream + '.pth'))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    #model.cuda()

    model.train()

    for idx in range(args.resume_iter, len(dataset)):
        print("stream: " + args.stream + "   idx: " + str(idx))

        if True: # lololo
        #with torch.no_grad():
            #input = torch.autograd.Variable(torch.from_numpy(dataset[idx]['video'].transpose(0, 4, 1, 2, 3))).cuda()
            #input = torch.autograd.Variable(torch.from_numpy(dataset[idx]['video'].transpose(0, 4, 1, 2, 3))).to(device) #debug


            print("*** Getting video sequence... ***") #debug
            label, video = dataset[idx]
            print("*** Transforming video sequence and loading onto GPU, if available... ***")  # debug
            video = torch.autograd.Variable(torch.from_numpy(video.transpose(0, 4, 1, 2, 3))).to(device)

            target = torch.tensor([action_classes.index(label)], dtype=torch.long, device=device)

            print("*** Computing forward pass... ***")  # debug
            out_var, logits = model(video)

            print("*** Computing loss... ***")  # debug
            loss = F.cross_entropy(input=out_var, target=target)  # debug
            optimizer.zero_grad()

            print("*** Computing backward pass... ***")  # debug
            loss.backward()
            print("*** Optimizer step... ***")  # debug
            optimizer.step()

            print("OKKKKK") #debug
            #sys.exit() #debug


    # Clearing GPU cache and clearing model from memory
    torch.cuda.empty_cache()
    model = None
    gc.collect()
    np.savez('out/' + args.model + '/' + args.model + '-' + args.dataset + '-' + args.stream + args.suffix + '.npz',
             truth_labels=truth_labels, 
             out_logits=out_logits) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluating the I3D model, and possible variants, on video datasets')
    # TODO: eventually, we will want to evaluate models on various possible checkpoints, thus add an argument for specific checkpoints

    parser.add_argument(
        '--model',
        type=str,
        default='i3d',
        choices=['i3d'],
        help='Model to use')
    parser.add_argument(
        '--stream',
        type=str,
        default='rgb',
        choices=['rgb', 'flow'],
        help='Input stream for model')
    parser.add_argument(
        '--resize_frames',
        type=float,
        default=1.0,
        help='Factor for resizing frames')
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'valid', 'train'],
        help='Dataset split')
    parser.add_argument(
        '--pre-trained',
        type=str,
        default='both',
        #choices=['rgb', 'flow', 'both', 'none']
        # Since this is an evaluation script, we have to load pre-trained weights, but use the line above for a training script
        choices=['both'],
        help='Whether to use pre-trained weights (from Kinetics-400)')
    parser.add_argument(
        '--dataset',
        type=str,
        default='ViolentHumanActions_v2',
        help='Name of dataset to use')
    parser.add_argument(
        '--suffix',
        type=str,
        default='',
        help='Suffix to append to results file')

    parser.add_argument(
        '--resume_iter',
        type=int,
        default=0,
        help='Iteration or sample index at which to resume, if resuming an interrupted evaluation')


    args = parser.parse_args()

    run_demo(args)
