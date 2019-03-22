import argparse
import os
import numpy as np
from video_dataset import VideoDataset
import torch
from models import *
import sys
import gc

import subprocess #debug

def run_demo(args):
    action_classes = [x.strip() for x in open('models/' + args.model + '/label_map.txt')]
    num_samples = len(VideoDataset(root_dir='datasets/' + args.dataset, stream=args.stream, split=args.split))
    if args.model == 'i3d':
        model_class = I3D
    else:
        pass


    out_logits = np.zeros([num_samples, len(action_classes)])
    truth_labels = np.zeros([num_samples,1])
    truth_labels = VideoDataset(root_dir='datasets/' + args.dataset, split=args.split, stream=args.stream).get_labels()

    np.savez('out/' + args.model + '/' + args.model + '-' + args.dataset + '-' + args.stream + args.suffix + '.npz',
             truth_labels=truth_labels,
             out_logits=out_logits)

    dataset = VideoDataset(root_dir='datasets/' + args.dataset, split=args.split, stream=args.stream, resize_frames=args.resize_frames)
    model = model_class(num_classes=len(action_classes), modality=args.stream)

    model.load_state_dict(torch.load('models/' + args.model + '/checkpoints/' + args.model + '_' + args.stream + '.pth'))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    #model.cuda()

    model.eval()

    for idx in range(args.resume_iter, len(dataset)):
        print("stream: " + args.stream + "   idx: " + str(idx))
        with torch.no_grad():
            #input = torch.autograd.Variable(torch.from_numpy(dataset[idx]['video'].transpose(0, 4, 1, 2, 3))).cuda()

            # input = torch.autograd.Variable(torch.from_numpy(dataset[idx]['video'].transpose(0, 4, 1, 2, 3))).cuda()
            # input = torch.autograd.Variable(torch.from_numpy(dataset[idx]['video'].transpose(0, 4, 1, 2, 3))).to(device) #debug

            print("*** Getting video sequence... ***")  # debug
            label, video = dataset[idx]
            print("*** Transforming video sequence and loading onto GPU, if available... ***")  # debug
            video = torch.autograd.Variable(torch.from_numpy(video.transpose(0, 4, 1, 2, 3))).to(device)

            target = torch.tensor([action_classes.index(label)], dtype=torch.long, device=device)

            print("*** Computing forward pass... ***")  # debug
            out_var, logits = model(video)
            print("ground truth: " + label + "\tprediction: " + action_classes[torch.argmax(out_var[0]).item()])
            print("OKKKKK")  # debug
            # sys.exit() #debug

            """
            try:
                #subprocess.Popen("nvidia-smi") #debug
                out_var, logits  = model(input)
                out_logits[idx,:] = logits.data.cpu().numpy()

                #To display predictions at output
                ground_truth = truth_labels[idx]
                prediction_idx = np.argmax(logits.data.cpu().numpy())
                print("ground truth: " + ground_truth + "\tprediction: " + action_classes[prediction_idx])
                print('\n\n')
            except:
                print('error handling video ' + str(idx))

                ##### DEBUGGING
                # Performing what goes wrong then halting program
                subprocess.Popen("nvidia-smi")  # debug
                out_var, logits = model(input)
                out_logits[idx, :] = logits.data.cpu().numpy()

                # To display predictions at output
                ground_truth = truth_labels[idx]
                prediction_idx = np.argmax(logits.data.cpu().numpy())
                print("ground truth: " + ground_truth + "\tprediction: " + action_classes[prediction_idx])
                print('\n\n')
                sys.exit() #debug
                
            """

            #np.savez('out/' + args.model + '/' + args.model + '-' + args.dataset + '-' + args.stream + args.suffix + '.npz',
            #         truth_labels=truth_labels,
            #         out_logits=out_logits)


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
