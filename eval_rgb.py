import argparse
import os
import numpy as np
from video_dataset import VideoDataset
import torch
from models import *
import sys
import gc

def run_demo(args):
    action_classes = [x.strip() for x in open('models/' + args.model + '/label_map.txt')]
    num_samples = len(VideoDataset(root_dir='datasets/' + args.dataset, stream='rgb', split=args.split))
    if args.model == 'i3d':
        model_class = I3D
    else:
        pass

    streams = ['rgb']
    num_preds_out_channels = 1 

    preds = np.zeros([num_samples, len(action_classes), num_preds_out_channels])
    truth_labels = np.zeros([num_samples,1])
    truth_labels = VideoDataset(csv_file='datasets/' + args.dataset + '/dataset.csv',
                                root_dir='datasets/' + args.dataset + '/data/',
                                stream='rgb').get_labels()

    np.savez('out/' + args.model + '/' + args.model + '-' + args.dataset + '-' + args.stream + '.npz',
             truth_labels=truth_labels,
             out_logits=out_logits, 
             preds=preds) 

    stream_indices = {'rgb': 0, 'flow': 1}
    for input_stream in ['rgb']:
        dataset = VideoDataset(csv_file='datasets/' + args.dataset + '/dataset.csv',
                               root_dir='datasets/' + args.dataset + '/data/',
                               stream=input_stream)
        model = model_class(num_classes=len(action_classes), modality=input_stream)
        model.eval()
        model.load_state_dict(torch.load('models/' + args.model + '/checkpoints/' + args.model + '_' + input_stream + '.pth'))
        model.cuda()

        for idx in range(len(dataset)):
            print ("stream: " + input_stream + "   idx: " + str(idx))
            print ("video shape: ")
            print (dataset[idx]['video'].shape)
            try:
                out_var, logits  = model( torch.autograd.Variable(torch.from_numpy(dataset[idx]['video'].transpose(0, 4, 1, 2, 3)).cuda()) )
                preds[idx,:,stream_indices[input_stream]] = logits.data.cpu().numpy()
                out_logits[idx][stream_indices[input_stream]] = logits
                # When done debugging, remove above 3 lines
                ground_truth = truth_labels[idx]
                prediction_idx = np.argmax(logits.data.cpu().numpy())
                print("ground truth: " + ground_truth + "\tprediction: " + action_classes[prediction_idx])
            except:
                print('error handling video ' + str(idx))

            np.savez('out/' + args.model + '/' + args.model + '-' + args.dataset + '-' + args.stream + '.npz',
                     truth_labels=truth_labels,
                     out_logits=out_logits, 
                     preds=preds)

        # Clearing GPU cache and clearing model from memory
        torch.cuda.empty_cache()
        model = None
        gc.collect()
        np.savez('out/' + args.model + '/' + args.model + '-' + args.dataset + '-' + args.stream + '.npz',
                 truth_labels=truth_labels, 
                 preds=preds) 


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
        choices=['rgb'],
        help='Input stream for model')
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test'],
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
        default='ViolentHumanActions_v1',
        help='Name of dataset to use')

    args = parser.parse_args()

    run_demo(args)
