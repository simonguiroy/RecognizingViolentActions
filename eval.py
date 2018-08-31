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
    num_samples = len(VideoDataset(csv_file='datasets/' + args.dataset + '/dataset.csv',
                               root_dir='datasets/' + args.dataset + '/data/',
                               stream='rgb'))
    if args.model == 'i3d':
        model_class = I3D
    else:
        pass

    if args.stream == 'rgb':
        streams = ['rgb']
        num_preds_out_channels = 1 
    elif args.stream == 'flow':
        streams = ['flow'] 
        num_preds_out_channels = 1 
    else:
        streams = ['rgb', 'flow']
        num_preds_out_channels = 3 
        out_logits = [[None,None]] * num_samples

    preds = np.zeros([num_samples, len(action_classes), num_preds_out_channels])
    truth_labels = np.zeros([num_samples,1])
    truth_labels = VideoDataset(csv_file='datasets/' + args.dataset + '/dataset.csv',
                                root_dir='datasets/' + args.dataset + '/data/',
                                stream='rgb').get_labels()

    np.savez('out/' + args.model + '/' + args.model + '-' + args.dataset + '-' + args.stream + '.npz',
             truth_labels=truth_labels, 
             preds=preds) 

    stream_idx = 0
    for input_stream in streams:
        dataset = VideoDataset(csv_file='datasets/' + args.dataset + '/dataset.csv',
                               root_dir='datasets/' + args.dataset + '/data/',
                               stream=input_stream)
        model = model_class(num_classes=len(action_classes), modality=input_stream)
        model.eval()
        model.load_state_dict(torch.load('models/' + args.model + '/checkpoints/' + args.model + '_' + input_stream + '.pth'))
        model.cuda()

        #for idx in range(len(dataset)):
        for idx in range(5):
            idx += 95
            print ("stream: " + input_stream + "   idx: " + str(idx))
            try:
                out_var, logits  = model( torch.autograd.Variable(torch.from_numpy(dataset[idx]['video'].transpose(0, 4, 1, 2, 3)).cuda()) )
                preds[idx,:,stream_idx] = logits.data.cpu().numpy()
                out_logits[idx][stream_idx] = logits
            except:
                print('error handling video ' + str(idx))

            if idx % 20 == 0:
                np.savez('out/' + args.model + '/' + args.model + '-' + args.dataset + '-' + args.stream + '.npz',
                         truth_labels=truth_labels, 
                         preds=preds)

        # Clearing GPU cache and clearing model from memory
        torch.cuda.empty_cache()
        model = None
        gc.collect()
        np.savez('out/' + args.model + '/' + args.model + '-' + args.dataset + '-' + args.stream + '.npz',
                 truth_labels=truth_labels, 
                 preds=preds) 
        stream_idx += 1

    # Joint model
    if args.stream == 'joint':
        for idx in range(len(dataset)):
            joint_logits = out_logits[idx][0] + out_logits[idx][1]
            print ( torch.nn.functional.softmax(joint_logits, 1).data.cpu().numpy().shape )
            preds[idx,:,2] = torch.nn.functional.softmax(joint_logits, 1).data.cpu().numpy().T

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
        default='joint',
        choices=['rgb', 'flow', 'joint'],
        help='Input stream for model')
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
        default='ViolentHumanActions_v0',
        help='Name of dataset to use',
        choices=['ViolentHumanActions_v0'])
    parser.add_argument(
        '--top_k',
        type=int,
        default='5',
        help='When display_samples, number of top classes to display')

    args = parser.parse_args()
    
    run_demo(args)
