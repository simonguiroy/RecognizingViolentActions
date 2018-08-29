import argparse
import os
import numpy as np
from video_dataset import VideoDataset
import torch
from models import *
import sys
import gc

#rgb_pt_checkpoint = 'models/i3d/checkpoints/i3d_RGB.pth'
#flow_pt_checkpoint = 'models/i3d/checkpoints/i3d_FLOW.pth'

"""
    def get_scores(sample, model):
        sample_var = torch.autograd.Variable(torch.from_numpy(sample).cuda())
        out_var, out_logit = model(sample_var)
        out_tensor = out_var.data.cpu()

        top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

        print(
            'Top {} classes and associated probabilities: '.format(args.top_k))
        for i in range(args.top_k):
            print('[{}]: {:.6E}'.format(action_classes[top_idx[0, i]],
                                        top_val[0, i]))
        return out_logit
"""


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
    elif args.stream == 'flow':
        streams = ['flow'] 
    else:
        streams = ['rgb', 'flow']

    preds = np.zeros([num_samples, len(action_classes), len(streams)])
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
        model.load_state_dict(torch.load('models/' + args.model + '/checkpoints/' + args.model + '_RGB.pth'))
        model.cuda()


        for idx in range(len(dataset)):
            print ('stream: ' + input_stream + '   idx: ' + str(idx))
            out_var, out_logit  = model( torch.autograd.Variable(torch.from_numpy(dataset[idx]['video'].transpose(0, 4, 1, 2, 3)).cuda()) )

            preds[idx,:,stream_idx] = out_var.data.cpu().numpy()
            if idx % 20 == 0:
                np.savez('out/' + args.model + '/' + args.model + '-' + args.dataset + '-' + args.stream + '.npz',
                         truth_labels=truth_labels, 
                         preds=preds)

        # Clearing GPU cache and clearing model from memory
        torch.cuda.empty_cache()
        model = None
        gc.collect()
        stream_idx += 1

"""
    # Joint model
    if args.stream == 'joint':
        out_logit = out_rgb_logit + out_flow_logit
        out_softmax = torch.nn.functional.softmax(out_logit, 1).data.cpu()
        top_val, top_idx = torch.sort(out_softmax, 1, descending=True)

        print('===== Final predictions ====')
        print('logits proba class '.format(args.top_k))
        for i in range(args.top_k):
            logit_score = out_logit[0, top_idx[0, i]].data[0]
            print('{:.6e} {:.6e} {}'.format(logit_score, top_val[0, i],
                                            action_classes[top_idx[0, i]]))
"""

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
