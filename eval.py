import argparse
import os
import numpy as np
from datasets.video_dataset import VideoDataset
import torch
from models import *
import sys

#rgb_pt_checkpoint = 'models/i3d/checkpoints/i3d_RGB.pth'
#flow_pt_checkpoint = 'models/i3d/checkpoints/i3d_FLOW.pth'


def run_demo(args):

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


    action_classes = [x.strip() for x in open(args.classes_path)]

    if args.model == 'i3d':
        model_class = I3D
    else:
        pass

    # Run RGB model
    if args.stream == 'rgb' or args.stream == 'joint':
        dataset_rgb = VideoDataset(csv_file='datasets/' + args.dataset + '/dataset.csv',
                               root_dir='datasets/' + args.dataset + '/data/',
                               stream='rgb')
        model_rgb = model_class(num_classes=len(action_classes), modality='rgb')
        model_rgb.eval()
        model_rgb.load_state_dict(torch.load('models/' + args.model + '/checkpoints/' + args.model + '_RGB.pth'))
        model_rgb.cuda()
    # Run flow model
    if args.stream == 'flow' or args.stream == 'joint':
        dataset_flow = VideoDataset(csv_file='datasets/' + args.dataset + '/dataset.csv',
                               root_dir='datasets/' + args.dataset + '/data/',
                               stream='flow')
        model_flow = model_class(num_classes=len(action_classes), modality='flow')
        model_flow.eval()
        model_flow.load_state_dict(torch.load('models/' + args.model + '/checkpoints/' + args.model + '_FLOW.pth'))
        model_flow.cuda()

    for idx in range(len(dataset)):
        if args.stream == 'rgb' or args.stream == 'joint':
            out_var_rgb, out_logit_rgb  = model_rgb( torch.autograd.Variable(torch.from_numpy(dataset_rgb[idx]['video'].transpose(0, 4, 1, 2, 3)).cuda()) )
            out_tensor_rgb = out_var_rgb.data.cpu()

        if args.stream == 'flow' or args.stream == 'joint':
            out_var_flow, out_logit_flow  = model_flow( torch.autograd.Variable(torch.from_numpy(dataset_flow[idx]['video'].transpose(0, 4, 1, 2, 3)).cuda()) )
            out_tensor_rgb = out_var_rgb.data.cpu()





    # Joint model
    if args.flow and args.rgb:
        out_logit = out_rgb_logit + out_flow_logit
        out_softmax = torch.nn.functional.softmax(out_logit, 1).data.cpu()
        top_val, top_idx = torch.sort(out_softmax, 1, descending=True)

        print('===== Final predictions ====')
        print('logits proba class '.format(args.top_k))
        for i in range(args.top_k):
            logit_score = out_logit[0, top_idx[0, i]].data[0]
            print('{:.6e} {:.6e} {}'.format(logit_score, top_val[0, i],
                                            action_classes[top_idx[0, i]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluating the I3D model, and possible variants, on video datasets')
    # TODO: eventually, we will want to evaluate models on various possible checkpoints, thus add an argument for specific checkpoints

    parser.add_argument(
        '--model',
        type=str,
        default='i3d',
        choices=['i3d']
        help='Model to use')
    parser.add_argument(
        '--stream',
        type=str,
        default='joint',
        choices=['rgb', 'flow', 'joint']
        help='Input stream for model')
    parser.add_argument(
        '--pre-trained',
        type=str,
        default='both',
        #choices=['rgb', 'flow', 'both', 'none']
        # Since this is an evaluation script, we have to load pre-trained weights, but use the line above for a training script
        choices=['both']
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
