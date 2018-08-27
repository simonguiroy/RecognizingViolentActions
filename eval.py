import argparse
import numpy as np
import torch
from models.i3d.i3d import I3D

rgb_pt_checkpoint = 'models/i3d/checkpoints/i3d_RGB.pth'
flow_pt_checkpoint = 'models/i3d/checkpoints/i3d_FLOW.pth'


def run_demo(args):
    kinetics_classes = [x.strip() for x in open(args.classes_path)]

    def get_scores(sample, model):
        sample_var = torch.autograd.Variable(torch.from_numpy(sample).cuda())
        out_var, out_logit = model(sample_var)
        out_tensor = out_var.data.cpu()

        top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

        print(
            'Top {} classes and associated probabilities: '.format(args.top_k))
        for i in range(args.top_k):
            print('[{}]: {:.6E}'.format(kinetics_classes[top_idx[0, i]],
                                        top_val[0, i]))
        return out_logit

    # Rung RGB model
    if args.rgb:
        i3d_rgb = I3D(num_classes=400, modality='rgb')
        i3d_rgb.eval()
        i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
        i3d_rgb.cuda()

        rgb_sample = np.load(args.rgb_sample_path).transpose(0, 4, 1, 2, 3)
        out_rgb_logit = get_scores(rgb_sample, i3d_rgb)

    # Run flow model
    if args.flow:
        i3d_flow = I3D(num_classes=400, modality='flow')
        i3d_flow.eval()
        i3d_flow.load_state_dict(torch.load(args.flow_weights_path))
        i3d_flow.cuda()

        flow_sample = np.load(args.flow_sample_path).transpose(0, 4, 1, 2, 3)
        out_flow_logit = get_scores(flow_sample, i3d_flow)

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
                                            kinetics_classes[top_idx[0, i]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Runs inflated inception v1 network on\
    cricket sample from tensorflow demo (generate the network weights with\
    i3d_tf_to_pt.py first)')


    parser.add_argument(
        '--sample_path', type=str, help='path of video clip sample')


    # RGB arguments
    parser.add_argument(
        '--rgb', action='store_true', help='Evaluate RGB pretrained network')
    parser.add_argument(
        '--rgb_weights_path',
        type=str,
        default='model/model_rgb.pth',
        help='Path to rgb model state_dict')
    parser.add_argument(
        '--rgb_sample_path',
        type=str,
        default='data/kinetic-samples/punching_person-Kinetics-train-0001-RGB-crop.npy',
        help='Path to kinetics rgb numpy sample')

    # Flow arguments
    parser.add_argument(
        '--flow', action='store_true', help='Evaluate flow pretrained network')
    parser.add_argument(
        '--flow_weights_path',
        type=str,
        default='model/model_flow.pth',
        help='Path to flow model state_dict')
    parser.add_argument(
        '--flow_sample_path',
        type=str,
        default='data/kinetic-samples/punching_person-Kinetics-train-0001-FLOW-crop.npy',
        help='Path to kinetics flow numpy sample')

    parser.add_argument(
        '--classes_path',
        type=str,
        default='data/kinetic-samples/label_map.txt',
        help='Number of video_frames to use (should be a multiple of 8)')
    parser.add_argument(
        '--top_k',
        type=int,
        default='5',
        help='When display_samples, number of top classes to display')
    args = parser.parse_args()

    args.rgb_sample_path = args.sample_path + '-RGB-crop.npy'
    args.rgb_flow_path = args.sample_path + '-FLOW-crop.npy'


    run_demo(args)
