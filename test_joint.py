import numpy as np
from video_dataset import VideoDataset
import torch
from models import *
import csv
import argparse


def run(args):

    # List of action class labels
    action_classes = [x.strip() for x in open('models/i3d/label_map.txt')]

    rgb_outputs = torch.load(args.rgb_preds_file)
    flow_outputs = torch.load(args.flow_preds_file)

    rgb_preds = rgb_outputs['preds']
    flow_preds = flow_outputs['preds']

    rgb_indices = rgb_outputs['shuffled_indices']
    flow_indices = flow_outputs['shuffled_indices']

    test_dataset = VideoDataset(root_dir='datasets/' + args.dataset, split='test', stream=args.stream,
                                 max_frames_per_clip=-1)

    target_labels = test_dataset.get_labels()

    running_corrects_rgb = 0
    running_corrects_flow = 0
    running_corrects_joint = 0
    for idx in range(rgb_indices):

        rgb_pred = rgb_preds[idx]
        rgb_index = rgb_indices[idx]
        flow_index = np.where(flow_indices.numpy() == rgb_index)[0][0]
        flow_pred = flow_preds[flow_index]

        label = target_labels[rgb_index]
        target = torch.tensor([action_classes.index(label)], dtype=torch.long)

        joint_pred = 0.5 * (rgb_pred + flow_pred)

        running_corrects_rgb += target.data.cpu().item() == torch.argmax(rgb_pred).cpu().item()
        running_corrects_flow += target.data.cpu().item() == torch.argmax(flow_pred).cpu().item()
        running_corrects_joint += target.data.cpu().item() == torch.argmax(joint_pred).cpu().item()
        print("Target: " + label + " Predictions --  [RGB]: " + action_classes[torch.argmax(rgb_pred)] + " [FLOW]: " +
              action_classes[torch.argmax(flow_pred)] + " [JOINT]: " + action_classes[torch.argmax(joint_pred)])

    rgb_acc = running_corrects_rgb / len(test_dataset)
    flow_acc = running_corrects_flow / len(test_dataset)
    joint_acc = running_corrects_joint / len(test_dataset)

    print("*****************************************")
    print("Test Completed! Accuracies --  [RGB]: " + rgb_acc + " [FLOW]: " +
          flow_acc + " [JOINT]: " + joint_acc)
    with open('out/i3d/logs/joint-test_' + args.rgb_preds_file + '___' + args.flow_preds_file + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['RGB_Acc', 'Flow_Acc', 'Joint_Acc'])
        writer.writerow([rgb_acc, flow_acc, joint_acc])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recognizing Violent Human Actions')
    parser.add_argument('--rgb_preds_file', type=str, help='Filepath to RGB predictions')
    parser.add_argument('--flow_preds_file', type=str, help='Filepath to Optical Flow predictions')
    args = parser.parse_args()

    run(args)
