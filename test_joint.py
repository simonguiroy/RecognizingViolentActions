import numpy as np
from video_dataset import VideoDataset
import torch
import csv
import argparse


def run(args):

    # List of action class labels
    action_classes = [x.strip() for x in open('models/i3d/label_map.txt')]

    rgb_outputs = torch.load(args.rgb_preds_path)
    flow_outputs = torch.load(args.flow_preds_path)

    rgb_preds = rgb_outputs['preds']
    flow_preds = flow_outputs['preds']

    rgb_indices = rgb_outputs['shuffled_indices']
    flow_indices = flow_outputs['shuffled_indices']

    rgb_pred_indices = rgb_indices[0:len(rgb_preds)]
    flow_pred_indices = flow_indices[0:len(flow_preds)]

    common_pred_indices = list(set(rgb_pred_indices.tolist()).intersection(set(flow_pred_indices.tolist())))

    test_dataset = VideoDataset(root_dir='datasets/' + args.dataset, split='test',
                                 max_frames_per_clip=-1)

    target_labels = test_dataset.get_labels()

    running_corrects_rgb = 0
    running_corrects_flow = 0
    running_corrects_joint = 0

    for video_idx in common_pred_indices:
        rgb_pred = rgb_preds[np.where(rgb_indices.numpy() == video_idx)[0][0]]
        flow_pred = flow_preds[np.where(flow_indices.numpy() == video_idx)[0][0]]

        label = target_labels[video_idx]
        target = torch.tensor([action_classes.index(label)], dtype=torch.long)

        joint_pred = 0.5 * (rgb_pred + flow_pred)

        running_corrects_rgb += target.data.cpu().item() == torch.argmax(rgb_pred).cpu().item()
        running_corrects_flow += target.data.cpu().item() == torch.argmax(flow_pred).cpu().item()
        running_corrects_joint += target.data.cpu().item() == torch.argmax(joint_pred).cpu().item()
        print("Target: " + label + " Predictions --  [RGB]: " + action_classes[torch.argmax(rgb_pred)] + " [FLOW]: " +
              action_classes[torch.argmax(flow_pred)] + " [JOINT]: " + action_classes[torch.argmax(joint_pred)])

    rgb_acc = running_corrects_rgb / len(common_pred_indices)
    flow_acc = running_corrects_flow / len(common_pred_indices)
    joint_acc = running_corrects_joint / len(common_pred_indices)

    print("*****************************************")
    print("Test Completed! Accuracies --  [RGB]: " + str(rgb_acc) + " [FLOW]: " +
          str(flow_acc) + " [JOINT]: " + str(joint_acc))
    with open('out/i3d/logs/' + args.output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['RGB_Acc', 'Flow_Acc', 'Joint_Acc'])
        writer.writerow([rgb_acc, flow_acc, joint_acc])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recognizing Violent Human Actions')
    parser.add_argument('--rgb_preds_path', type=str, help='Path to .pkl file with RGB predictions')
    parser.add_argument('--flow_preds_path', type=str, help='Path to .pkl file with Optical Flow predictions')
    parser.add_argument('--dataset', type=str, default='ViolentHumanActions_v2', help='Name of dataset to use')
    parser.add_argument('--output_file', type=str, help='Name of .csv output file to save test performance results')
    args = parser.parse_args()

    run(args)
