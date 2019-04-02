import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Recognizing Violent Human Actions')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=5, help='Batch_size for training.')
    parser.add_argument('--max_frames_per_clip', type=int, default=60, help='Maximum number of frames per clip.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum to use with SGD optimizer.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number training epoch.')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Resume training at given epoch.')
    parser.add_argument('--multistep_lr_schedule', type=list, default=[],
                        help='Initial lr decayed by gamma once '
                             'the number of epoch reaches one of the milestones. '
                             'eg. [gamma, [milestone1, milestone2, milestone2]]. '
                             'See doc for torch.optim.lr_scheduler.MultiStepLR. If left empty, lr is left fixed.')

    parser.add_argument('--seed', type=int, default=0, help='Seed for generating random numbers')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data providing workers.')

    parser.add_argument('--stream', type=str, default='rgb', choices=['rgb', 'flow'],
                        help='Input stream.')
    parser.add_argument('--dataset', type=str, default='ViolentHumanActions_v2', help='Name of dataset to use')
    parser.add_argument('--device_type', type=str, default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--gpu_id', type=int, default=0, help='Identifier for GPU to use. ')

    args = parser.parse_args()

    return args
