from numpy import genfromtxt
from torch.utils.data import Dataset
#from preprocessing import preprocess_rgb, preprocess_flow
from preprocessing import preprocess_video
import torch


class VideoDataset(Dataset):
    """Dataset of video sequences, pre-processed as numpy arrays for rgb and optical-flow stream"""

    def __init__(self,root_dir, stream='rgb', split='train', max_frames_per_clip=-1):
        """
        :param root_dir: Root directory of the dataset
        :param stream: Type of stream for model input [rgb/flow]
        :param split: Split of the dataset to take samples from [train/valid/test/all]
        :param max_frames_per_clip: Maximujm number of frames to process per clip. If -1, process entire clip.
        """
        self.csv_data = genfromtxt(root_dir + '/dataset_' + split + '.csv', delimiter=',', dtype=str)
        self.root_dir = root_dir
        self.stream = stream
        self.split = split
        self.max_frames_per_clip = max_frames_per_clip

    def __len__(self):
        return self.csv_data.shape[0]

    def get_labels(self):
        return self.csv_data[:, 0]

    def __getitem__(self, idx):
        video_filename = self.csv_data[idx][1]
        label = self.csv_data[idx][0]
        video = preprocess_video(self.root_dir + '/data/' + self.split + '/' + label + '/' + video_filename,
                                 self.stream, max_frames_per_clip=self.max_frames_per_clip)

        return label, video


def my_collate(batch):
    """
    This custom collate function returns the batch where videos and labels are sorted in a list. This allows to have a
    batch with videos of variable number of frames. This function is used by the dataloader internally.
    :param batch: The batch returned by the dataset.
    :return: A single batch (list), where the first element is a list of labels, and the second is a list of videos
    """
    labels = [item[0] for item in batch]
    videos = [item[1] for item in batch]
    return [labels, videos]


class DeterministicSubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement. To enable reproducibility, list
    of randomly shuffled indices if created during __init__().

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.shuffled_indices = torch.randperm(len(indices))

    def __iter__(self):
        return (self.shuffled_indices[i] for i in range(len(self.shuffled_indices)))

    def __len__(self):
        return len(self.shuffled_indices)
