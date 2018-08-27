import torch
from torchvision import datasets, transforms
import numpy as np
from numpy import genfromtxt
from torch.utils.data import Dataset
import cv2
import time
from preprocessing import preprocess_rgb, preprocess_flow
from torch.utils.data import DataLoader


class VideoDataset(Dataset):
    """Dataset of video sequences, pre-processed as Torch tensors for rgb and optical-flow stream"""

    def __init__(self,
                 csv_file, 
                 root_dir,
                 stream='rgb',
                 transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_data = genfromtxt(csv_file, delimiter=',', dtype=str)
        self.root_dir = root_dir
        self.stream = stream
        self.transform = transform

    def __len__(self):
        return self.csv_data.shape[0]
        
        
    def __getitem__(self, idx):
        video_id = self.csv_data[idx][0]
        label = self.csv_data[idx][1]
        sample = {'videos': {'video_rgb': torch.zeros(224, 224), 'video_flow': torch.randn(224, 224)}, 'label': label}
        
        if self.stream == 'rgb' or self.stream == 'joint':
            sample['videos']['video_rgb'] = torch.from_numpy(preprocess_rgb(self.root_dir + video_id + '.mp4'))
        if self.stream == 'flow' or self.stream == 'joint':
            sample['videos']['video_flow'] = preprocess_flow(self.root_dir + video_id + '.mp4')
            
        return sample
