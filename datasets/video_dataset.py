import cv2
import torch
from torchvision import datasets, transforms
import numpy as np
from numpy import genfromtxt
from torch.utils.data import Dataset
import time
from datasets.preprocessing import preprocess_rgb, preprocess_flow
from torch.utils.data import DataLoader


# For now, this dataset class should only be used to evaluate a model, and not for training. Need to test support for DataLoader shuffle

"""
Since the numpy array generated from pre-processing the video files are typically very large (between 50Mb and 100Mb),
the dataset will only output either a numpy array for the RGB stream or the FLOW stream, but not both at the same time,
as we want to avoid creating huge variables as much as possible where this call object will be called (train or valid script)    

We also want to avoid as much as possible computing unnecessary pre-processing, thus the pre-processing was implemented in
two different functions, even if they easily could have been mergeg.
"""

class VideoDataset(Dataset):
    """Dataset of video sequences, pre-processed as numpy arrays for rgb and optical-flow stream"""

    def __init__(self,
                 csv_file, 
                 root_dir,
                 stream='rgb',
                 split='train',
                 transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the video files.
            stream (string): Type of stream for model input (rgb or flow)
            split (string): Split of the dataset to take samples from [train/valid/test/all]
            transform (callable, optional): Optional transform to be applied after preprocessing, like random croping.
        """
        self.csv_data = genfromtxt(csv_file, delimiter=',', dtype=str)
        self.root_dir = root_dir
        self.stream = stream
        self.transform = transform

    def __len__(self):
        return self.csv_data.shape[0]

    #TODO: Here we add this function to avoid creating huge intermediate variables in the calling script.
    # will need to make the __getitem__() function coherent with this function and eventually with the use
    # a DataLoader object 
    def get_labels(self):
        return self.csv_data[:][1]
        
        
    def __getitem__(self, idx):
        video_filename = self.csv_data[idx][0]
        label = self.csv_data[idx][1]
        sample = {'video': None, 'label': label}
        
        if self.stream == 'rgb':
            sample['video'] = preprocess_rgb(self.root_dir + video_filename)

        if self.stream == 'flow':
            sample['video'] = preprocess_flow(self.root_dir + video_filename)

        return sample
