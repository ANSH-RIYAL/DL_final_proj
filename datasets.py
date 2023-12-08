import os
import sys

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import GaussianBlur
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

# Ansh
# base_dir = './../../../scratch/ar7964/dataset_videos/dataset/'
base_dir = './../Dataset_Student/'

# # Shreemayi
# base_dir = './../../dataset/'

# # Isha
# base_dir = ''

class Segmentation_Mask_Dataset(Dataset):
    def __init__(self, all_frames):
        self.frames = all_frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        global base_dir        
        i, j = self.frames[idx]
        file_path = f"{base_dir}train/video_{i}/image_{j}.png"
        frame = torch.tensor(plt.imread(file_path)).permute(2, 0, 1)

        file_path = f"{base_dir}train/video_{i}/mask.npy"
        mask = np.load(file_path)[j]
        return frame, mask

    
class Frame_Prediction_Dataset(Dataset):
    def __init__(self, num_of_vids, evaluation_mode=False):
        if evaluation_mode:
            start_num = 1000
        else:
            start_num = 2000
        self.vid_indexes = torch.tensor([i for i in range(start_num, num_of_vids + start_num)])
        self.evaluation_mode = evaluation_mode

    def __getitem__(self, idx):
        global base_dir
        
        num_hidden_frames = 11
        num_total_frames = 22
        x = []
        y = []
        i = self.vid_indexes[idx]
        if self.evaluation_mode:
            mode = 'val'
        else:
            mode = 'unlabeled'
        filepath = f'{base_dir}{mode}/video_{i}/'
        # obtain x values.
        for j in range(num_hidden_frames):
            x.append(torch.tensor(plt.imread(filepath + f'image_{j}.png')).permute(2, 0, 1))
        x = torch.stack(x, 0)
        for j in range(num_hidden_frames, num_total_frames):
            y.append(torch.tensor(plt.imread(filepath + f'image_{j}.png')).permute(2, 0, 1))
        y = torch.stack(y, 0)
        return x, y

    def __len__(self):
        vid_len = len(self.vid_indexes)
        return vid_len
    
    
    
class Combined_Pipeline_Dataset(Dataset):
    def __init__(self, num_of_vids=1000, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode
        if self.evaluation_mode == True:
            self.mode = 'val'
            start_num = 1000
        elif self.evaluation_mode == 'hidden':
            self.mode = 'hidden'
            start_num = 15000
        else:
            self.mode = 'train'
            start_num = 0
        self.vid_indexes = torch.tensor([i for i in range(start_num, num_of_vids + start_num)])
        self.num_of_vids = num_of_vids

    def __len__(self):
        return self.num_of_vids

    def __getitem__(self, idx):
        global base_dir
        
        num_hidden_frames = 11
        num_total_frames = 22
        x = []
        i = self.vid_indexes[idx]

        filepath = f'{base_dir}{self.mode}/video_{i}/'
        # obtain x values.
        for j in range(num_hidden_frames):
            x.append(torch.tensor(plt.imread(filepath + f'image_{j}.png')).permute(2, 0, 1))
        x = torch.stack(x, 0)

        if self.evaluation_mode == 'hidden':
            return x

        file_path = f"{base_dir}{self.mode}/video_{i}/mask.npy"
        y = np.load(file_path)[21]  # last frame.
        return x, y

