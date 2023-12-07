# Need to separate all files into function defintions and main.py part
from models import *

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()


# Data Loader.
class CustomDataset(Dataset):
    def __init__(self, num_of_vids=1000, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode
        if self.evaluation_mode:
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
        num_hidden_frames = 11
        x = []
        i = self.vid_indexes[idx]
        # # Isha
        # base_dir = './../Dataset_Student/'
        # Shreemayi
        base_dir = './../../dataset/'
        # # Ansh
        # base_dir = './../../../scratch/ar7964/dataset_videos/dataset/'

        filepath = f'{base_dir}{self.mode}/video_{i}/'
        # obtain x values.
        for j in range(num_hidden_frames):
            x.append(torch.tensor(plt.imread(filepath + f'image_{j}.png')).permute(2, 0, 1))
        x = torch.stack(x, 0)

        if self.evaluation_mode:
            return x

        file_path = f"{base_dir}{self.mode}/video_{i}/mask.npy"
        y = np.load(file_path)[21]  # last frame.
        return x, y


def load_weights(model):
    best_model_path = './checkpoints/frame_prediction.pth'
    if os.path.isfile(best_model_path):
        print('frame prediction weights found')
        model.module.frame_prediction_model.load_state_dict(torch.load(best_model_path))

    best_model_path = './checkpoints/image_segmentation.pth'
    if os.path.isfile(best_model_path):
        print('image segmentation weights found')
        model.module.image_segmentation_model.load_state_dict(torch.load(best_model_path))


class combined_model(nn.Module):
    def __init__(self, device):
        super(combined_model, self).__init__()
        self.frame_prediction_model = DLModelVideoPrediction((11, 3, 160, 240), 64, 512, groups=4)
        self.frame_prediction_model = self.frame_prediction_model.to(device)
        self.frame_prediction_model = nn.DataParallel(self.frame_prediction_model)
#         self.image_segmentation_model = unet_model()
        self.image_segmentation_model = UNet(bilinear=True)
        self.image_segmentation_model = self.image_segmentation_model.to(device)
        self.image_segmentation_model = nn.DataParallel(self.image_segmentation_model)

    def forward(self, x):
        x = self.frame_prediction_model(x)
        #         print(x.shape)
        x = x[:, -1]
        #         print(x.shape)
        x = self.image_segmentation_model(x)
        #         print(x.shape)
        return x


# Create Val DataLoader
batch_size = 8
num_val_videos = 2000
val_data = CustomDataset(num_val_videos, evaluation_mode=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

model = combined_model(device)
model = nn.DataParallel(model)
load_weights(model)


val_loss = []
model.eval()
val_pbar = tqdm(val_loader)

with torch.no_grad():
    preds = []
    for batch_x in val_pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        out = model(batch_x)
        batch_out = np.argmax(out.detach().cpu().numpy(), axis=1)
        preds.append(batch_out)
    preds = np.concatenate(preds, axis=0)

print(preds.shape)
torch.save(preds, 'leaderboard_2_team_13.pt')