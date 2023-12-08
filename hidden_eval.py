# Need to separate all files into function defintions and main.py part
from models import *

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()


# Data Loader.
class CustomDataset(Dataset):
    def __init__(self, num_of_vids=1000, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode
        if self.evaluation_mode == 'hidden':
            self.mode = 'hidden'
            start_num = 15000
        elif self.evaluation_mode == 'val':
            self.mode = 'val'
            start_num = 1000
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

        if self.evaluation_mode == 'hidden':
            return x

        file_path = f"{base_dir}{self.mode}/video_{i}/mask.npy"
        y = np.load(file_path)[21]  # last frame.
        return x, y


def load_weights(model):
    best_model_path = './checkpoints/frame_prediction.pth'
    # best_model_path = './../../checkpoint_frame_predictione13.pth'
    if os.path.isfile(best_model_path):
        print('frame prediction weights found')
        model.module.frame_prediction_model.load_state_dict(torch.load(best_model_path))

    best_model_path = './checkpoints/image_segmentation.pth'
    # best_model_path = './../../image_segmentation_good.pth'
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
        x = x[:, -1]
        x = self.image_segmentation_model(x)
        return x


# Create Val DataLoader
batch_size = 8
num_val_videos = 1000
val_data = CustomDataset(num_val_videos, evaluation_mode='val')
val_loader = DataLoader(val_data, batch_size=batch_size)

hidden = False
# num_hidden_videos = 2000
# hidden_data = CustomDataset(num_hidden_videos, evaluation_mode='hidden')
# hidden_loader = DataLoader(hidden_data, batch_size=batch_size)
# hidden_pbar = tqdm(hidden_loader)

model = combined_model(device)
model = nn.DataParallel(model)
load_weights(model)


val_loss = []
model.eval()
val_pbar = tqdm(val_loader)

with torch.no_grad():
    if not hidden:
        preds = []
        total_y = []
        for batch_x, batch_y in val_pbar:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            batch_out = torch.argmax(out.detach().cpu(), dim=1)
            preds.append(batch_out)
            total_y.append(batch_y)
        preds = torch.cat(preds, dim=0)
        ground_truth = torch.cat(total_y, dim=0)
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
        final_iou = jaccard(preds, ground_truth)
        print(preds.shape, ground_truth.shape)
        print("Final iou on val", final_iou)
    else:
        preds = []
        for batch_x in hidden_pbar:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            batch_out = torch.argmax(out.detach().cpu(), dim=1)
            preds.append(batch_out)

        preds = torch.cat(preds, dim=0)
        print(preds.shape)
        torch.save(preds, 'leaderboard_2_team_13.pt')