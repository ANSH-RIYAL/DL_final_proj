import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from models import *
from datasets import Frame_Prediction_Dataset

import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()

    
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM()

    def forward(self, pred, gt):
        mse_loss = nn.MSELoss()(pred, gt)
        perceptual_loss = self.ssim(pred.view(-1, 3, 160, 240), gt.view(-1, 3, 160, 240))
        return mse_loss + perceptual_loss


batch_size = 8
num_videos = 13000
num_val_videos = 1000

train_data = Frame_Prediction_Dataset(num_videos)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data = Frame_Prediction_Dataset(num_val_videos, evaluation_mode=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

model = DLModelVideoPrediction((11, 3, 160, 240), 64, 512, groups=4)
model = nn.DataParallel(model)
model = model.to(device)

# Training Loop:
best_model_path = './checkpoints/frame_prediction.pth'  # load saved model to restart from previous best model
if os.path.isfile(best_model_path):
    model.load_state_dict(torch.load(best_model_path))

num_epochs = 5
lr = 0.0001
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader),
                                                epochs=num_epochs)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = []
    model.train()
    train_pbar = tqdm(train_loader)

    for batch_x, batch_y in train_pbar:
        optimizer.zero_grad()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        train_loss.append(loss.item())
        train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        loss.backward()
        optimizer.step()
        scheduler.step()

    train_loss = np.average(train_loss)
    print(f"Average train loss {train_loss}")
    train_losses.append(train_loss)
    torch.save(model.state_dict(), './checkpoints/frame_prediction.pth')

    val_loss = []
    model.eval()
    val_pbar = tqdm(val_loader)

    with torch.no_grad():
        if epoch % 2 == 0:
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred_y = model(batch_x)
                loss = criterion(pred_y, batch_y)
                val_loss.append(loss.item())
                val_pbar.set_description('val loss: {:.4f}'.format(loss.item()))
                torch.cuda.empty_cache()
            val_loss = np.average(val_loss)
            print(f'Average val loss {val_loss}')
            val_losses.append(val_loss)