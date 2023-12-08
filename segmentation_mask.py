import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import GaussianBlur

import sys

from models import *
from utils import *
from datasets import Segmentation_Mask_Dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# device = torch.device('cpu')
# print(device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# print(device)

torch.cuda.empty_cache()


def load_weights(model):
    best_model_path = './checkpoints/image_segmentation.pth'
    if os.path.isfile(best_model_path):
        print('image segmentation weights found')
        model.load_state_dict(torch.load(best_model_path))


def save_weights(model):
    if 'checkpoints' not in os.listdir():
        os.mkdir('checkpoints')
    torch.save(model.state_dict(), './checkpoints/image_segmentation.pth')
    print('model weights saved successfully')


# Create Train DataLoader
batch_size = 8
num_videos = 1000

num_frames_per_video = 22

all_frames = [[[i, j] for j in range(num_frames_per_video)] for i in range(num_videos)]
t = []
for i in all_frames:
    t += i
all_frames = torch.tensor(t)

# 22000 X 2

train_data = Segmentation_Mask_Dataset(all_frames)
# load the data.
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Hyperparameters:
num_epochs = 10
lr = 0.00001
weight_decay = 1e-8
momentum = 0.999
model = UNet(bilinear=True)
model = model.to(device)
model = nn.DataParallel(model)
load_weights(model)
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr)
optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, foreach=True)
grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=len(train_loader),
#                                                 epochs=num_epochs)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

train_losses = []
preds_per_epoch = []
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    train_loss = []
    model.train()
    train_pbar = tqdm(train_loader)

    for batch_x, batch_y in train_pbar:
        #         optimizer.zero_grad()
        batch_x, batch_y = get_blurry_images(batch_x,batch_y)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
        pred_y = model(batch_x)  # .long()
        loss = criterion(pred_y, batch_y)
        loss += dice_loss(
            F.softmax(pred_y, dim=1).float(),
            F.one_hot(batch_y, model.module.n_classes).permute(0, 3, 1, 2).float(),
            multiclass=True
        )
        train_loss.append(loss.item())
        train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
        #         print(loss)
        #         loss.backward()
        optimizer.zero_grad(set_to_none=True)
        #         optimizer.step()
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        score = evaluate(model, train_loader, device)
        scheduler.step(score)

    train_loss = np.average(train_loss)
    print(f"Average train loss {train_loss}")
    train_losses.append(train_loss)
    save_weights(model)