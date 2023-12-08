# Need to separate all files into function defintions and main.py part
from models import *
from utils import *
from datasets import Combined_Pipeline_Dataset

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()


def load_weights(model):
    best_model_path = './checkpoints/frame_prediction.pth'
    if os.path.isfile(best_model_path):
        print('frame prediction weights found')
        model.module.frame_prediction_model.load_state_dict(torch.load(best_model_path))

    best_model_path = './checkpoints/image_segmentation.pth'
    if os.path.isfile(best_model_path):
        print('image segmentation weights found')
        model.module.image_segmentation_model.load_state_dict(torch.load(best_model_path))

    # best_model_path = './checkpoints/combined_model.pth'
    # if os.path.isfile(best_model_path):
    #     print('combined model weights found')
    #     model.load_state_dict(torch.load(best_model_path))


def save_weights(model):
    torch.save(model.module.frame_prediction_model.state_dict(), './checkpoints/frame_prediction.pth')
    torch.save(model.module.image_segmentation_model.state_dict(), './checkpoints/image_segmentation.pth')
    # torch.save(model.state_dict(), './checkpoints/combined_model.pth')
    print('model weights saved successfully')


# Create Train DataLoader
batch_size = 5
num_videos = 1000
train_data = Combined_Pipeline_Dataset(num_videos)
# load the data.
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


batch_size = 5
num_videos = 10
val_data = Combined_Pipeline_Dataset(num_videos, evaluation_mode = True)
# load the data.
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


# Hyperparameters:
num_epochs = 20
lr = 0.00001
weight_decay = 1e-8
momentum = 0.999

model = combined_model(device)
model = nn.DataParallel(model)
load_weights(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)#, foreach=True)
grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

train_losses = []
preds_per_epoch = []
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    train_loss = []
    
    train_pbar = tqdm(train_loader)

    for batch_x, batch_y in train_pbar:
        model.train()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
        pred_y = model(batch_x)  # .long()
        loss = criterion(pred_y, batch_y)
        train_loss.append(loss.item())
        train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
        #         print(loss)
        optimizer.zero_grad()
#         optimizer.step()
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        score = evaluate(model, val_loader, device)
        scheduler.step(score)

    train_loss = np.average(train_loss)
    print(f"Average train loss {train_loss}")
    train_losses.append(train_loss)
    save_weights(model)