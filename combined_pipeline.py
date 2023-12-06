# Need to separate all files into function defintions and main.py part
from models import *

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
        num_total_frames = 22
        x = []
        i = self.vid_indexes[idx]
        
#         # Isha
#         base_dir = './../Dataset_Student/'
        
        # Ansh
        base_dir = './../../../scratch/ar7964/dataset_videos/dataset/'
        
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
    
    
# Dataloader
batch_size = 8

# Create Train DataLoader
num_videos = 1000
train_data = CustomDataset(num_videos)
# load the data.
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# # Create Val DataLoader
# num_val_videos = 1000
# val_data = CustomDataset(num_val_videos, evaluation_mode = True)
# # load the data.
# val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

gpu_name = 'cuda'
device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')

class combined_model(nn.Module):
    def __init__(self, device):
        super(combined_model, self).__init__()
        self.frame_prediction_model = DLModelVideoPrediction((11, 3, 160, 240), 64, 512, groups=4)
        self.frame_prediction_model = nn.DataParallel(self.frame_prediction_model)
        self.frame_prediction_model = self.frame_prediction_model.to(device)

        self.image_segmentation_model = unet_model()
        self.image_segmentation_model = nn.DataParallel(self.image_segmentation_model)
        self.image_segmentation_model = self.image_segmentation_model.to(device)
        
    def load_weights(self):
        best_model_path = './checkpoints/frame_prediction.pth'  # load saved model to restart from previous best model
        if os.path.isfile(best_model_path):
            print('frame prediction model weights found')
            self.frame_prediction_model.load_state_dict(torch.load(best_model_path))

        best_model_path = './checkpoints/image_segmentation.pth'  # load saved model to restart from previous best model
        if os.path.isfile(best_model_path):
            print('image segmentation model weights found')
            self.image_segmentation_model.load_state_dict(torch.load(best_model_path))
            
    def save_weights(self):
        torch.save(self.frame_prediction_model.state_dict(), './checkpoints/frame_prediction.pth')
        torch.save(self.frame_prediction_model.state_dict(), './checkpoints/image_segmentation.pth')
        print('model weights saved successfully')
        
        
    def forward(self,x):
        x = self.frame_prediction_model(x)
#         print(x.shape)
        x = x[:,-1]
#         print(x.shape)
        x = self.image_segmentation_model(x)
#         print(x.shape)
        return x

# Hyperparameters:
num_epochs = 10
lr = 0.0001
model = combined_model(device)
model.load_weights()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader),
                                                epochs=num_epochs)

train_losses = []
preds_per_epoch = []
for epoch in range(num_epochs):
    train_loss = []
    model.train()
    train_pbar = tqdm(train_loader)

    for batch_x, batch_y in train_pbar:
        optimizer.zero_grad()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
        pred_y = model(batch_x)#.long()
        loss = criterion(pred_y, batch_y)
        train_loss.append(loss.item())
        train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
#         print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

    train_loss = np.average(train_loss)
    print(f"Average train loss {train_loss}")
    train_losses.append(train_loss)
#     torch.save(model.state_dict(), './checkpoint_frame_prediction.pth')
    model.save_weights()

    
#     val_loss = []
#     model.eval()
#     val_pbar = tqdm(val_loader)

#     with torch.no_grad():
#         if epoch % 2 == 0:
#             for batch_x in val_pbar:
#                 batch_x = batch_x.to(device)
#                 pred_y = model(batch_x).float()
#                 preds_per_epoch.append(pred_y)
                
# latest_predictions = preds_per_epoch[-1]
# torch.save(latest_predictions, 'The_Big_Epochalypse_submission.pt')