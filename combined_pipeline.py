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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()


# Data Loader.
class CustomDataset(Dataset):
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
        num_hidden_frames = 11
        num_total_frames = 22
        x = []
        i = self.vid_indexes[idx]

        #         # Isha
        #         base_dir = './../Dataset_Student/'
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

def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False,
                          epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def evaluate(net, dataloader, device):#, amp=True):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
#     with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
    count = 0
    for x_batch, y_batch in tqdm(dataloader, total=num_val_batches, desc='Evaluation round', unit='batch', leave=False):

        x_batch, y_batch = x_batch.to(device), y_batch.to(device).long()
        # predict the mask
        mask_pred = net(x_batch)

        y_batch = F.one_hot(y_batch, 49).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), 49).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        dice_score += multiclass_dice_coeff(mask_pred[:, 1:], y_batch[:, 1:], reduce_batch_first=False)
        count += 1
        if count >=5:
            break

#     net.train()
    return dice_score / max(num_val_batches, 5)


# Create Train DataLoader
batch_size = 5
num_videos = 1000
train_data = CustomDataset(num_videos)
# load the data.
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


batch_size = 5
num_videos = 10
val_data = CustomDataset(num_videos, evaluation_mode = True)
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