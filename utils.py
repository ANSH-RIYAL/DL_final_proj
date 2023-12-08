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

def get_blurry_images(x_batch, y_batch, noise_level = 0.1):
    # Shape of x_batch is (batch_size, num_channels, height, width)
    # Shape of x_augmented is (batch_size * 6, num_channels, height, width)
    
#     print(x_batch.shape, y_batch.shape)
    
    x_random_noise = x_batch.clone()
    noise = torch.randn_like(x_random_noise) * noise_level
    x_random_noise += noise
        
    kernels = [5,9]
    sigmas = [1,5]
    x_gaussian_blurring = []
    x_g_k_s = x_batch.clone()
    for kernel_size in kernels:
        for sigma in sigmas:
            blur_transform = GaussianBlur(kernel_size, sigma)
            x_gaussian_blurring.append(blur_transform(x_g_k_s))
    x_gaussian_blurring = torch.cat(x_gaussian_blurring,0)
    
    x_augmented = torch.cat([x_batch, x_random_noise, x_gaussian_blurring],0)
    
    n_y = 1 + x_random_noise.shape[0]//int(x_batch.shape[0]) + x_gaussian_blurring.shape[0]//int(x_batch.shape[0])
    y_augmented = torch.cat([y_batch for i in range(n_y)],0)
    
#     print(x_augmented.shape, y_augmented.shape)

    return x_augmented, y_augmented
    

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
    return dice_score / min(num_val_batches, 5)
