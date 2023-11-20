import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Video 1

frames = []

base_dir = './Dataset_Student/train/video_0/'

image_names = [f'image_{i}.png' for i in range(22)]

for file_name in image_names:
    img = plt.imread(base_dir + file_name)
    frames.append(img)

mask = np.load(base_dir + 'mask.npy')

class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, num_classes, kernel_size=5, padding = 2)
        
        self.bnorm1 = nn.BatchNorm2d(128)
        self.bnorm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bnorm1(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = self.bnorm2(x)
        x = F.relu(x)
        
        x = self.conv5(x)
        x = F.relu(x)
        
        x = self.conv6(x)
        return x



class CustomDataset(Dataset):
    def __init__(self, all_frames, all_masks):
        self.frames = torch.tensor(all_frames).permute(0, 3, 1, 2)
        self.masks = all_masks #torch.stack([ohe_mask(mask) for mask in all_masks]).permute(0,3,1,2)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        mask = self.masks[idx]
        return frame, mask

# Hyperparameters
num_classes = 49
batch_size = 8
learning_rate = 0.001
num_epochs = 10

# Instantiate the model and set up the optimizer and loss function
model = FCN(num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Create DataLoader
train_dataset = CustomDataset(frames, mask)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
#         masks = masks.argmax(dim=1)
        masks = masks.long()
        print(outputs.shape, masks.shape)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

# Save the trained model if needed
torch.save(model.state_dict(), 'fcn_model.pth')
