import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys


class CustomDataset(Dataset):
    def __init__(self, all_frames):
        self.frames = torch.tensor(all_frames)
#         self.masks = all_masks.cuda()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        global net_id
        i,j = self.frames[idx]
        file_path = f"./../../../scratch/{net_id}/dataset_videos/dataset/train/video_{i}/image_{j}.png"
        frame = torch.tensor(plt.imread(file_path)).permute(2, 0, 1)

        file_path = f"./../../../scratch/{net_id}/dataset_videos/dataset/train/video_{i}/mask.npy"
        mask = np.load(file_path)[j]
        return frame, mask


net_id = sys.argv[1]
# net_id = 'ar7964'

all_sequences = []
all_sequence_masks = []

num_videos = 1000
num_frames_per_video = 22

all_frames = [[[i,j] for j in range(num_frames_per_video)] for i in range(num_videos)]
t = []
for i in all_frames:
    t += i
all_frames = torch.tensor(t)
# 1000 X 22

# for i in range(num_videos):
#     frames = torch.tensor([])
#     base_dir = f"./../../../scratch/{net_id}/dataset_videos/dataset/train/video_{i}/"
#     image_names = [f'image_{i}.png' for i in range(num_frames_per_video)]
#     for file_name in image_names:
#         img = plt.imread(base_dir + file_name)
#         frames = torch.cat([frames,torch.tensor(img).unsqueeze(0)], dim = 0)
#     all_sequences.append(torch.tensor(frames))
#     mask = np.load(base_dir + 'mask.npy')
#     all_sequence_masks.append(torch.tensor(mask))

# all_frames = torch.cat([i for i in all_sequences], dim = 0)
# all_masks = torch.cat([i for i in all_sequence_masks], dim = 0)
# print(f"All frames and masks loaded,\nShape of frames : {all_frames.shape}, Shape of masks: {all_masks.shape}")

class FCN(nn.Module):
    def __init__(self, num_input_channels=3, num_classes=49):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1)
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




# Hyperparameters
num_input_channels = 3
num_classes = 49
batch_size = 8
learning_rate = 0.001
num_epochs = 10

# Instantiate the model and set up the optimizer and loss function
model = FCN(num_input_channels, num_classes).cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Create DataLoader
train_dataset = CustomDataset(all_frames) #, all_masks)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print('Dataset created, starting training')

train_loss = []
# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for images, masks in tqdm(train_loader):
        images, masks = images.cuda(), masks.cuda()
        optimizer.zero_grad()
        outputs = model(images)
#         masks = masks.argmax(dim=1)
        masks = masks.long()
        # print(outputs.shape, masks.shape)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    train_loss.append(average_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    # Save the trained model if needed
    torch.save(model.state_dict(), 'fcn_model.pth')


plt.plot(train_loss)
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.title('Epoch v/s Train loss')
plt.savefig("epoch_loss.jpg")