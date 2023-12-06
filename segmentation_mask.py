# import numpy as np
# import matplotlib.pyplot as plt
# import os

# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# import sys

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# class CustomDataset(Dataset):
#     def __init__(self, all_frames):
#         self.frames = torch.tensor(all_frames)
# #         self.masks = all_masks.cuda()

#     def __len__(self):
#         return len(self.frames)

#     def __getitem__(self, idx):
# #         global net_id
#         i,j = self.frames[idx]
#         file_path = f"./../dataset/train/video_{i}/image_{j}.png"
#         frame = torch.tensor(plt.imread(file_path)).permute(2, 0, 1)

#         file_path = f"./../dataset_videos/dataset/train/video_{i}/mask.npy"
#         mask = np.load(file_path)[j]
#         return frame, mask


# # net_id = sys.argv[1]
# # net_id = 'ar7964'

# # all_sequences = []
# # all_sequence_masks = []

# num_videos = 1000
# num_frames_per_video = 22

# all_frames = [[[i,j] for j in range(num_frames_per_video)] for i in range(num_videos)]
# t = []
# for i in all_frames:
#     t += i
# all_frames = torch.tensor(t)

# # 1000 X 22

# # for i in range(num_videos):
# #     frames = torch.tensor([])
# #     base_dir = f"./../../../scratch/{net_id}/dataset_videos/dataset/train/video_{i}/"
# #     image_names = [f'image_{i}.png' for i in range(num_frames_per_video)]
# #     for file_name in image_names:
# #         img = plt.imread(base_dir + file_name)
# #         frames = torch.cat([frames,torch.tensor(img).unsqueeze(0)], dim = 0)
# #     all_sequences.append(torch.tensor(frames))
# #     mask = np.load(base_dir + 'mask.npy')
# #     all_sequence_masks.append(torch.tensor(mask))

# # all_frames = torch.cat([i for i in all_sequences], dim = 0)
# # all_masks = torch.cat([i for i in all_sequence_masks], dim = 0)
# # print(f"All frames and masks loaded,\nShape of frames : {all_frames.shape}, Shape of masks: {all_masks.shape}")

# class FCN(nn.Module):
#     def __init__(self, num_classes):
#         super(FCN, self).__init__()
        
#         # Branch Connectors
#         self.bc_u1_c2 = nn.ConvTranspose2d(64,64, kernel_size=4, stride=2, padding=1)
#         self.bcnorm1 = nn.BatchNorm2d(64)
        
#         self.bc_u2_c3 = nn.ConvTranspose2d(128,128, kernel_size=4, stride=4, padding=0)
#         self.bcnorm2 = nn.BatchNorm2d(128)
        
#         self.bc_u3_c4 = nn.ConvTranspose2d(128,128, kernel_size=8, stride=8, padding=0)
#         self.bcnorm3 = nn.BatchNorm2d(128)
        
#         self.bc_u4_c5 = nn.ConvTranspose2d(128,128, kernel_size=8, stride=8, padding=0)
#         self.bcnorm4 = nn.BatchNorm2d(128)
        
#         self.bc_u5_c6 = nn.ConvTranspose2d(128,128, kernel_size=4, stride=4, padding=0)
#         self.bcnorm5 = nn.BatchNorm2d(128)
        
#         self.bc_u6_c7 = nn.ConvTranspose2d(128,128, kernel_size=4, stride=2, padding=1)
#         self.bcnorm6 = nn.BatchNorm2d(128)
        
        
#         # First branch of the network

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

#         self.conv2 = nn.Conv2d(64*2, 128, kernel_size=5, padding=2)
#         self.bnorm1 = nn.BatchNorm2d(128)
        
#         self.conv3 = nn.Conv2d(128*2, 128, kernel_size=9, padding=4)
        
#         self.conv4 = nn.Conv2d(128*2, 128, kernel_size=15, padding=7)
#         self.bnorm2 = nn.BatchNorm2d(128)

#         self.conv5 = nn.Conv2d(128*2, 128, kernel_size=15, padding=7)
#         self.bnorm3 = nn.BatchNorm2d(128)
        
#         self.conv6 = nn.Conv2d(128*2,128, kernel_size=9, padding=4)
        
#         self.conv7 = nn.Conv2d(128*2,64, kernel_size=5, padding=2)
#         self.bnorm4 = nn.BatchNorm2d(64)
                
        
#         # Second branch of the network
#         # b, 3, 160, 240
#         self.uconv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)

#         # b, 64, 80, 120
#         self.uconv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
#         self.ump1 = nn.MaxPool2d(kernel_size= 2)
#         self.ubnorm1 = nn.BatchNorm2d(128)
        
#         # b, 128, 40, 60
#         self.uconv3 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
#         self.ump2 = nn.MaxPool2d(kernel_size= 2)
        
#         # b, 128, 20, 30
#         self.uconv4 = nn.Conv2d(128, 128, kernel_size=15, padding=7)
#         self.ubnorm2 = nn.BatchNorm2d(128)

#         # b, 128, 20, 30
#         self.uconv5 = nn.Conv2d(128, 128, kernel_size=15, padding=7)
#         self.ubnorm3 = nn.BatchNorm2d(128)
#         self.upsamp1 = nn.ConvTranspose2d(128,128, kernel_size=4, stride=2, padding=1)
        
#         # b, 128, 40, 60
#         self.uconv6 = nn.Conv2d(128,128, kernel_size=9, padding= 4)
#         self.upsamp2 = nn.ConvTranspose2d(128,128, kernel_size=4, stride=2, padding=1)
        
#         # b, 128, 80, 120
#         self.uconv7 = nn.Conv2d(128,128, kernel_size=5, padding = 2)
#         self.upsamp3 = nn.ConvTranspose2d(128,64, kernel_size=4, stride=2, padding=1)
#         self.ubnorm4 = nn.BatchNorm2d(64)
#         # b, 64, 160, 240
        
        
#         # Final layer of the network
#         # x: b, 64, 160, 240 and x_u: b, 64, 160, 240
#         # Overall input shape for final layer: b, 128, 160, 240
#         self.conv8 = nn.Conv2d(64*2, num_classes, kernel_size=3, padding = 1)

        
        
#     def forward(self, x):
        
#         x_u = x.clone().to(device)
# #         print(f'layer 1 inputs: x: {x.shape}, x_u: {x_u.shape}')
        
#         # LAYER 1
#         # x: batch, 3, 160, 240
#         # x_u: batch, 3, 160, 240
#         x = self.conv1(x)
#         x = F.relu(x)

#         x_u = self.uconv1(x_u)
#         x_u = F.relu(x_u)
# #         print('uconv 1:',x_u.shape)
        

#         # LAYER 2
#         # x: batch, 64, 160, 240
#         # x_u: batch, 64, 80, 120
# #         print(f'layer 2 inputs: x: {x.shape}, x_u: {x_u.shape}')
#         x_u_upsampled = self.bc_u1_c2(x_u)
#         x = torch.cat([x,x_u_upsampled], axis = 1)
#         x = self.conv2(x)
#         x = self.bnorm1(x)
#         x = F.relu(x)
        
#         x_u = self.uconv2(x_u)
#         x_u = self.ump1(x_u)
#         x_u = self.ubnorm1(x_u)
#         x_u = F.relu(x_u)
# #         print('uconv 2:',x_u.shape)
        
        

#         # LAYER 3
#         # x: batch, 128, 160, 240
#         # x_u: batch, 128, 40, 60
# #         print(f'layer 3 inputs: x: {x.shape}, x_u: {x_u.shape}')
#         x_u_upsampled = self.bc_u2_c3(x_u)
#         x = torch.cat([x,x_u_upsampled], axis = 1)
#         x = self.conv3(x)
#         x = F.relu(x)
        
#         x_u = self.uconv3(x_u)
#         x_u = self.ump1(x_u)
#         x_u = F.relu(x_u)
# #         print('uconv 3:',x_u.shape)

        

#         # LAYER 4
#         # x: batch, 128, 160, 240
#         # x_u: batch, 128, 20, 30
# #         print(f'layer 4 inputs: x: {x.shape}, x_u: {x_u.shape}')
#         x_u_upsampled = self.bc_u3_c4(x_u)
# #         print(x_u_upsampled.shape)
#         x = torch.cat([x,x_u_upsampled], axis = 1)
#         x = self.conv4(x)
#         x = self.bnorm2(x)
#         x = F.relu(x)

#         x_u = self.uconv4(x_u)
#         x_u = self.ubnorm2(x_u)
#         x_u = F.relu(x_u)
# #         print('uconv 4:',x_u.shape)   

        

#         # LAYER 5
#         # x: batch, 128, 160, 240
#         # x_u: batch, 128, 20, 30
# #         print(f'layer 5 inputs: x: {x.shape}, x_u: {x_u.shape}')
#         x_u_upsampled = self.bc_u4_c5(x_u)
#         x = torch.cat([x,x_u_upsampled], axis = 1)
#         x = self.conv5(x)
#         x = self.bnorm3(x)
#         x = F.relu(x)

#         x_u = self.uconv5(x_u)
# #         print(x.shape)
#         x_u = self.ubnorm3(x_u)
#         x_u = self.upsamp1(x_u)
#         x_u = F.relu(x_u)
# #         print('uconv 5:',x_u.shape)
        
        

#         # LAYER 6
#         # x: batch, 128, 160, 240
#         # x_u: batch, 128, 40, 60
# #         print(f'layer 6 inputs: x: {x.shape}, x_u: {x_u.shape}')
#         x_u_upsampled = self.bc_u5_c6(x_u)
#         x = torch.cat([x,x_u_upsampled], axis = 1)
#         x = self.conv6(x)
#         x = F.relu(x)

#         x_u = self.uconv6(x_u)
#         x_u = self.upsamp2(x_u)
#         x_u = F.relu(x_u)
# #         print('uconv 6:',x_u.shape)

        

#         # LAYER 7
#         # x: batch, 128, 160, 240
#         # x_u: batch, 128, 80, 120
# #         print(f'layer 7 inputs: x: {x.shape}, x_u: {x_u.shape}')
#         x_u_upsampled = self.bc_u6_c7(x_u)
#         x = torch.cat([x,x_u_upsampled], axis = 1)
#         x = self.conv7(x)
#         x = F.relu(x)
#         x = self.bnorm4(x)

#         x_u = self.uconv7(x_u)
#         x_u = F.relu(x_u)
#         x_u = self.upsamp3(x_u)
#         x_u = self.ubnorm4(x_u)
# #         print('uconv 7:',x_u.shape)


        
#         # LAYER 8 : Final
#         # x: batch, 64, 160, 240
#         # x_u: batch, 64, 80, 120
# #         print(f'layer 8 inputs: x: {x.shape}, x_u: {x_u.shape}')
# #         x_u_upsampled = self.bc_u6_c7(x_u)
#         x = torch.cat([x,x_u], axis = 1)
#         x = self.conv8(x)
# #         print('Final:',x.shape)
        
#         # x: batch, 49, 160, 240
#         return x

# # Hyperparameters
# num_input_channels = 3
# num_classes = 49
# batch_size = 8
# learning_rate = 0.001
# num_epochs = 10

# # Instantiate the model and set up the optimizer and loss function
# model = FCN(num_classes).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()

# # Create DataLoader
# train_dataset = CustomDataset(all_frames) #, all_masks)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# print('Dataset created, starting training')

# try:
#     model.load_state_dict(torch.load('fcn_model.pth'))
# except:
#     print('Could not find saved weights, beginning training from scratch')
    
# train_loss = []
# # Training loop
# for epoch in range(num_epochs):
#     total_loss = 0
#     for images, masks in tqdm(train_loader):
#         images, masks = images.to(device), masks.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
# #         masks = masks.argmax(dim=1)
#         masks = masks.long()
#         # print(outputs.shape, masks.shape)
#         loss = criterion(outputs, masks)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     average_loss = total_loss / len(train_loader)
#     train_loss.append(average_loss)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

#     # Save the trained model if needed
#     torch.save(model.state_dict(), 'fcn_model.pth')


# plt.plot(train_loss)
# plt.xlabel('Epochs')
# plt.ylabel('Training loss')
# plt.title('Epoch v/s Train loss')
# plt.savefig("epoch_loss.jpg")






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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# print(device)

torch.cuda.empty_cache()

class CustomDataset(Dataset):
    def __init__(self, all_frames):
        self.frames = torch.tensor(all_frames)
#         self.masks = all_masks.cuda()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        i,j = self.frames[idx]
        file_path = f"./../../dataset/train/video_{i}/image_{j}.png"
        frame = torch.tensor(plt.imread(file_path)).permute(2, 0, 1)

        file_path = f"./../../dataset/train/video_{i}/mask.npy"
        mask = np.load(file_path)[j]
        return frame, mask



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=49, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



def load_weights(model):
    best_model_path = './checkpoints/image_segmentation.pth'
    if os.path.isfile(best_model_path):
        print('image segmentation weights found')
        model.load_state_dict(torch.load(best_model_path))


def save_weights(model):
    torch.save(model.state_dict(), './checkpoints/image_segmentation.pth')
    print('model weights saved successfully')


# Create Train DataLoader
batch_size = 5
num_videos = 1000

num_frames_per_video = 22

all_frames = [[[i, j] for j in range(num_frames_per_video)] for i in range(num_videos)]
t = []
for i in all_frames:
    t += i
all_frames = torch.tensor(t)

# 1000 X 22

train_data = CustomDataset(all_frames)
# load the data.
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Hyperparameters:
num_epochs = 20
lr = 0.00001
model = UNet(bilinear=True)
model = model.to(device)
model = nn.DataParallel(model)
load_weights(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=len(train_loader),
                                                epochs=num_epochs)

train_losses = []
preds_per_epoch = []
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    train_loss = []
    model.train()
    train_pbar = tqdm(train_loader)

    for batch_x, batch_y in train_pbar:
        optimizer.zero_grad()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
        pred_y = model(batch_x)  # .long()
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
    save_weights(model)
    