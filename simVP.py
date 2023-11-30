import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.cuda.empty_cache()

try:
    device = torch.device('cuda')
except:
    device = torch.device('cpu')

print(device)


# video_dir provided till unlabeled


class CustomDataset(Dataset):
    def __init__(self, n_videos, video_dir):
        self.video_dir = video_dir
        if video_dir[-3:] == 'val':
            self.video_idxs = torch.tensor([i for i in range(1000, n_videos + 1000)])
        else:
            self.video_idxs = torch.tensor([i for i in range(10000, n_videos + 10000)])

    def __len__(self):
        return len(self.video_idxs)

    def __getitem__(self, idx):
        i = self.video_idxs[idx]
        file_path = f'{self.video_dir}/video_{i}/'
        x = []
        for j in range(11):
            x.append(torch.tensor(plt.imread(file_path + f'image_{j}.png')).permute(2, 0, 1))
        x = torch.stack(x, 0)
        y = []
        for j in range(11, 22):
            y.append(torch.tensor(plt.imread(file_path + f'image_{j}.png')).permute(2, 0, 1))
        y = torch.stack(y, 0)
        return x, y


batch_size = 8

# Create DataLoader
train_dataset = CustomDataset(n_videos=10, video_dir='./../Dataset_Student/unlabeled')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset(n_videos=1000, video_dir='./../Dataset_Student/val')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.inp_enc = nn.Sequential(nn.Conv2d(in_channels, hid_channels, kernel_size=3, stride=1, padding=1),
                                     nn.GroupNorm(2, hid_channels),
                                     nn.LeakyReLU(0.2))
        self.enc = nn.Sequential(nn.Conv2d(hid_channels, hid_channels, kernel_size=3, stride=2, padding=1),
                                 nn.GroupNorm(2, hid_channels),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(hid_channels, hid_channels, kernel_size=3, stride=1, padding=1),
                                 nn.GroupNorm(2, hid_channels),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(hid_channels, hid_channels, kernel_size=3, stride=2, padding=1),
                                 nn.GroupNorm(2, hid_channels),
                                 nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.inp_enc(x)
        res = x.clone()
        x = self.enc(x)
        return x, res


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker // 2,
                                      groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class Translator(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super().__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            dec_layers.append(
                Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # Encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # Decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class Decoder(nn.Module):
    def __init__(self, hid_channels, out_channels):
        super().__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(2, hid_channels),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, hid_channels),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(2, hid_channels),
            nn.LeakyReLU(0.2))
        self.out_dec = nn.Sequential(
            nn.ConvTranspose2d(2 * hid_channels, hid_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, hid_channels),
            nn.LeakyReLU(0.2))

        self.out = nn.Conv2d(hid_channels, out_channels, 1)

    def forward(self, x, enc):
        #         print(f'x shape:{x.shape}, enc shape:{enc.shape}')
        #         for dec_layer in self.dec:
        x = self.dec(x)
        #             print(f'x shape:{x.shape}, enc shape:{enc.shape}')

        #         print(f'x shape:{x.shape}, enc shape:{enc.shape}')
        y = self.out_dec(torch.cat([x, enc], dim=1))
        y = self.out(y)
        return y


class ModelVideoPrediction(nn.Module):
    def __init__(self, shape_in, hidden_size=16, translator_size=256, incep_ker=[3, 5, 7, 11], groups=8):
        super().__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hidden_size)
        self.hid = Translator(T * hidden_size, translator_size, 8, incep_ker, groups)
        self.dec = Decoder(hidden_size, C)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y


model = ModelVideoPrediction((11, 3, 160, 240), 64, 512, groups=4)
model = model.to(device)

best_model_path = './checkpoint.pth'  # load saved model to restart from previous best model (lowest val loss)
# checkpoint


if os.path.isfile(best_model_path):
    model.load_state_dict(torch.load(best_model_path))

num_epochs = 20
lr = 0.001
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
                                                epochs=num_epochs)

train_losses = []

for epoch in range(num_epochs):
    train_loss = []
    model.train()
    train_pbar = tqdm(train_loader)
    count = 0

    for batch_x, batch_y in train_pbar:
        # print(count)
        optimizer.zero_grad()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred_y = model(batch_x)

        loss = criterion(pred_y, batch_y)
        train_loss.append(loss.item())
        train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        loss.backward()
        optimizer.step()
        scheduler.step()
        count = count + 1
        torch.cuda.empty_cache()

    train_loss = np.average(train_loss)
    train_losses.append(train_loss)

    torch.save(model.state_dict(), best_model_path)


    if epoch % 5 == 0:
        with torch.no_grad():
            val_loss = []
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred_y = model(batch_x)
                loss = criterion(pred_y, batch_y)
                val_loss.append(loss.item())

        current_val_loss = np.average(val_loss)
        print(current_val_loss)
        torch.cuda.empty_cache()

# model.load_state_dict(torch.load(best_model_path))
