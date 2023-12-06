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
torch.cuda.empty_cache()

class CreateDatasetCustom(Dataset):
    def __init__(self, num_of_vids, evaluation_mode=False):
        if evaluation_mode:
            start_num = 1000
        else:
            start_num = 2000
        self.vid_indexes = torch.tensor([i for i in range(start_num, num_of_vids + start_num)])
        self.evaluation_mode = evaluation_mode

    def __getitem__(self, idx):
        num_hidden_frames = 11
        num_total_frames = 22
        x = []
        y = []
        i = self.vid_indexes[idx]
        if self.evaluation_mode:
            mode = 'val'
        else:
            mode = 'unlabeled'
        filepath = f'./../../dataset/{mode}/video_{i}/'
        # obtain x values.
        for j in range(num_hidden_frames):
            x.append(torch.tensor(plt.imread(filepath + f'image_{j}.png')).permute(2, 0, 1))
        x = torch.stack(x, 0)
        for j in range(num_hidden_frames, num_total_frames):
            y.append(torch.tensor(plt.imread(filepath + f'image_{j}.png')).permute(2, 0, 1))
        y = torch.stack(y, 0)
        return x, y

    def __len__(self):
        vid_len = len(self.vid_indexes)
        return vid_len


class CNN_Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        k_size = 3
        p_len = 1
        self.input_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=k_size, stride=1, padding=p_len),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2))
        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=k_size, stride=2, padding=p_len),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=k_size, stride=1, padding=p_len),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=k_size, stride=2, padding=p_len),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.input_encoder(x)
        x_enc = x.clone()
        x = self.encoder(x)
        return x, x_enc


class GroupConvolution(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConvolution, self).__init__()
        self.k = kernel_size
        self.s = stride
        self.p = padding
        self.a_normalization = act_norm
        if input_channels % groups != 0:
            groups = 1
        self.convolution = nn.Conv2d(input_channels, output_channels, kernel_size=self.k, stride=self.s,
                                     padding=self.p, groups=groups)
        self.normalization = nn.GroupNorm(groups, output_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        y_label = self.convolution(x)
        if self.a_normalization:
            y_norm = self.normalization(y_label)
            y_label = self.activation(y_norm)
        return y_label


class InceptionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, inception_kernel=[3, 5, 7, 11], groups=8):
        super().__init__()
        self.k_size = 1
        self.s_size = 1
        list_layers = []
        self.convolution = nn.Conv2d(input_dim, hidden_dim, kernel_size=self.k_size, stride=self.s_size, padding=0)
        for k in inception_kernel:
            list_layers.append(
                GroupConvolution(hidden_dim, output_dim, kernel_size=k, stride=self.s_size, padding=k // 2,
                                 groups=groups, act_norm=True))
        self.layers = nn.Sequential(*list_layers)

    def forward(self, x):
        x = self.convolution(x)
        y_label = 0
        for layer in self.layers:
            y_label += layer(x)
        return y_label


class InceptionBridge(nn.Module):
    def __init__(self, input_channels, hidden_channels, N_T, inception_kernel=[3, 5, 7, 11], groups=8):
        super().__init__()
        self.N_T = N_T
        # encoder.
        encoder_layers = [
            InceptionModule(input_channels, hidden_channels // 2, hidden_channels, inception_kernel=inception_kernel,
                            groups=groups)]
        for i in range(1, N_T - 1):
            encoder_layers.append(InceptionModule(hidden_channels, hidden_channels // 2, hidden_channels,
                                                  inception_kernel=inception_kernel, groups=groups))
        encoder_layers.append(
            InceptionModule(hidden_channels, hidden_channels // 2, hidden_channels, inception_kernel=inception_kernel,
                            groups=groups))
        # decoder.
        decoder_layers = [
            InceptionModule(hidden_channels, hidden_channels // 2, hidden_channels, inception_kernel=inception_kernel,
                            groups=groups)]
        for i in range(1, N_T - 1):
            decoder_layers.append(InceptionModule(2 * hidden_channels, hidden_channels // 2, hidden_channels,
                                                  inception_kernel=inception_kernel, groups=groups))
        decoder_layers.append(InceptionModule(2 * hidden_channels, hidden_channels // 2, input_channels,
                                              inception_kernel=inception_kernel, groups=groups))
        # self vars.
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # Encoder.
        list_pass = []
        z_hid = x
        for i in range(self.N_T):
            z_hid = self.encoder[i](z_hid)
            if (i < self.N_T - 1):
                list_pass.append(z_hid)

        # Decoder.
        z_hid = self.decoder[0](z_hid)
        for i in range(1, self.N_T):
            z_hid = self.decoder[i](torch.cat([z_hid, list_pass[-i]], dim=1))

        y_label = z_hid.reshape(B, T, C, H, W)
        return y_label


class CNN_Decoder(nn.Module):
    def __init__(self, hidden_channels, output_channels):
        super().__init__()
        self.k_size = 3
        self.p_size = 1
        self.output_p = 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=self.k_size, stride=2, padding=self.p_size,
                               output_padding=self.output_p),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=self.k_size, stride=1,
                               padding=self.p_size),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=self.k_size, stride=2, padding=self.p_size,
                               output_padding=self.output_p),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2))
        self.output_decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * hidden_channels, hidden_channels, kernel_size=self.k_size, stride=1,
                               padding=self.p_size),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2))

        self.output = nn.Conv2d(hidden_channels, output_channels, 1)

    def forward(self, x, encoding):
        x = self.decoder(x)
        y_label = self.output_decoder(torch.cat([x, encoding], dim=1))
        y_label = self.output(y_label)
        return y_label


class DLModelVideoPrediction(nn.Module):
    def __init__(self, input_dim, hidden_size=16, translator_size=256, inception_kernel=[3, 5, 7, 11], groups=8):
        super().__init__()
        T, C, H, W = input_dim
        self.encoding = CNN_Encoder(C, hidden_size)
        self.hidden = InceptionBridge(T * hidden_size, translator_size, 8, inception_kernel, groups)
        self.decoding = CNN_Decoder(hidden_size, C)

    def forward(self, x_orig):
        B, T, C, H, W = x_orig.shape
        x = x_orig.view(B * T, C, H, W)

        embed, pass_ = self.encoding(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hidden_el = self.hidden(z)
        hidden_el = hidden_el.reshape(B * T, C_, H_, W_)

        Y = self.decoding(hidden_el, pass_)
        Y = Y.reshape(B, T, C, H, W)
        return Y


batch_size = 8
num_videos = 13000
num_val_videos = 1000

train_data = CreateDatasetCustom(num_videos)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data = CreateDatasetCustom(num_val_videos, evaluation_mode=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

model = DLModelVideoPrediction((11, 3, 160, 240), 64, 512, groups=4)
model = nn.DataParallel(model)
model = model.to(device)

# Training Loop:
best_model_path = './checkpoint_frame_prediction.pth'  # load saved model to restart from previous best model
if os.path.isfile(best_model_path):
    model.load_state_dict(torch.load(best_model_path))

num_epochs = 10
lr = 0.001
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=len(train_loader),
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
    torch.save(model.state_dict(), './checkpoint_frame_prediction.pth')

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