# This file will have the class definitions for all of our model classes

import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import sys


# Segmentation Mask - Unet model:

# Unet definition 2

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


# class encoding_block(nn.Module):
    
#     def __init__(self, in_channels, out_channels):
#         super(encoding_block, self).__init__()
#         model = []
#         model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
#         model.append(nn.BatchNorm2d(out_channels))
#         model.append(nn.ReLU(inplace=True))
#         model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
#         model.append(nn.BatchNorm2d(out_channels))
#         model.append(nn.ReLU(inplace=True))
#         self.conv = nn.Sequential(*model)

#     def forward(self, x):
#         return self.conv(x)


# class unet_model(nn.Module):
#     '''
#     example usage:
#     model = unet_model()
#     model = nn.DataParallel(model)
#     model = model.to(device)
#     '''
#     def __init__(self, out_channels=49, features=[64, 128, 256, 512]):
#         super(unet_model, self).__init__()
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#         self.conv1 = encoding_block(3, features[0])
#         self.conv2 = encoding_block(features[0], features[1])
#         self.conv3 = encoding_block(features[1], features[2])
#         self.conv4 = encoding_block(features[2], features[3])
#         self.conv5 = encoding_block(features[3] * 2, features[3])
#         self.conv6 = encoding_block(features[3], features[2])
#         self.conv7 = encoding_block(features[2], features[1])
#         self.conv8 = encoding_block(features[1], features[0])
#         self.tconv1 = nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2)
#         self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
#         self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
#         self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
#         self.bottleneck = encoding_block(features[3], features[3] * 2)
#         self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

#     def forward(self, x):
#         skip_connections = []
#         x = self.conv1(x)
#         skip_connections.append(x)
#         x = self.pool(x)
#         x = self.conv2(x)
#         skip_connections.append(x)
#         x = self.pool(x)
#         x = self.conv3(x)
#         skip_connections.append(x)
#         x = self.pool(x)
#         x = self.conv4(x)
#         skip_connections.append(x)
#         x = self.pool(x)
#         x = self.bottleneck(x)
#         skip_connections = skip_connections[::-1]
#         x = self.tconv1(x)
#         x = torch.cat((skip_connections[0], x), dim=1)
#         x = self.conv5(x)
#         x = self.tconv2(x)
#         x = torch.cat((skip_connections[1], x), dim=1)
#         x = self.conv6(x)
#         x = self.tconv3(x)
#         x = torch.cat((skip_connections[2], x), dim=1)
#         x = self.conv7(x)
#         x = self.tconv4(x)
#         x = torch.cat((skip_connections[3], x), dim=1)
#         x = self.conv8(x)
#         x = self.final_layer(x)
#         return x

# Frame Prediction model:


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

    
    
    
# COMBINED MODEL
    
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
