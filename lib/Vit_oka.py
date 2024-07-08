import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import (Conv2d, CrossEntropyLoss, Dropout, LayerNorm, Linear,
                      Softmax)

from lib.DeiT import deit_base_patch16_384 as deit_base

assert timm.__version__ == "0.3.2"  # version check

from lib.DeiT import deit_small_patch16_224 as deit  # deit用？
from lib.models_vit import vit_large_patch16 as vit_large


class Vit_oka(nn.Module):
    def __init__(
        self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False
    ):
        super(Vit_oka, self).__init__()

        # self.transformer = deit_base(pretrained=pretrained)
        # self.transformer = vit_large(pretrained=pretrained)
        # self.transformer = vit_large(pretrained=False)
        self.transformer = deit(pretrained=pretrained)  # deit用？

        #############################################################################################
        # self.up1 = Up(in_ch1=1024, out_ch=512) # vit
        # self.up1 = Up(in_ch1=768, out_ch=512) # deit
        self.up1 = Up(in_ch1=384, out_ch=128)  # deit用？
        #############################################################################################
        # self.up2 = Up(512, 256)
        self.up2 = Up(128, 64)  # deit用？

        # self.final_1 = nn.Sequential(
        #     Conv(256, 256, 3, bn=True, relu=True),
        #     Conv(256, num_classes, 3, bn=False, relu=False)
        # )  # Transformer

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False),
        )  # Transformer#deit用？

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b = self.transformer(
            imgs
        )  # torch.Size([16, 3, 224, 224]) → torch.Size([16, 196, 1024])
        x_b = torch.transpose(x_b, 1, 2)  # torch.Size([16, 1024, 196])
        # x_b = x_b.view(x_b.shape[0], -1, 14, 14)  # t0 # torch.Size([16, 1024, 14, 14]) # 224*224
        x_b = x_b.view(
            x_b.shape[0], -1, 22, 22
        )  # t0 # torch.Size([16, 1024, 14, 14]) # 352*352

        x_b = self.drop(x_b)
        x_b_1 = self.up1(x_b)  # t1 # torch.Size([16, 512, 28, 28])
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(
            x_b_1
        )  # transformer pred supervise here #t2 # torch.Size([16, 256, 56, 56])
        x_b_2 = self.drop(x_b_2)

        # decoder part
        map_1 = F.interpolate(
            self.final_1(x_b_2), scale_factor=4, mode="bilinear"
        )  # Transformer pred map
        return map_1

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_1.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        """
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        """
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


class Conv(nn.Module):
    def __init__(
        self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True
    ):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(
            inp_dim,
            out_dim,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
