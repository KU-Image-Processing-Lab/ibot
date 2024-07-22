import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import (Conv2d, CrossEntropyLoss, Dropout, LayerNorm, Linear,
                      Softmax)
from torchvision.models import resnet50 as Resnet

from lib.Cara.axial_atten import AA_kernel
from lib.Cara.context_module import CFPModule
from lib.Cara.conv_layer import BNPReLU
from lib.Cara.conv_layer import Conv as Conv2
# from lib.DeiT import deit_base_patch16_384 as deit_base
from lib.models_vit import vit_large_patch16 as vit_large
# from .Res2Net_v1b import res2net50_v1b_26w_4s
from lib.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s

# from lib.TransFuse_s import TransFuse_S


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


# partial_decoderのコード
class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)

        self.conv4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4 * channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )
        x4_1 = (
            self.conv_upsample4(self.upsample(self.upsample(self.upsample(x1))))
            * self.conv_upsample5(self.upsample(self.upsample(x2)))
            * self.conv_upsample6(self.upsample(x3))
            * x4
        )

        x2_2 = torch.cat((x2_1, self.conv_upsample7(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample8(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x4_2 = torch.cat((x4_1, self.conv_upsample9(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x


# TransCaraNetのコード
class Trans_CaraNet_L(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, pretrained=False):
        super(Trans_CaraNet_L, self).__init__()
        # ---- ResNet Backbone ----

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # self.resnet = res2net101_v1b_26w_4s(pretrained=True)
        # self.Resnet = Resnet()
        # self.Resnet.load_state_dict(torch.load('resnet50_a1_0-14fe96d1.pth'))

        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(256, channel)
        self.rfb3_1 = RFB_modified(512, channel)
        self.rfb4_1 = RFB_modified(1024, channel)
        self.rfb5_1 = RFB_modified(2048, channel)

        # ---- Partial Decoder ----
        self.agg = aggregation(channel)

        # self.transformer = deit_base(pretrained=True)
        self.transformer = vit_large(pretrained=pretrained)

        self.up1 = Up(in_ch1=1024, out_ch=512)
        self.up2 = Up(512, 256)
        self.down1 = Down(1024, 2048)
        drop_rate = 0.2
        self.drop = nn.Dropout2d(drop_rate)

        self.up_c = BiFusion_block(
            ch_1=2048,
            ch_2=2048,
            r_2=4,
            ch_int=2048,
            ch_out=2048,
            drop_rate=drop_rate / 2,
        )  # top

        self.up_c_1_1 = BiFusion_block(
            ch_1=1024,
            ch_2=1024,
            r_2=2,
            ch_int=1024,
            ch_out=1024,
            drop_rate=drop_rate / 2,
        )  # mid
        self.up_c_1_2 = Up(in_ch1=2048, out_ch=1024, in_ch2=1024, attn=True)

        self.up_c_2_1 = BiFusion_block(
            ch_1=512, ch_2=512, r_2=1, ch_int=512, ch_out=512, drop_rate=drop_rate / 2
        )  # under
        self.up_c_2_2 = Up(1024, 512, 512, attn=True)

        self.up_c_3_1 = BiFusion_block(
            ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2
        )  # under
        self.up_c_3_2 = Up(512, 256, 256, attn=True)
        self.drop = nn.Dropout2d(drop_rate)

        self.init_weights()

        self.CFP_1 = CFPModule(32, d=8)
        self.CFP_2 = CFPModule(32, d=8)
        self.CFP_3 = CFPModule(32, d=8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv2(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv2(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv2(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv2(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv2(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv2(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv2(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv2(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv2(32, 1, 3, 1, padding=1, bn_acti=True)

        self.aa_kernel_1 = AA_kernel(32, 32)
        self.aa_kernel_2 = AA_kernel(32, 32)
        self.aa_kernel_3 = AA_kernel(32, 32)

    def forward(self, x):
        ####### ViTの特徴を抽出 #######
        x_b = self.transformer(x)
        x_b = torch.transpose(x_b, 1, 2)
        x_b = x_b.view(x_b.shape[0], -1, 22, 22)
        x_b = self.drop(x_b)

        ####### ResNetの特徴と融合するためにResNetのサイズと同じサイズに変更 #######
        x_b_1 = self.up1(x_b)  # t1 # torch.Size([16, 512, 48, 48])
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(
            x_b_1
        )  # transformer pred supervise here #t2 # torch.Size([16, 256, 96, 96])
        x_b_2 = self.drop(x_b_2)

        x_b_0 = self.down1(x_b)
        x_b_0 = self.drop(x_b_0)

        ####### ResNetの特徴を抽出 #######
        x = self.resnet.conv1(x)  # torch.Size([16, 64, 192, 192])
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # torch.Size([16, 64, 96, 96])

        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # g2 # torch.Size([16, 256, 96, 96])
        x1 = self.drop(x1)

        x2 = self.resnet.layer2(x1)  # g1 # torch.Size([16, 512, 48, 48])
        x2 = self.drop(x2)

        x3 = self.resnet.layer3(x2)  # g0 # torch.Size([16, 1024, 24, 24])
        x3 = self.drop(x3)
        x4 = self.resnet.layer4(x3)  # torch.Size([16, 2048, 12, 12])
        x4 = self.drop(x4)

        ####### BiFusion ModuleによりViTの特徴とResNetの特徴を融合 #######
        x_c = self.up_c(
            x4, x_b_0
        )  # biFusion pred here #f0 # torch.Size([16, 1024, 24, 24])

        x_c_1_1 = self.up_c_1_1(x3, x_b)  # f1 # torch.Size([16, 512, 48, 48])
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)  # torch.Size([16, 512, 48, 48])

        x_c_2_1 = self.up_c_2_1(x2, x_b_1)  # f2 # torch.Size([16, 256, 96, 96])
        x_c_2 = self.up_c_2_2(
            x_c_1, x_c_2_1
        )  # joint predict low supervise here # torch.Size([16, 256, 96, 96])

        x_c_3_1 = self.up_c_3_1(x1, x_b_2)
        x_c_3 = self.up_c_3_2(x_c_2, x_c_3_1)

        ####### partial decoder #######
        x2_rfb = self.rfb2_1(x_c_3)  # channel -> 32
        x3_rfb = self.rfb3_1(x_c_2)  # channel -> 32
        x4_rfb = self.rfb4_1(x_c_1)  # channel -> 32
        x5_rfb = self.rfb5_1(x_c)  # channel -> 32

        ra5_feat = self.agg(x5_rfb, x4_rfb, x3_rfb, x2_rfb)

        ####### 1つ目の出力 #######
        lateral_map_5 = F.interpolate(
            ra5_feat, scale_factor=4, mode="bilinear"
        )  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        ####### CFP ModuleとA-RAの機構1 #######
        decoder_2 = F.interpolate(ra5_feat, scale_factor=0.25, mode="bilinear")
        cfp_out_1 = self.CFP_3(x4_rfb)  # 32 - 32
        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3_o = decoder_2_ra.expand(-1, 32, -1, -1).mul(aa_atten_3)

        ra_3 = self.ra3_conv1(aa_atten_3_o)  # 32 - 32
        ra_3 = self.ra3_conv2(ra_3)  # 32 - 32
        ra_3 = self.ra3_conv3(ra_3)  # 32 - 1

        x_3 = ra_3 + decoder_2
        ####### 2つ目の出力 #######
        lateral_map_4 = F.interpolate(x_3, scale_factor=16, mode="bilinear")

        ####### CFP ModuleとA-RAの機構2 #######
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode="bilinear")
        cfp_out_2 = self.CFP_2(x3_rfb)  # 32 - 32
        decoder_3_ra = -1 * (torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2_o = decoder_3_ra.expand(-1, 32, -1, -1).mul(aa_atten_2)

        ra_2 = self.ra2_conv1(aa_atten_2_o)  # 32 - 32
        ra_2 = self.ra2_conv2(ra_2)  # 32 - 32
        ra_2 = self.ra2_conv3(ra_2)  # 32 - 1

        x_2 = ra_2 + decoder_3
        ####### 3つ目の出力 #######
        lateral_map_3 = F.interpolate(x_2, scale_factor=8, mode="bilinear")

        ####### CFP ModuleとA-RAの機構3 #######
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode="bilinear")
        cfp_out_3 = self.CFP_1(x2_rfb)  # 32 - 32
        decoder_4_ra = -1 * (torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1_o = decoder_4_ra.expand(-1, 32, -1, -1).mul(aa_atten_1)

        ra_1 = self.ra1_conv1(aa_atten_1_o)  # 32 - 32
        ra_1 = self.ra1_conv2(ra_1)  # 32 - 32
        ra_1 = self.ra1_conv3(ra_1)  # 32 - 1

        x_1 = ra_1 + decoder_4
        lateral_map_2 = F.interpolate(x_1, scale_factor=4, mode="bilinear")
        ####### 4つ目の出力 #######
        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.0):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)  # ^bi

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in  # ^gi

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in  # ^ti
        fuse = self.residual(torch.cat([g, x, bp], 1))  # fi

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


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


class Down(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.max(x1)
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


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


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


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


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


# if __name__ == "__main__":
#     ras = TransFuse_S().cuda()
#     input_tensor = torch.randn(1, 3, 384, 384).cuda()
#     out = ras(input_tensor)
#     print(out.shape())
