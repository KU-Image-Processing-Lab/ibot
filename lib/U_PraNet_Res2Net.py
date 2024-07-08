import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .Res2Net_v1b import res2net50_v1b_26w_4s


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


class Conv2d_bn(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(Conv2d_bn, self).__init__()
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

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicConv2d_xaviel(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConv2d_xaviel, self).__init__()
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

        nn.init.xavier_uniform_(self.conv.weight.data)

    def forward(self, x):
        x = self.conv(x)
        return x


class BasicConv2drelu(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConv2drelu, self).__init__()
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


class Squeeze_Excite_block(nn.Module):
    def __init__(self, out_channels):
        super(Squeeze_Excite_block, self).__init__()

        self.av1 = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(
            in_features=out_channels, out_features=out_channels // 8, bias=False
        )
        self.relu_a = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(
            in_features=out_channels // 8, out_features=out_channels, bias=False
        )
        self.sigmoid1 = nn.Sigmoid()

        nn.init.xavier_uniform_(self.l2.weight.data)

    def forward(self, x):
        batch_size = x.shape[0]
        se = self.av1(x)
        se = se.reshape(batch_size, -1)
        se = self.l1(se)
        se = self.relu_a(se)
        se = self.l2(se)
        se = self.sigmoid1(se)
        se = se.reshape(batch_size, -1, 1, 1)
        x = torch.mul(x, se)
        return x


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block, self).__init__()

        self.cbr2_1 = BasicConv2drelu(in_channels, out_channels, 3, padding=1)
        self.cbr2_2 = BasicConv2drelu(out_channels, out_channels, 3, padding=1)
        self.se = Squeeze_Excite_block(out_channels)

    def forward(self, x):
        x = self.cbr2_1(x)
        x = self.cbr2_2(x)
        x = self.se(x)

        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            # Squeeze_Excite_block(out_channel)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
            # Squeeze_Excite_block(out_channel)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
            # Squeeze_Excite_block(out_channel)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
            # Squeeze_Excite_block(out_channel)
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
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

        self.conv_block0 = Conv_block(832, 3 * channel)
        self.conv_block1 = Conv_block(3 * channel * 2, 3 * channel)
        self.conv_block2 = Conv_block(3 * channel + 256 + 128, 3 * channel)
        self.conv_block3 = Conv_block(3 * channel + 128 + 64, 3 * channel)
        self.conv_block4 = Conv_block(3 * channel + 64 + 32, 3 * channel)

        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

        nn.init.xavier_uniform_(self.conv5.weight.data)

    def forward(self, x1, x2, x3, a1, a2, a3, a4, b1, b2, b3, b4, b5):

        x1_1 = x1

        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2

        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)

        b1 = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(b1)
        b1 = torch.cat([b1, a1, b5], dim=1)
        b1 = self.conv_block0(b1)

        x = torch.cat([x, b1], dim=1)
        x = self.conv_block1(x)

        x = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(x)
        x = torch.cat([x, a2, b4], dim=1)
        x = self.conv_block2(x)

        x = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(x)
        x = torch.cat([x, a3, b3], dim=1)
        x = self.conv_block3(x)

        x = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(x)
        x = torch.cat([x, a4, b2], dim=1)
        x = self.conv_block4(x)

        x = self.conv5(x)

        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.avpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.cbr_1 = BasicConv2drelu(in_channels, out_channels, 1)
        self.cbr_2 = BasicConv2drelu(in_channels, out_channels, 1)
        self.cbr_3 = BasicConv2drelu(
            in_channels, out_channels, 3, padding=6, dilation=6
        )
        self.cbr_4 = BasicConv2drelu(
            in_channels, out_channels, 3, padding=12, dilation=12
        )
        self.cbr_5 = BasicConv2drelu(
            in_channels, out_channels, 3, padding=18, dilation=18
        )
        self.cbr_6 = BasicConv2drelu(out_channels * 5, out_channels, 1)

    def forward(self, x):
        shape = x.shape
        y1 = self.avpool(x)
        y1 = self.cbr_1(y1)
        y1 = nn.Upsample(
            scale_factor=(shape[2], shape[3]), mode="bilinear", align_corners=True
        )(y1)
        y2 = self.cbr_2(x)
        y3 = self.cbr_3(x)
        y4 = self.cbr_4(x)
        y5 = self.cbr_5(x)
        y = torch.cat([y1, y2, y3, y4, y5], dim=1)
        y = self.cbr_6(y)
        return y


class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.vgg = models.vgg19(pretrained=False)
        self.vgg.load_state_dict(torch.load("vgg19-dcbb9e9d.pth"))

    def forward(self, x):
        skip_connections = []
        x = self.vgg.features[0](x)
        skip_connections.append(x)
        for i in range(1, 8):
            x = self.vgg.features[i](x)
        skip_connections.append(x)
        for i in range(8, 17):
            x = self.vgg.features[i](x)
        skip_connections.append(x)
        for i in range(17, 26):
            x = self.vgg.features[i](x)
        skip_connections.append(x)
        for i in range(26, 35):
            x = self.vgg.features[i](x)

        output = x
        return output, skip_connections


class Decoder1(nn.Module):
    def __init__(self, ratio):
        super(Decoder1, self).__init__()

        self.ratio = ratio

        self.conv_block1 = Conv_block(576, 256)
        self.conv_block2 = Conv_block(512, 128)
        self.conv_block3 = Conv_block(256, 64)
        self.conv_block4 = Conv_block(128, 32)

    def forward(self, x, skip_connections):
        skip_connections.reverse()

        x = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.conv_block1(x)

        x = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = self.conv_block2(x)

        x = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = self.conv_block3(x)

        x = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(x)
        x = torch.cat([x, skip_connections[3]], dim=1)
        x = self.conv_block4(x)

        return x


class Encoder2(nn.Module):
    def __init__(self, ratio):
        super(Encoder2, self).__init__()
        self.ratio = ratio

        self.conv_block1 = Conv_block(3, 32)
        self.conv_block2 = Conv_block(32, 64)
        self.conv_block3 = Conv_block(64, 128)
        self.conv_block4 = Conv_block(128, 256)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []
        x = self.conv_block1(x)
        skip_connections.append(x)
        x = self.maxpool1(x)

        x = self.conv_block2(x)
        skip_connections.append(x)
        x = self.maxpool2(x)

        x = self.conv_block3(x)
        skip_connections.append(x)
        x = self.maxpool3(x)

        x = self.conv_block4(x)
        skip_connections.append(x)
        x = self.maxpool4(x)

        output = x
        return output, skip_connections


class Decoder2(nn.Module):
    def __init__(self, ratio):
        super(Decoder2, self).__init__()

        self.ratio = ratio

        self.conv_block1 = Conv_block(832, 256)
        self.conv_block2 = Conv_block(640, 128)
        self.conv_block3 = Conv_block(320, 64)
        self.conv_block4 = Conv_block(160, 32)

    def forward(self, x, skip_1, skip_2):
        skip_2.reverse()

        x = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(x)
        x = torch.cat([x, skip_1[0], skip_2[0]], dim=1)
        x = self.conv_block1(x)

        x = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(x)
        x = torch.cat([x, skip_1[1], skip_2[1]], dim=1)
        x = self.conv_block2(x)

        x = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(x)
        x = torch.cat([x, skip_1[2], skip_2[2]], dim=1)
        x = self.conv_block3(x)

        x = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)(x)
        x = torch.cat([x, skip_1[3], skip_2[3]], dim=1)
        x = self.conv_block4(x)

        return x


class Output_Block(nn.Module):
    def __init__(self):
        super(Output_Block, self).__init__()
        self.cbr = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.cbr.weight.data)

    def forward(self, x):
        x = self.cbr(x)
        x = self.sigmoid(x)
        return x


class U_PraNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(U_PraNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)

        # ---- Partial Decoder ----
        self.agg = aggregation(channel)
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = Conv2d_bn(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d_xaviel(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = Conv2d_bn(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d_xaviel(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = Conv2d_bn(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d_xaviel(64, 1, kernel_size=3, padding=1)

        self.ASPP1 = ASPP(in_channels=512, out_channels=64)
        self.ASPP2 = ASPP(in_channels=256, out_channels=64)
        self.Encoder1 = Encoder1()
        self.Dncoder1 = Decoder1(ratio=8)
        self.Encoder2 = Encoder2(ratio=8)
        self.Dncoder2 = Decoder2(ratio=8)
        self.Output_Block1 = Output_Block()

    def forward(self, x):
        input_1 = x
        x, skip_1 = self.Encoder1(x)
        x = self.ASPP1(x)
        x = self.Dncoder1(x, skip_1)
        output_1 = self.Output_Block1(x)
        input_2 = input_1 * output_1

        x, skip_2 = self.Encoder2(input_2)
        xx = self.ASPP2(x)

        x = self.resnet.conv1(input_2)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22

        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x3_rfb = self.rfb3_1(x3)  # channel -> 32
        x4_rfb = self.rfb4_1(x4)  # channel -> 32

        ra5_feat = self.agg(
            x4_rfb,
            x3_rfb,
            x2_rfb,
            skip_1[0],
            skip_1[1],
            skip_1[2],
            skip_1[3],
            xx,
            skip_2[0],
            skip_2[1],
            skip_2[2],
            skip_2[3],
        )
        lateral_map_5 = F.interpolate(
            ra5_feat, scale_factor=1, mode="bilinear"
        )  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25 / 8, mode="bilinear")
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = self.ra4_conv2(x)
        x = self.ra4_conv3(x)
        x = self.ra4_conv4(x)

        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(
            x, scale_factor=32, mode="bilinear"
        )  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = self.ra3_conv2(x)
        x = self.ra3_conv3(x)

        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(
            x, scale_factor=16, mode="bilinear"
        )  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = self.ra2_conv2(x)
        x = self.ra2_conv3(x)

        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(
            x, scale_factor=8, mode="bilinear"
        )  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2


if __name__ == "__main__":
    ras = U_PraNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)
