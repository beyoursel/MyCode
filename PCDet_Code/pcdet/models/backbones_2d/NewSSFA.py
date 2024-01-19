import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SCConv(nn.Module):
    def __init__(self, planes, stride, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.k3 = nn.Sequential(
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.k4 = nn.Sequential(
            conv3x3(planes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class SCBottleneck(nn.Module):
    #expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = conv1x1(inplanes, planes)
        self.bn1_a = nn.BatchNorm2d(planes)

        self.k1 = nn.Sequential(
            conv3x3(planes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

        self.conv1_b = conv1x1(inplanes, planes)
        self.bn1_b = nn.BatchNorm2d(planes)

        self.scconv = SCConv(planes, stride, self.pooling_r)

        self.conv3 = conv1x1(planes * 2, planes * 2 )
        self.bn3 = nn.BatchNorm2d(planes * 2 )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class NewSSFA(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        self.bottom_up_block_0 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(input_channels, 128, 3, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU())

        self.SC01 = SCBottleneck(128, 128)
        self.SC02 = SCBottleneck(128, 128)

        self.bottom_up_block_1 = nn.Sequential(
            # [200, 176] -> [100, 88]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU())

        self.SC11 = SCBottleneck(256, 256)
        self.SC12 = SCBottleneck(256, 256)

        self.trans_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.trans_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.deconv_block_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.conv_0 = SCBottleneck(256, 256)

        self.w_0 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )

        self.conv_1 = SCBottleneck(256, 256)

        self.w_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )

        self.num_bev_features = 256

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.xavier_init(m, distribution="uniform")

    def forward(self, data_dict):

        spatial_features = data_dict['spatial_features']
        x_0 = self.bottom_up_block_0(spatial_features)  # spatial group的输出
        x_0 = self.SC01(x_0)
        x_0 = self.SC02(x_0)

        x_1 = self.bottom_up_block_1(x_0)  # semantic group的输出
        x_1 = self.SC11(x_1)
        x_1 = self.SC12(x_1)

        x_trans_0 = self.trans_0(x_0)  # spatial group后面conv的输出
        x_trans_1 = self.trans_1(x_1)  # semantic group后面的conv
        x_middle_0 = self.deconv_block_0(x_trans_1) + x_trans_0  # CIA-SSD中加号
        x_middle_1 = self.deconv_block_1(x_trans_1)  # conv后面的deconv的输出
        x_output_0 = self.conv_0(x_middle_0)
        x_output_1 = self.conv_1(x_middle_1)

# Fusion模块
        x_weight_0 = self.w_0(x_output_0)
        x_weight_1 = self.w_1(x_output_1)
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1)  # 将x_weight0和x_weight1拼接起来，然后进行softmax
        x_output = x_output_0 * x_weight[:, 0:1, :, :] + x_output_1 * x_weight[:, 1:, :, :]
        data_dict['spatial_features_2d'] = x_output.contiguous()

        return data_dict