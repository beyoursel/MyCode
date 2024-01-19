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


class Scnet2DBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            # for k in range(layer_nums[idx]):
            #     cur_layers.extend([
            #         nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
            #         nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
            #         nn.ReLU()
            #     ])
            self.blocks.append(nn.Sequential(*cur_layers))

            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters[0:2])
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, num_upsample_filters[-1], upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(num_upsample_filters[-1], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.SC01 = SCBottleneck(128, 128)
        self.SC02 = SCBottleneck(128, 128)
        self.SC03 = SCBottleneck(128, 128)
        self.SC04 = SCBottleneck(128, 128)
        self.SC05 = SCBottleneck(128, 128)
        self.SC11 = SCBottleneck(256, 256)
        self.SC12 = SCBottleneck(256, 256)
        self.SC13 = SCBottleneck(256, 256)
        self.SC14 = SCBottleneck(256, 256)
        self.SC15 = SCBottleneck(256, 256)

        self.num_bev_features = num_upsample_filters[-1]


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i == 0:
                x = self.SC01(x)
                x = self.SC02(x)
                x = self.SC03(x)
                x = self.SC04(x)
                x = self.SC05(x)
            elif i == 1:
                x = self.SC11(x)
                x = self.SC12(x)
                x = self.SC13(x)
                x = self.SC14(x)
                x = self.SC15(x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
