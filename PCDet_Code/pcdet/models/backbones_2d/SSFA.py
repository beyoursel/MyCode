import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 use_norm=True,
                 activation=False
                 ):
        super().__init__()

        self.use_norm = use_norm
        self.activation = activation

        if self.use_norm:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding)

        if self.activation:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = ConvLayer(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, use_norm=True, activation=False)

    def forward(self, w):
        att = self.compress(w)
        # feature = self.compress(x)
        # att = torch.cat((scale, feature), dim=1)
        att = self.spatial(att)
        att = torch.sigmoid(att)  # broadcasting
        return att


class SSFAV1(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS  # 卷积层的数量
            layer_strides = self.model_cfg.LAYER_STRIDES  # stride
            num_filters = self.model_cfg.NUM_FILTERS  # 卷积核数量
        else:
            layer_nums = layer_strides = num_filters = []

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
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

        self.trans_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.trans_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.deconv_block_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.spatial = SpatialAttention()

        self.num_bev_features = 128

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.xavier_init(m, distribution="uniform")

    def forward(self, data_dict):

        spatial_features = data_dict['spatial_features']
        x_0 = self.blocks[0](spatial_features)  # spatial group的输出
        x_1 = self.blocks[1](x_0)  # semantic group的输出
        x_trans_0 = self.trans_0(x_0)  # spatial group后面conv的输出
        x_trans_1 = self.trans_1(x_1)  # semantic group后面的conv
        x_middle_0 = self.deconv_block_0(x_trans_1) + x_trans_0  # CIA-SSD中加号
        x_middle_1 = self.deconv_block_1(x_trans_1)  # conv后面的deconv的输出
        x_output_0 = self.conv_0(x_middle_0)
        x_output_1 = self.conv_1(x_middle_1)

        x_weight_0 = self.spatial(x_output_0)
        x_weight_1 = self.spatial(x_output_1)
        x_output = x_output_0 * x_weight_0 + x_output_1 * x_weight_1
        data_dict['spatial_features_2d'] = x_output.contiguous()

        return data_dict


class SSFA(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        self.bottom_up_block_0 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(input_channels, 128, 3, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.bottom_up_block_1 = nn.Sequential(
            # [200, 176] -> [100, 88]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),

        )

        self.trans_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.trans_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.deconv_block_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.w_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )  # 注意这里是否需要batchbnorm

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.w_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )

        self.num_bev_features = 128

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.xavier_init(m, distribution="uniform")

    def forward(self, data_dict):

        spatial_features = data_dict['spatial_features']
        x_0 = self.bottom_up_block_0(spatial_features)  # spatial group的输出
        x_1 = self.bottom_up_block_1(x_0)  # semantic group的输出
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


class SSFAV2(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS  # 卷积层的数量
            layer_strides = self.model_cfg.LAYER_STRIDES  # stride
            num_filters = self.model_cfg.NUM_FILTERS  # 卷积核数量
        else:
            layer_nums = layer_strides = num_filters = []

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
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
        #     if len(upsample_strides) > 0:
        #         stride = upsample_strides[idx]
        #         if stride >= 1:
        #             self.deblocks.append(nn.Sequential(
        #                 nn.ConvTranspose2d(
        #                     num_filters[idx], num_upsample_filters[idx],
        #                     upsample_strides[idx],
        #                     stride=upsample_strides[idx], bias=False
        #                 ),
        #                 nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
        #                 nn.ReLU()
        #             ))
        #         else:
        #             stride = np.round(1 / stride).astype(np.int)
        #             self.deblocks.append(nn.Sequential(
        #                 nn.Conv2d(
        #                     num_filters[idx], num_upsample_filters[idx],
        #                     stride,
        #                     stride=stride, bias=False
        #                 ),
        #                 nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
        #                 nn.ReLU()
        #             ))
        #
        # c_in = sum(num_upsample_filters)
        # if len(upsample_strides) > num_levels:
        #     self.deblocks.append(nn.Sequential(
        #         nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
        #         nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
        #         nn.ReLU(),
        #     ))




        # self.bottom_up_block_0 = nn.Sequential(
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(input_channels, 128, 3, stride=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        # )

        # self.bottom_up_block_1 = nn.Sequential(
        #     # [200, 176] -> [100, 88]
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
        #     nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
        #     nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
        #     nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #
        # )

        self.trans_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.trans_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.deconv_block_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        # self.w_0 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
        #     nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        # )  # 注意这里是否需要batchbnorm

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        # self.w_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
        #     nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        # )

        self.w = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(2, eps=1e-3, momentum=0.01),
        )
        self.num_bev_features = 256

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.xavier_init(m, distribution="uniform")

    def forward(self, data_dict):

        spatial_features = data_dict['spatial_features']
        x_0 = self.blocks[0](spatial_features)  # spatial group的输出
        x_1 = self.blocks[1](x_0)  # semantic group的输出
        x_trans_0 = self.trans_0(x_0)  # spatial group后面conv的输出
        x_trans_1 = self.trans_1(x_1)  # semantic group后面的conv
        x_middle_0 = self.deconv_block_0(x_trans_1) + x_trans_0  # CIA-SSD中加号
        x_middle_1 = self.deconv_block_1(x_trans_1)  # conv后面的deconv的输出
        x_output_0 = self.conv_0(x_middle_0)
        x_output_1 = self.conv_1(x_middle_1)

        x_output3 = torch.cat((x_output_0, x_output_1), dim=1)
        x_weight1 = torch.softmax(self.w(x_output3), dim=1)
        x_output = torch.cat((x_output_0 * x_weight1[:, 0:1, :, :], x_output_1 * x_weight1[:, 1:, :, :]), dim=1)
# Fusion模块
#         x_weight_0 = self.w_0(x_output_0)
#         x_weight_1 = self.w_1(x_output_1)
#         x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1)  # 将x_weight0和x_weight1拼接起来，然后进行softmax
#         # x_output = x_output_0 * x_weight[:, 0:1, :, :] + x_output_1 * x_weight[:, 1:, :, :]
#         x_output = torch.cat((x_output_0 * x_weight[:, 0:1, :, :], x_output_1 * x_weight[:, 1:, :, :]), dim=1)
        data_dict['spatial_features_2d'] = x_output.contiguous()

        return data_dict