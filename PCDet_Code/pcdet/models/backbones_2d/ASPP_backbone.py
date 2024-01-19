import torch
import torch.nn as nn


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn



class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class ASPP_Backbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super(ASPP_Backbone, self).__init__()

        self.rates = model_cfg.get('dialate_rates', None)
        self.out = model_cfg.get('num_bev_features', None)
        # ASPP uses atrous convolutions with different rates

        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=self.rates[0], dilation=self.rates[0])
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=self.rates[1], dilation=self.rates[1])
        self.conv3 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=self.rates[2], dilation=self.rates[2])
        # self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=self.rates[3], dilation=self.rates[3])

        # Image-level features
        # self.image_pooling = nn.AdaptiveAvgPool2d(1)
        # self.image_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Output convolution
        # self.output_conv = nn.Conv2d(in_channels * 5, in_channels, kernel_size=1)
        self.num_bev_features = self.out


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """        
        spatial_features = data_dict['spatial_features']
        feature1 = self.conv1(spatial_features)
        feature2 = self.conv2(spatial_features)
        feature3 = self.conv3(spatial_features)
        # feature4 = self.conv4(x)
        # feature5 = self.conv5(x)

        # Concatenate features
        out_concat = torch.cat([feature1, feature2, feature3, spatial_features], dim=1)

        # Output convolution
        # output = self.output_conv(concatenated) # neet to adjust
        data_dict['spatial_features_2d'] = out_concat
        
        return data_dict


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPPModule, self).__init__()

        # ASPP uses atrous convolutions with different rates
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[3], dilation=rates[3])

        # Image-level features
        self.image_pooling = nn.AdaptiveAvgPool2d(1)
        self.image_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Output convolution
        self.output_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        feature1 = self.conv1(x)
        feature2 = self.conv2(x)
        feature3 = self.conv3(x)
        feature4 = self.conv4(x)
        feature5 = self.conv5(x)

        image_feature = self.image_conv(self.image_pooling(x))
        image_feature = torch.nn.functional.interpolate(image_feature, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate features
        concatenated = torch.cat([feature1, feature2, feature3, feature4, feature5, image_feature], dim=1)

        # Output convolution
        output = self.output_conv(concatenated)

        return output

if __name__ == "__main__":

    # Example usage
    in_channels = 256
    out_channels = 128
    rates = [1, 6, 12, 18]

    aspp_module = ASPPModule(in_channels, out_channels, rates)
    input_tensor = torch.randn(1, in_channels, 64, 64)  # Replace with your input size
    output_tensor = aspp_module(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
