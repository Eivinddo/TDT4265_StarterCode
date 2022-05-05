import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import OrderedDict, Tuple, List

"""
This code is inspired by:
https://github.dev/tristandb/EfficientDet-PyTorch/blob/master/bifpn.py
"""

class DepthwiseSeparableConvBlock(nn.Module):
    """
    Depthwise separable convolution block, with batch normalization and ReLU activation.
    The depthwise separable convolution is divided into a depthwise and a pointwise convolution.
    The depthwise convolution keeps the depth of the input -> out_channels = in_channels
    The pointwise convolution keeps the wxh of the input, and changes depth. -> kernel=1x1, stride=1, padding=0
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=1, stride=1, padding=0):
        super(DepthwiseSeparableConvBlock,self).__init__()

        if out_channels == None:
            out_channels = in_channels

        self.depthConv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.pointConv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = self.depthConv(inputs)
        x = self.pointConv(x)
        x = self.batchnorm(x)
        return self.activation(x)


class ConvBlock(nn.Module):
    """
    Convolution block, with batch normalization and ReLU activation
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=1, stride=1, padding=0):
        super(ConvBlock,self).__init__()

        if out_channels == None:
            out_channels = in_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        return self.activation(x)


class BiFPNBlock(nn.Module):
    """
    Making a complete block of the BiFPN as shown in:
    Mingxing Tan, Ruoming Pang, and Quoc V Le. Efficientdet: Scalable and efficient object detection. 
    In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10781–10790, 2020. 7
    """
    def __init__(self, feature_size=128, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p3_td = DepthwiseSeparableConvBlock(feature_size)
        self.p4_td = DepthwiseSeparableConvBlock(feature_size)
        self.p5_td = DepthwiseSeparableConvBlock(feature_size)
        self.p6_td = DepthwiseSeparableConvBlock(feature_size)
        self.p7_td = DepthwiseSeparableConvBlock(feature_size)
        
        self.p4_out = DepthwiseSeparableConvBlock(feature_size)
        self.p5_out = DepthwiseSeparableConvBlock(feature_size)
        self.p6_out = DepthwiseSeparableConvBlock(feature_size)
        self.p7_out = DepthwiseSeparableConvBlock(feature_size)
        self.p8_out = DepthwiseSeparableConvBlock(feature_size)
        
        # TODO: Init weights
        # self.w1_td = nn.Parameter(torch.Tensor(2, 5))
        # nn.init.kaiming_uniform_(self.w1_td, nonlinearity='relu')
        self.w1_td = torch.as_tensor(torch.ones(2, 5))
        self.w1_relu = nn.ReLU()
        # self.w2_up = nn.Parameter(torch.Tensor(3, 5))
        # nn.init.kaiming_uniform_(self.w2_up, nonlinearity='relu')
        self.w2_up = torch.as_tensor(torch.ones(3, 5))
        self.w2_relu = nn.ReLU()
    
    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x, p7_x, p8_x = inputs
        
        # Calculate Top-Down Pathway
        w1_td = self.w1_relu(self.w1_td)
        w1_td /= torch.sum(w1_td, dim=0) + self.epsilon
        w2_up = self.w2_relu(self.w2_up)
        w2_up /= torch.sum(w2_up, dim=0) + self.epsilon
        
        p8_td = p8_x
        p7_td = self.p7_td(w1_td[0, 0] * p7_x + w1_td[1, 0] * F.interpolate(p8_td, scale_factor=2))
        p6_td = self.p6_td(w1_td[0, 1] * p6_x + w1_td[1, 1] * F.interpolate(p7_td, scale_factor=2))
        p5_td = self.p5_td(w1_td[0, 2] * p5_x + w1_td[1, 2] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1_td[0, 3] * p4_x + w1_td[1, 3] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1_td[0, 4] * p3_x + w1_td[1, 4] * F.interpolate(p4_td, scale_factor=2))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2_up[0, 0] * p4_x + w2_up[1, 0] * p4_td + w2_up[2, 0] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2_up[0, 1] * p5_x + w2_up[1, 1] * p5_td + w2_up[2, 1] * nn.Upsample(scale_factor=0.5)(p4_out))
        p6_out = self.p6_out(w2_up[0, 2] * p6_x + w2_up[1, 2] * p6_td + w2_up[2, 2] * nn.Upsample(scale_factor=0.5)(p5_out))
        p7_out = self.p7_out(w2_up[0, 3] * p7_x + w2_up[1, 3] * p7_td + w2_up[2, 3] * nn.Upsample(scale_factor=0.5)(p6_out))
        p8_out = self.p8_out(w2_up[0, 4] * p8_x + w2_up[1, 4] * p8_td + w2_up[2, 4] * nn.Upsample(scale_factor=0.5)(p7_out))

        return [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out]
    
class BiFPN(nn.Module):
    def __init__(self, pretrained: bool, 
                output_feature_sizes: List[List[int]], 
                fpn_out_channels: int = 128, 
                num_layers=3, 
                epsilon=0.0001):
        super(BiFPN, self).__init__()
        
        self.fpn_out_channels = fpn_out_channels
        self.out_channels = [self.fpn_out_channels]*6
        self.output_feature_shape = output_feature_sizes
        
        self.resnet_out_channels = [64, 128, self.fpn_out_channels, self.fpn_out_channels, self.fpn_out_channels, self.fpn_out_channels, self.fpn_out_channels, self.fpn_out_channels]

        self.feature_extractor = nn.Sequential(*list(torchvision.models.resnet34(pretrained=pretrained).children())[:-4])

        # print("---------------------------------------------")
        # print("ResNet34")
        # print(nn.Sequential(*list(torchvision.models.resnet34(pretrained=pretrained).children())))
        # print("---------------------------------------------")

        # print("---------------------------------------------")
        # print("Feature extractor without last 4 from resnet")
        # print(self.feature_extractor)
        # print("---------------------------------------------")
        
        self.feature_extractor.add_module("p3", nn.Sequential(
            ConvBlock(64, self.fpn_out_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock(self.fpn_out_channels, self.fpn_out_channels, kernel_size=1, stride=1, padding=0),
        ))

        self.feature_extractor.add_module("p4", nn.Sequential(
            ConvBlock(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=2, padding=1),
        ))

        self.feature_extractor.add_module("p5", nn.Sequential(
            ConvBlock(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=2, padding=1),
        ))

        self.feature_extractor.add_module("p6", nn.Sequential(
            ConvBlock(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=2, padding=1),
        ))

        self.feature_extractor.add_module("p7", nn.Sequential(
            ConvBlock(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=2, padding=1),
        ))

        self.feature_extractor.add_module("p8", nn.Sequential(
            ConvBlock(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=2, padding=1),
        ))

        # print("---------------------------------------------")
        # print("Feature extractor")
        # print(self.feature_extractor)
        # print("---------------------------------------------")

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(self.fpn_out_channels))
        self.bifpn = nn.Sequential(*bifpns)
        # print("---------------------------------------------")
        # print(self.bifpn)
        # print("---------------------------------------------")

    def forward(self, x):
        # print(self.feature_extractor)
        # Ignore five first "layers"/operations, befor we start "storing" at p3
        for i in range(5):
            x = self.feature_extractor[i](x)
        
        # Calculate the input column of BiFPN
        p3_x = self.feature_extractor[6](x)
        p4_x = self.feature_extractor[7](p3_x)
        p5_x = self.feature_extractor[8](p4_x)
        p6_x = self.feature_extractor[9](p5_x)
        p7_x = self.feature_extractor[10](p6_x)        
        p8_x = self.feature_extractor[11](p7_x)        
        
        features = [p3_x, p4_x, p5_x, p6_x, p7_x, p8_x]
        # print("-------------------------------------")
        # for i, el in enumerate(features):
        #     print(f"- p{i+3}_x", str(el.shape).ljust(29), '-')
        # print("-------------------------------------")

        # out_features = self.fpn(pn_x).values()
        out_features = self.bifpn(features)
        # print("out features", len(out_features))
        for idx, feature in enumerate(out_features):
            h, w = self.output_feature_shape[idx]
            expected_shape = (self.fpn_out_channels, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)
