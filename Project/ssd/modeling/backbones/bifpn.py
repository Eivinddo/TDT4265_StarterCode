import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import OrderedDict, Tuple, List

"""
This code is inspired by:
https://github.dev/tristandb/EfficientDet-PyTorch/blob/master/bifpn.py
"""

class SeperatableConvBlock(nn.Module):
    """
    Seperateable convolution block, with batch normalization and ReLU activation
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=1, stride=1, padding=0):
        super(SeperatableConvBlock,self).__init__()

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
    Seperateable convolution block, with batch normalization and ReLU activation
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
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p3_td = SeperatableConvBlock(feature_size)
        self.p4_td = SeperatableConvBlock(feature_size)
        self.p5_td = SeperatableConvBlock(feature_size)
        self.p6_td = SeperatableConvBlock(feature_size)
        
        self.p4_out = SeperatableConvBlock(feature_size)
        self.p5_out = SeperatableConvBlock(feature_size)
        self.p6_out = SeperatableConvBlock(feature_size)
        self.p7_out = SeperatableConvBlock(feature_size)
        
        # TODO: Init weights
        self.w1_td = nn.Parameter(torch.Tensor(2, 4))
        self.w1_relu = nn.ReLU()
        self.w2_up = nn.Parameter(torch.Tensor(3, 4))
        self.w2_relu = nn.ReLU()
    
    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x, p7_x = inputs
        
        # Calculate Top-Down Pathway
        w1_td = self.w1_relu(self.w1_td)
        w1_td /= torch.sum(w1_td, dim=0) + self.epsilon
        w2_up = self.w2_relu(self.w2_up)
        w2_up /= torch.sum(w2_up, dim=0) + self.epsilon
        
        print("p6_x",p6_x.shape)
        print("p7_x", p7_x.shape)
        print("interpolate", F.interpolate(p7_x, scale_factor=2).shape)
        p7_td = p7_x
        p6_td = self.p6_td(w1_td[0, 0] * p6_x + w1_td[1, 0] * F.interpolate(p7_td, scale_factor=2))        
        p5_td = self.p5_td(w1_td[0, 1] * p5_x + w1_td[1, 1] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1_td[0, 2] * p4_x + w1_td[1, 2] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1_td[0, 3] * p3_x + w1_td[1, 3] * F.interpolate(p4_td, scale_factor=2))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2_up[0, 0] * p4_x + w2_up[1, 0] * p4_td + w2_up[2, 0] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2_up[0, 1] * p5_x + w2_up[1, 1] * p5_td + w2_up[2, 1] * nn.Upsample(scale_factor=0.5)(p4_out))
        p6_out = self.p6_out(w2_up[0, 2] * p6_x + w2_up[1, 2] * p6_td + w2_up[2, 2] * nn.Upsample(scale_factor=0.5)(p5_out))
        p7_out = self.p7_out(w2_up[0, 3] * p7_x + w2_up[1, 3] * p7_td + w2_up[2, 3] * nn.Upsample(scale_factor=0.5)(p6_out))

        return [p3_out, p4_out, p5_out, p6_out, p7_out]
    
class BiFPN(nn.Module):
    def __init__(self, pretrained: bool, 
                output_feature_sizes: List[List[int]], 
                feature_size: int = 64,
                fpn_out_channels: int=128, 
                num_layers=3, 
                epsilon=0.0001):
        super(BiFPN, self).__init__()
        
        self.fpn_out_channels = fpn_out_channels
        self.out_channels = [self.fpn_out_channels for i in range(5)]
        self.output_feature_shape = output_feature_sizes
        
        self.resnet_out_channels = [64, 64, 64, 64, 64]
        # Get a pretrained ResNet34 model
        self.feature_extractor = nn.Sequential(*list(torchvision.models.resnet34(pretrained=pretrained).children())[:-5])
        
        self.feature_extractor.add_module("p3", nn.Sequential(
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
        ))

        self.feature_extractor.add_module("p4", nn.Sequential(
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
        ))

        self.feature_extractor.add_module("p5", nn.Sequential(
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
        ))

        self.feature_extractor.add_module("p6", nn.Sequential(
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
        ))

        self.feature_extractor.add_module("p7", nn.Sequential(
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
        ))

        self.fpn = torchvision.ops.FeaturePyramidNetwork(self.resnet_out_channels, self.fpn_out_channels)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpns)
    

    def forward(self, x):
        # Ignore four first "layers"/operations
        for i in range(5):
            x = self.feature_extractor[i](x)


        # pn_x = OrderedDict()
        
        # # Pass x through the resnet
        # pn_x['p3'] = self.feature_extractor[4](x)
        # pn_x['p4'] = self.feature_extractor[5](pn_x['p3'])
        # pn_x['p5'] = self.feature_extractor[6](pn_x['p4'])
        # pn_x['p6'] = self.feature_extractor[7](pn_x['p5'])
        # pn_x['p7'] = self.feature_extractor[8](pn_x['p6'])


        # Calculate the input column of BiFPN
        p3_x = self.feature_extractor[5](x)
        p4_x = self.feature_extractor[6](p3_x)
        p5_x = self.feature_extractor[7](p4_x)
        p6_x = self.feature_extractor[8](p5_x)
        p7_x = self.feature_extractor[9](p6_x)        
        
        features = [p3_x, p4_x, p5_x, p6_x, p7_x]
        i=3
        for el in features:
            print(f"sizes{i}", el.shape)
            i +=1

        # out_features = self.fpn(pn_x).values()
        out_features = self.bifpn(features)
        print("out features", len(out_features))
        for idx, feature in enumerate(out_features):
            h, w = self.output_feature_shape[idx]
            expected_shape = (self.fpn_out_channels, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)
