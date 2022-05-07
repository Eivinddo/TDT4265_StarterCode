import torch
import torch.nn as nn
import torchvision
from typing import OrderedDict
from torch.nn import functional as F


class FPN2(nn.Module):
    """
    This is a basic backbone for RetinaNet - Feature Pyramid network based on ResNet.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 64, 64),
     shape(-1, output_channels[1], 128, 128),
     shape(-1, output_channels[2], 256, 256),
     shape(-1, output_channels[3], 512, 512),
     shape(-1, output_channels[3], 1024, 1024),
     shape(-1, output_channels[4], 2048, 2048)]
    """
    def __init__(self,
                 pretrained: bool,
                 fpn_out_channels: int,
                 anchors):
        super().__init__()
        
        self.fpn_out_channels = fpn_out_channels
        self.out_channels = [self.fpn_out_channels for i in range(6)]
        self.output_feature_shape = anchors.feature_sizes
        
        self.resnet_out_channels = [64, 128, 256, 512, 1024, 2048]
        # Get a pretrained ResNet34 model
        self.feature_extractor = nn.Sequential(*list(torchvision.models.resnet34(pretrained=pretrained).children())[:-2])

        self.feature_extractor.add_module("layer5", nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),    # Downsample
            nn.BatchNorm2d(1024),
            nn.ReLU()
        ))
        self.feature_extractor.add_module("layer6", nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),    # Downsample
            nn.BatchNorm2d(2048),
            nn.ReLU()
        ))
        
        self.fpn = torchvision.ops.FeaturePyramidNetwork(self.resnet_out_channels, self.fpn_out_channels)
    
    
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """ 
        
        # Ignore four first "layers"/operations
        for i in range(4):
            x = self.feature_extractor[i](x)
        
        pyramid = OrderedDict()
        
        # Pass x through the resnet
        x = self.feature_extractor[4](x)
        x = F.interpolate(x, size=self.output_feature_shape[0])
        pyramid['feat0'] = x
        
        x = self.feature_extractor[5](x)
        x = F.interpolate(x, size=self.output_feature_shape[1])
        pyramid['feat1'] = x
        
        x = self.feature_extractor[6](x)
        x = F.interpolate(x, size=self.output_feature_shape[2])
        pyramid['feat2'] = x
        
        x = self.feature_extractor[7](x)
        x = F.interpolate(x, size=self.output_feature_shape[3])
        pyramid['feat3'] = x
        
        x = self.feature_extractor[8](x)
        x = F.interpolate(x, size=self.output_feature_shape[4])
        pyramid['feat4'] = x
        
        x = self.feature_extractor[9](x)
        x = F.interpolate(x, size=self.output_feature_shape[5])
        pyramid['feat5'] = x
        

        out_features = self.fpn(pyramid).values()
        
        for idx, feature in enumerate(out_features):
            h, w = self.output_feature_shape[idx]
            expected_shape = (self.fpn_out_channels, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

