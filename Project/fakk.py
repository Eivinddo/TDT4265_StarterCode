from numpy import pad
from torch import batch_norm
import torch
import torch.nn as nn
import torchvision


print("==================================================================================================================")
print("==================================================================================================================")

feature_extractor = nn.Sequential(*list(torchvision.models.resnet34(pretrained=True).children())[:-2])

feature_extractor.add_module("layer5", nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),    # Downsample
    nn.ReLU()
))
feature_extractor.add_module("layer6", nn.Sequential(
    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),    # Downsample
    nn.ReLU()
))

print(feature_extractor)

a = feature_extractor(torch.randn(32,3,128,1024))
print(a.shape)

