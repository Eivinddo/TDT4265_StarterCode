from cv2 import log
from numpy import pad
from torch import batch_norm
import torch
import torch.nn as nn
import torchvision

# Notation from "Focal Loss for Dense Object Detection"
A = 9       # Num anchors at each feature map
K = 9       # Number of classes
C = 256     # Number of channels per feature map

classification_heads = nn.Sequential(
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, K*A, kernel_size=3, padding=1),
    nn.Sigmoid()
)

regression_heads = nn.Sequential(
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, 4*A, kernel_size=3, padding=1),
    nn.Sigmoid()
)

layers = [regression_heads, classification_heads]
module_children = list(regression_heads.children())
print(module_children)
named_params = list(module_children[0].named_parameters())
print(named_params)
for layer in layers:
    for param in layer.parameters():
        if param.dim() > 1: nn.init.xavier_uniform_(param)