from cv2 import log
from numpy import pad
from torch import batch_norm
import torch
import torch.nn as nn
import torchvision

# Notation from "Focal Loss for Dense Object Detection"
A = 6       # Num anchors at each feature map
K = 9       # Number of classes
C = 256     # Number of channels per feature map

a = nn.ModuleList([nn.Conv2d(C, C, kernel_size=3, padding=1),nn.ReLU(inplace=True) ,nn.Conv2d(C, C, kernel_size=3, padding=1)])
# print(a)
# print(a[:])



classification_heads = nn.Sequential(
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, C, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(C, K*A, kernel_size=3, padding=1)
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
    nn.Conv2d(C, 4*A, kernel_size=3, padding=1)
)

layers = [regression_heads, classification_heads]
modules = nn.ModuleList(layers)

reg_mod = nn.ModuleList(regression_heads)
class_mod = nn.ModuleList(classification_heads)
# nn.init.constant_(reg_mod[:].bias.data, 0)



# nn.init.constant_(self.classification_heads[:].bias.data, 0)

# nn.init.constant_(self.classification_heads[-1].bias.data[:4], math.log(0.99 * (8 / 0.1)))

for layer in class_mod:
    if hasattr(layer, "weight"):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        print(layer.weight.data)
    if hasattr(layer, "bias"):
        nn.init.constant_(layer.bias.data, 0)
        #print(layer.bias.data)


for layer in reg_mod:
    if hasattr(layer, "weight"):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    if hasattr(layer, "bias"):
        nn.init.constant_(layer.bias.data, 0)
        #print(layer.bias.data)

p = 0.99
b = torch.log(torch.tensor(p*((K-1)/(1-p))))
nn.init.constant_(class_mod[-1].bias.data[:A],b)

# for layer in reg_mod:
#     if hasattr(layer, "bias"):
#         print(layer.bias)



# module_children = list(classification_heads.children())
# print(module_children)
# named_params = list(module_children[-2].named_parameters())

# print(named_params)
# bias = named_params[1]
# print("Bias pre")
# print(bias)
# print(type(bias))
# nn.init.zeros_(bias[1])
# print("Eivin")
# bias[1][0] = 1
# print(bias[1])
# bias[1][0:-1:A] = 1

# print("Bias post")
# print(bias)
# print(len(bias[1]))

# for layer in layers:
#     for param in layer.parameters():
#         if param.dim() > 1: nn.init.xavier_uniform_(param)