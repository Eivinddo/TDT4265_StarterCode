from math import gamma
from ssd.modeling import backbones, SSDFocalLoss, AnchorBoxes
from tops.config import LazyCall as L
from .task2_2 import (
    train,
    optimizer,
    schedulers,
    anchors,
    model,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    label_map,
    loss_objective
)


backbone = L(backbones.FPN)(pretrained=True,
                  fpn_out_channels = 256,
                  output_feature_sizes="${anchors.feature_sizes}")

