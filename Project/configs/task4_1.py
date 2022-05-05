from math import gamma
from ssd.modeling import backbones
from tops.config import LazyCall as L
from .task2_3_v3 import (
    train,
    optimizer,
    schedulers,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    label_map,
    anchors,
    model,
    # backbone,
    loss_objective
)

backbone = L(backbones.BiFPN)(pretrained=True,
                              output_feature_sizes="${anchors.feature_sizes}",
                              fpn_out_channels = 128)