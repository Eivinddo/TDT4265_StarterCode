from ssd.modeling import backbones
from tops.config import LazyCall as L
from .task2_2 import (
    train, 
    anchors, 
    optimizer, 
    schedulers, 
    loss_objective,
    model, 
    # backbone, 
    data_train, 
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    label_map
)

backbone = L(backbones.FPN)(pretrained=True,
                  fpn_out_channels = 256,
                  output_feature_sizes="${anchors.feature_sizes}")
