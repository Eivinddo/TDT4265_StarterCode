from math import gamma
from ssd.modeling import backbones, SSDFocalLoss, AnchorBoxes, RetinaNet
from tops.config import LazyCall as L
from ssd.modeling.backbones import FPN
from .task2_3_v4 import (
    train,
    optimizer,
    schedulers,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    label_map,
    anchors,
    backbone,
    loss_objective,
    model
)

# The only change is the alpha in ssd_focal_loss.py