from math import gamma
from ssd.modeling import SSDFocalLoss
from tops.config import LazyCall as L
from .task2_3_v1 import (
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
    backbone
)

loss_objective = L(SSDFocalLoss)(anchors="${anchors}", gamma=2)



