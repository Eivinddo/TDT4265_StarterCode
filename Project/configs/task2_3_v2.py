from ssd.modeling import SSDFocalLoss
from tops.config import LazyCall as L
from .task2_3_v1 import (
    train, 
    anchors, 
    optimizer, 
    schedulers, 
    # loss_objective,
    backbone, 
    model, 
    data_train, 
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    label_map
)

loss_objective = L(SSDFocalLoss)(anchors="${anchors}", gamma=2)



