from math import gamma
from ssd.modeling import RetinaNet
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
    backbone,
    loss_objective
)


model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8+1,  # Add 1 for background
    anchor_prob_initialization=True
)