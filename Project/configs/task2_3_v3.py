from ssd.modeling import RetinaNet
from tops.config import LazyCall as L
from .task2_3_v2 import (
    train, 
    anchors, 
    optimizer, 
    schedulers, 
    loss_objective,
    backbone, 
    # model, 
    data_train, 
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    label_map
)

anchors.aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8+1,
    anchor_prob_initialization=False # Weight initialization for Task 2.3.4
)
