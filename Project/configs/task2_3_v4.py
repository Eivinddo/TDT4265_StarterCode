from .task2_3_v3 import (
    train, 
    anchors, 
    optimizer, 
    schedulers, 
    loss_objective,
    backbone, 
    model, 
    data_train, 
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    label_map
)

model.anchor_prob_initialization=True
