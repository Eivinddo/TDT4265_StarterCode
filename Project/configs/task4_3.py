from ssd.modeling import backbones, SSDFocalLoss, AnchorBoxesCustom, RetinaNet
from tops.config import LazyCall as L
from ssd.modeling.backbones import FPN2
from .task2_3_v4 import (
    train,
    optimizer,
    schedulers,
    data_train, 
    data_val,
    label_map,
    val_cpu_transform,
    train_cpu_transform
)


anchors = L(AnchorBoxesCustom)(
    image_shape="${train.imshape}",
    aspect_ratios_per_size = 5,
    scale_center_variance=0.1,
    scale_size_variance=0.2,
    annotation_path='data/tdt4265_2022/train_annotations.json'
)


backbone = L(FPN2)(pretrained=True,
                  fpn_out_channels = 256,
                  anchors="${anchors}")

loss_objective = L(SSDFocalLoss)(anchors="${anchors}", gamma=2)

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8+1,  # Add 1 for background
    anchor_prob_initialization=True
)