# from math import gamma
# from ssd.modeling import backbones, SSDFocalLoss, AnchorBoxes, RetinaNet
# from tops.config import LazyCall as L
# from ssd.modeling.backbones import FPN
# from .task2_3_v4 import (
#     train,
#     optimizer,
#     schedulers,
#     data_train, 
#     data_val,
#     label_map,
#     val_cpu_transform,
#     train_cpu_transform
# )


# anchors = L(AnchorBoxes)(
#     feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
#     # Strides is the number of pixels (in image space) between each spatial position in the feature map
#     strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
#     min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
#     aspect_ratios=[[1.8, 3.5], [1.8, 3.5], [1.8, 3.5], [1.8, 3.5], [1.8, 3.5], [1.8, 3.5]],
#     image_shape="${train.imshape}",
#     scale_center_variance=0.1,
#     scale_size_variance=0.2
# )


# backbone = L(FPN)(pretrained=True,
#                   fpn_out_channels = 256,
#                   output_feature_sizes="${anchors.feature_sizes}")

# loss_objective = L(SSDFocalLoss)(anchors="${anchors}", gamma=2)

# model = L(RetinaNet)(
#     feature_extractor="${backbone}",
#     anchors="${anchors}",
#     loss_objective="${loss_objective}",
#     num_classes=8+1,  # Add 1 for background
#     anchor_prob_initialization=True
# )

from math import gamma
from ssd.modeling import backbones, SSDFocalLoss, AnchorBoxes2, RetinaNet2
from tops.config import LazyCall as L
from ssd.modeling.backbones import FPN
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


anchors = L(AnchorBoxes2)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[12, 12], [24, 24], [40, 40], [60, 60], [84, 88], [120, 136], [134, 420]],
    aspect_ratios=[[2, 3, 4 ,5], [2, 3, 4, 5], [2, 3], [2, 3], [1.5, 3], [1.5, 3]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)


backbone = L(FPN)(pretrained=True,
                  fpn_out_channels = 256,
                  output_feature_sizes="${anchors.feature_sizes}")

loss_objective = L(SSDFocalLoss)(anchors="${anchors}", gamma=2)

model = L(RetinaNet2)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8+1,  # Add 1 for background
    anchor_prob_initialization=True
)