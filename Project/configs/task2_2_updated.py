# Inherit configs from the default ssd300
import torchvision
from ssd.data import TDT4265Dataset
from ssd.modeling import backbones, SSDFocalLoss, AnchorBoxes2, RetinaNet2
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomSampleCrop, GroundTruthBoxesToAnchors, 
    RandomBrightness, RandomContrast)
from .tdt4265 import (
    train, 
    anchors, 
    optimizer, 
    schedulers, 
    backbone, 
    model, 
    data_train, 
    data_val, 
    loss_objective )
from .utils import get_dataset_dir

anchors = L(AnchorBoxes2)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    aspect_ratios=[[4, 1.5], [4, 1.5], [2, 3], [2, 3], [1.5, 3], [1.5, 3]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

# Keep the model, except change the backbone and number of classes
train.imshape = (128, 1024)
train.image_channels = 3
model.num_classes = 8 + 1  # Add 1 for background class


train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
    L(RandomHorizontalFlip)(),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])
val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
])
train_gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomContrast)(),
    L(RandomBrightness)(),
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
])

val_gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
])

data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022_updated"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022_updated/train_annotations.json"))

data_val.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022_updated"),
    transform="${val_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022_updated/val_annotations.json"))

data_val.gpu_transform = val_gpu_transform
data_train.gpu_transform = train_gpu_transform

label_map = {idx: cls_name for idx, cls_name in enumerate(TDT4265Dataset.class_names)}
