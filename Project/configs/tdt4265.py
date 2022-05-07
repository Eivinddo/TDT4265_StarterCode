import torchvision
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from .utils import get_dataset_dir
from ssd.data.transforms import (
    ToTensor, Normalize, Resize, GroundTruthBoxesToAnchors)
from .ssd300 import (
    train, 
    anchors, 
    optimizer, 
    schedulers,
    loss_objective, 
    model,
    backbone, 
    data_train, 
    data_val,
    # label_map
)

train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
    # GroundTruthBoxesToAnchors assigns each ground truth to anchors, required to compute loss in training.
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])

val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
])

data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    annotation_file=get_dataset_dir("tdt4265_2022/train_annotations.json"),
    transform="${train_cpu_transform}")

data_val.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    annotation_file=get_dataset_dir("tdt4265_2022/val_annotations.json"),
    transform="${val_cpu_transform}")

gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
])
  
data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform

label_map = {idx: cls_name for idx, cls_name in enumerate(TDT4265Dataset.class_names)}
