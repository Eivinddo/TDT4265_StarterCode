import torchvision
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomSampleCrop, GroundTruthBoxesToAnchors, 
    RandomBrightness, RandomContrast)
from .tdt4265 import (
    train, 
    anchors, 
    optimizer, 
    schedulers, 
    loss_objective,
    model, 
    backbone, 
    data_train, 
    data_val,
    # train_cpu_transform,
    val_cpu_transform,
    label_map
)

train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
    L(RandomHorizontalFlip)(),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])

train_gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomContrast)(),
    L(RandomBrightness)(),
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
])

data_train.gpu_transform = train_gpu_transform
