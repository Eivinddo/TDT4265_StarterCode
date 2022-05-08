from .task2_3_v4 import (
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
    loss_objective,
    model
)
from .utils import get_dataset_dir
data_train.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_train.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/train_annotations.json")
data_val.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_val.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/val_annotations.json")
