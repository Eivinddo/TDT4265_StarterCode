from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tops.torch_utils import set_seed


def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_something(dataloader, cfg):
    N = len(dataloader)
    label_map = {0: 'background', 1: 'car', 2: 'truck', 3: 'bus', 4: 'motorcycle', 5: 'bicycle', 6: 'scooter', 7: 'person', 8: 'rider'}
    classes = list(label_map.values())[1:]  # Drop background
    num_boxes_per_class = np.empty((N, len(label_map)-1))
    
    for i, batch in enumerate(tqdm(dataloader)):
        for j in range(1, 8+1):
            num_boxes_per_class[i, j-1] = np.sum(np.array(batch['labels'])==j)
    
    plt.figure()
    bars = plt.bar(range(len(classes)), np.sum(num_boxes_per_class, axis=0))
    bar_height_max = 0
    for bar in bars:
        height = bar.get_height()
        if height > bar_height_max: bar_height_max = height
        plt.text(bar.get_x() + bar.get_width()/2., height+10,
                '%d' % int(height),
                ha='center', va='bottom')
    
    plt.xticks(range(len(classes)), ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'])
    plt.ylabel("Number of boxes")
    ax = plt.gca()
    plt.ylim(0, bar_height_max*1.1)
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
    
    # plt.waitforbuttonpress()
    plt.savefig("dataset_exploration/distribution.png")
    plt.savefig("dataset_exploration/distribution.svg")


def main():
    set_seed(42)
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
