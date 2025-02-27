from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tops.torch_utils import set_seed
from math import ceil


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


def analyze_boxes(dataloader, cfg):
    label_map = {0: 'background', 1: 'car', 2: 'truck', 3: 'bus', 4: 'motorcycle', 5: 'bicycle', 6: 'scooter', 7: 'person', 8: 'rider'}
    aspect_ratios = []
    #dict_keys(['image', 'boxes', 'labels', 'width', 'height', 'image_id'])
    for batch in tqdm(dataloader):
        for box, lab in zip(batch['boxes'][0], batch['labels'][0]):
            if label_map[int(lab)] == 'person':
                aspect_ratios.append(float((box[3] - box[1]) * batch['height'] / ((box[2] - box[0]) * batch['width'])))

    print(type(aspect_ratios[0]))
    bins = np.array(range(0, 70)) / 10.0
    x = np.bincount(np.digitize(aspect_ratios, bins))[1:]

    print(bins)
    print(x)

    plt.figure()
    plt.bar(bins, x, width=0.1)
    plt.xticks(range(int(bins[0]), ceil(bins[-1])+1))
    plt.xlabel("Aspect Ratios Person")
    plt.ylabel("Count")
    
    # plt.waitforbuttonpress()
    plt.savefig("dataset_exploration/aspect_ratios_person.png")


def main():
    set_seed(42)
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_boxes(dataloader, cfg)


if __name__ == '__main__':
    main()
