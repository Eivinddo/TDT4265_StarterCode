from turtle import color
from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import numpy as np
import pickle
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


def analyze_boxes(dataloader, cfg):
    N = len(dataloader)
    label_map = {0: 'background', 1: 'car', 2: 'truck', 3: 'bus', 4: 'motorcycle', 5: 'bicycle', 6: 'scooter', 7: 'person', 8: 'rider'}
    label_color_map = {1: 'red', 2: 'peru', 3: 'darkolivegreen', 4: 'turquoise', 5: 'purple', 6: 'hotpink', 7: 'blue', 8: 'lime'}
    classes = list(label_map.values())[1:]
    
    box_data = []

    #dict_keys(['image', 'boxes', 'labels', 'width', 'height', 'image_id'])
    plt.figure()
    for i, batch in enumerate(tqdm(dataloader)):
        for k, box in enumerate(batch['boxes'][0]):
            box_size = float((box[2] - box[0])*batch['width'] * (box[3] - box[1])*batch['height'])
            aspect_ratio = float((box[3] - box[1]) * batch['height'] / ((box[2] - box[0]) * batch['width']))
            label = int(batch['labels'][0][k])
            box_data.append([box_size, aspect_ratio, label])
    
    box_data = np.array(box_data)

    for lab_num in label_color_map.keys():
        plt.scatter(box_data[box_data[:,2]==lab_num, 0], box_data[box_data[:,2]==lab_num, 1], marker='x', s=1, color=label_color_map[lab_num], label=label_map[lab_num])
    
    plt.legend()
    plt.ylabel("Aspect ratio")
    plt.xlabel("Box size")
    
    # plt.waitforbuttonpress()
    plt.xlim([0, 55000])
    plt.ylim([0, 12])
    plt.savefig("figures/box_size_and_ar.png")
    plt.savefig("figures/svgs/box_size_and_ar.svg")
    ax = plt.gca()
    with open('figures/box_size_and_ar.pkl','wb') as file:
        pickle.dump(ax, file)
    
    plt.xlim([0, 10000])
    plt.ylim([0, 6])
    plt.savefig("figures/box_size_and_ar_zoom.png")
    plt.savefig("figures/svgs/box_size_and_ar_zoom.svg")
    plt.xlim([0, 1000])
    plt.ylim([0, 6])
    plt.savefig("figures/box_size_and_ar_zoom_zoom.png")
    plt.savefig("figures/svgs/box_size_and_ar_zoom_zoom.svg")


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
