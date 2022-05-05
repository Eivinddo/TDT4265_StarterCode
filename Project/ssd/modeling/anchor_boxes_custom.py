# Modified from: https://github.com/lufficc/SSD
import torch
from typing import List
from math import sqrt
from scripts.anchor_specialization_k_means import analyze_ratios


class AnchorBoxesCustom(object):
    def __init__(self, 
            image_shape: tuple, 
            feature_sizes: List[tuple], 
            min_sizes: List[int],
            strides: List[tuple],
            aspect_ratios: List[int],
            scale_center_variance: float,
            scale_size_variance: float):
        """Generate SSD anchors Boxes.
            It returns the center, height and width of the anchors. The values are relative to the image size
            Args:
                image_shape: tuple of (image height, width)
                feature_sizes: each tuple in the list is the feature shape outputted by the backbone (H, W)
            Returns:
                anchors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        self.scale_center_variance = scale_center_variance
        self.scale_size_variance = scale_size_variance
        self.num_boxes_per_fmap = [6 for ratio in aspect_ratios]

        annotation_path = "data/tdt4265_2022/train_annotations.json"

        sizes, aspect_ratios_per_size = analyze_ratios(annotation_path, 6)

        anchors = []
        # size of feature and number of feature
        for sidx, size in enumerate(sizes):
            bbox_sizes = []
            #aspect_ratios = aspect_ratios_per_size[sidx]
            
            for r in aspect_ratios_per_size[sidx]:
                h = sqrt(size / r)
                w = r * h
                h = h / image_shape[0]
                w = w / image_shape[1]
                bbox_sizes.append((w, h))
            
            square = sqrt(size)
            fH = round(image_shape[0] / square)
            fW = round(image_shape[1] / square)
            scale_y = fH / 2
            scale_x = fW / 2
            for w, h in bbox_sizes:
                for i in range(fH):
                    for j in range(fW):
                        cx = (j + 0.5)/scale_x
                        cy = (i + 0.5)/scale_y
                        anchors.append((cx, cy, w, h))

        print()

        self.anchors_xywh = torch.tensor(anchors).clamp(min=0, max=1).float()
        self.anchors_ltrb = self.anchors_xywh.clone()
        self.anchors_ltrb[:, 0] = self.anchors_xywh[:, 0] - 0.5 * self.anchors_xywh[:, 2]
        self.anchors_ltrb[:, 1] = self.anchors_xywh[:, 1] - 0.5 * self.anchors_xywh[:, 3]
        self.anchors_ltrb[:, 2] = self.anchors_xywh[:, 0] + 0.5 * self.anchors_xywh[:, 2]
        self.anchors_ltrb[:, 3] = self.anchors_xywh[:, 1] + 0.5 * self.anchors_xywh[:, 3]

    def __call__(self, order):
        if order == "ltrb":
            return self.anchors_ltrb
        if order == "xywh":
            return self.anchors_xywh

    @property
    def scale_xy(self):
        return self.scale_center_variance

    @property
    def scale_wh(self):
        return self.scale_size_variance
