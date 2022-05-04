# Modified from: https://github.com/lufficc/SSD
import torch
from typing import List
from math import sqrt
import pickle
import matplotlib.pyplot as plt

class AnchorBoxesTester(object):
    def __init__(self, 
            image_shape: tuple, 
            feature_sizes: List[tuple], 
            min_sizes: List[int],
            strides: List[tuple],
            aspect_ratios: List[int],
            scale_center_variance: float,
            scale_size_variance: float,
            ax: plt.Axes):
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
        self.num_boxes_per_fmap = [2 + 2*len(ratio) for ratio in aspect_ratios]
        # Calculation method slightly different from paper
        W = 1024
        H = 128
        anchors = []
        # size of feature and number of feature
        for fidx, [fH, fW] in enumerate(feature_sizes):
            print()
            print(f"===== Feature map {fidx} =====")
            print(f"Size of feature map: ({fH}, {fW})")
            bbox_sizes = []
            h_min = min_sizes[fidx][0] / image_shape[0]
            w_min = min_sizes[fidx][1] / image_shape[1]
            print(f"Box 1: ({round(w_min * W, 2)}, {round(h_min*H, 2)}). Size: {round(h_min * w_min * W * H, 2)}")
            ax.scatter(h_min * w_min * W * H, h_min*H/(w_min * W), s=70, facecolors='dimgray', edgecolors='k')
            bbox_sizes.append((w_min, h_min))
            h_max = sqrt(min_sizes[fidx][0]*min_sizes[fidx+1][0]) / image_shape[0]
            w_max = sqrt(min_sizes[fidx][1]*min_sizes[fidx+1][1]) / image_shape[1]
            print(f"Box 2: ({round(w_max * W, 2)}, {round(h_max*H, 2)}). Size: {round(h_max * w_max * W * H, 2)}")
            ax.scatter(h_max * w_max * W * H, (h_max*H)/(w_max * W), s=70, facecolors='dimgray', edgecolors='k')
            bbox_sizes.append((w_max, h_max))
            for ridx, r in enumerate(aspect_ratios[fidx]):
                h = h_min*sqrt(r)
                w = w_min/sqrt(r)
                bbox_sizes.append((w_min/sqrt(r), h_min*sqrt(r)))
                bbox_sizes.append((w_min*sqrt(r), h_min/sqrt(r)))
                print(f"Box {3+2*ridx}: ({round(w_min*W/sqrt(r), 2)}, {round(h_min*H*sqrt(r), 2)}). Size: {round((h_min*sqrt(r)) * (w_min/sqrt(r)) * W * H, 2)}")
                ax.scatter((h_min*sqrt(r)) * (w_min/sqrt(r)) * W * H, r, s=70, facecolors='dimgray', edgecolors='k')
                print(f"Box {4+2*ridx}: ({round(w_min*W*sqrt(r), 2)}, {round(h_min*H/sqrt(r), 2)}). Size: {round((h_min/sqrt(r)) * (w_min*sqrt(r)) * W * H, 2)}")
                ax.scatter((h_min/sqrt(r)) * (w_min*sqrt(r)) * W * H, 1/r, s=70, facecolors='dimgray', edgecolors='k')
            scale_y = image_shape[0] / strides[fidx][0]
            scale_x = image_shape[1] / strides[fidx][1]
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

with open('figures/box_size_and_ar.pkl', 'rb') as file:
    ax = pickle.load(file)

anchors = AnchorBoxesTester(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    #min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    min_sizes=[[14, 10], [26, 24], [40, 40], [60, 60], [84, 88], [120, 136], [134, 420]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    #aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    aspect_ratios=[[3, 4], [2, 4], [2, 3], [2, 3], [2, 3], [2, 3]],
    image_shape=(128, 1024),
    scale_center_variance=0.1,
    scale_size_variance=0.2,
    ax=ax
)

# Original
# anchors = AnchorBoxesTester(
#     feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
#     # Strides is the number of pixels (in image space) between each spatial position in the feature map
#     strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
#     min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
#     aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
#     image_shape=(128, 1024),
#     scale_center_variance=0.1,
#     scale_size_variance=0.2,
#     ax=ax
# )

ax.set_ylim([0, 10])
ax.set_xlim([0, 40000])
ax.set_ylabel("Aspect ratio")
ax.set_xlabel("Box size")

plt.savefig('figures/anchor_box_coverage_original.png')
plt.savefig('figures/svgs/anchor_box_coverage_original.svg')
ax.set_ylim([0, 6])
ax.set_xlim([0, 5000])
plt.savefig('figures/anchor_box_coverage_original_zoom.png')
plt.savefig('figures/svgs/anchor_box_coverage_original_zoom.svg')
ax.set_ylim([0, 12])
ax.set_xlim([0, 55000])
plt.show()
