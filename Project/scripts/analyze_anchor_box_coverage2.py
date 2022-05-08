# Modified version of analyze anchor box coverage in order 
# to test the more advanced (and manual) placement of anchor
# boxes in task 2.4
import torch
from typing import List
from math import sqrt
import pickle
import matplotlib.pyplot as plt


class AnchorBoxesTester2(object):
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
        #self.num_boxes_per_fmap = [2 + 2*len(ratio) for ratio in aspect_ratios]
        i = 0
        for ratio in aspect_ratios:
            if i < 2:
                self.num_boxes_per_fmap = [2 + 1*len(ratio)]
            else:
                self.num_boxes_per_fmap = [2 + 2*len(ratio)]
            i += 1
        # Calculation method slightly different from paper
        H, W = image_shape
        image_shape
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
                if fidx < 4:
                    print(f"Box {3+2*ridx}: ({round(w_min*W/sqrt(r), 2)}, {round(h_min*H*sqrt(r), 2)}). Size: {round((h_min*sqrt(r)) * (w_min/sqrt(r)) * W * H, 2)}")
                    ax.scatter((h_min*sqrt(r)) * (w_min/sqrt(r)) * W * H, r, s=70, facecolors='dimgray', edgecolors='k')
                elif ridx < (len(aspect_ratios[fidx]) - 1):
                    print(f"Box {3+2*ridx}: ({round(w_min*W/sqrt(r), 2)}, {round(h_min*H*sqrt(r), 2)}). Size: {round((h_min*sqrt(r)) * (w_min/sqrt(r)) * W * H, 2)}")
                    ax.scatter((h_min*sqrt(r)) * (w_min/sqrt(r)) * W * H, r, s=70, facecolors='dimgray', edgecolors='k')
                #print(f"Box {4+2*ridx}: ({round(w_min*W*sqrt(r), 2)}, {round(h_min*H/sqrt(r), 2)}). Size: {round((h_min/sqrt(r)) * (w_min*sqrt(r)) * W * H, 2)}")
                #ax.scatter((h_min/sqrt(r)) * (w_min*sqrt(r)) * W * H, 1/r, s=70, facecolors='dimgray', edgecolors='k')
                if fidx >= 2:
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


plt.figure()
with open('figures/box_size_and_ar.pkl', 'rb') as file:
    ax = pickle.load(file)

anchors = AnchorBoxesTester2(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4/2, 4/2], [8/2, 8/2], [16/2, 16/2], [32/2, 32/2], [64, 64], [128, 128]],
    min_sizes=[[16/2, 16/2], [32/2, 32/2], [48/2, 48/2], [64, 64], [86, 86], [128, 128], [128, 400]],
    aspect_ratios=[[0.5, 0.75, 1.3, 1.5, 2, 3, 4, 5, 6], [0.75, 1.3, 1.5, 1.8, 2, 2.5, 3, 4, 5], [2, 3], [1.5, 2], [1.5, 2], [1.2, 1.5]],
    image_shape=(128,1024),
    scale_center_variance=0.1,
    scale_size_variance=0.2,
    ax=ax
)

plt.show()
