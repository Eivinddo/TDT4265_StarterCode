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
            
        print()

        self.anchors_xywh = torch.tensor(anchors).clamp(min=0, max=1).float()
        self.anchors_ltrb = self.anchors_xywh.clone()
        self.anchors_ltrb[:, 0] = self.anchors_xywh[:, 0] - 0.5 * self.anchors_xywh[:, 2]
        self.anchors_ltrb[:, 1] = self.anchors_xywh[:, 1] - 0.5 * self.anchors_xywh[:, 3]
        self.anchors_ltrb[:, 2] = self.anchors_xywh[:, 0] + 0.5 * self.anchors_xywh[:, 2]
        self.anchors_ltrb[:, 3] = self.anchors_xywh[:, 1] + 0.5 * self.anchors_xywh[:, 3]

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import sys
import os
import json
from sklearn.cluster import KMeans
from scipy import signal


def annotations_to_boxes(filename, rescale_width=None, rescale_height=None):
    """Extracts bounding-box widths and heights from ground-truth dataset.

    Args:
    filename : filename to a .json annotation file for your dataset.
    rescale_width : Scaling factor to rescale width of bounding box.
    rescale_height : Scaling factor to rescale height of bounding box.

    Returns:
    bboxes : A numpy array with pairs of box dimensions as [width, height].
    """

    dimensions_list = []
    file = open(filename)
    annot_data = json.load(file)
    for box in annot_data['annotations']:
        bbox = box['bbox']
        bbox_width = float(bbox[2])
        bbox_height = float(bbox[3])
        size = float(box['area'])
        #if rescale_width and rescale_height:
        #    size = root.find('size')
        #    bbox_width = bbox_width * (rescale_width / int(size.find('width').text))
        #    bbox_height = bbox_height * (rescale_height / int(size.find('height').text))
        dimensions_list.append([size, bbox_width, bbox_height])
    
    bboxes = np.array(dimensions_list)
    return bboxes


def average_iou(bboxes, anchors):
    """Calculates the Intersection over Union (IoU) between bounding boxes and
    anchors.

    Args:
    bboxes : Array of bounding boxes in [width, height] format.
    anchors : Array of aspect ratios [n, 2] format.

    Returns:
    avg_iou_perc : A Float value, average of IOU scores from each aspect ratio
    """
    intersection_width = np.minimum(anchors[:, [0]], bboxes[:, 0]).T
    intersection_height = np.minimum(anchors[:, [1]], bboxes[:, 1]).T

    if np.any(intersection_width == 0) or np.any(intersection_height == 0):
        raise ValueError("Some boxes have zero size.")

    intersection_area = intersection_width * intersection_height
    boxes_area = np.prod(bboxes, axis=1, keepdims=True)
    anchors_area = np.prod(anchors, axis=1, keepdims=True).T
    union_area = boxes_area + anchors_area - intersection_area
    avg_iou_perc = np.mean(np.max(intersection_area / union_area, axis=1)) * 100

    return avg_iou_perc


def kmeans_aspect_ratios(bboxes, kmeans_max_iter, num_aspect_ratios):
    """Calculate the centroid of bounding boxes clusters using Kmeans algorithm.

    Args:
    bboxes : Array of bounding boxes in [width, height] format.
    kmeans_max_iter : Maximum number of iterations to find centroids.
    num_aspect_ratios : Number of centroids to optimize kmeans.

    Returns:
    aspect_ratios : Centroids of cluster (optmised for dataset).
    avg_iou_prec : Average score of bboxes intersecting with new aspect ratios.
    """

    assert len(bboxes), "You must provide bounding boxes"

    normalized_bboxes = bboxes / np.sqrt(bboxes.prod(axis=1, keepdims=True))
  
    # Using kmeans to find centroids of the width/height clusters
    kmeans = KMeans(
        init='random', n_clusters=num_aspect_ratios, random_state=0, max_iter=kmeans_max_iter)
    kmeans.fit(X=normalized_bboxes)
    ar = kmeans.cluster_centers_

    assert len(ar), "Unable to find k-means centroid, try increasing kmeans_max_iter."

    avg_iou_perc = average_iou(normalized_bboxes, ar)

    if not np.isfinite(avg_iou_perc):
        sys.exit("Failed to get aspect ratios due to numerical errors in k-means")

    aspect_ratios = [w/h for w,h in ar]

    return aspect_ratios, avg_iou_perc


def analyze_ratios(annotation_path, num_aspect_ratios=6):
    
    # Tune the iterations based on the size and distribution of your dataset
    # You can check avg_iou_prec every 100 iterations to see how centroids converge
    kmeans_max_iter = 500

    # Get the ground-truth bounding boxes for our dataset
    bboxes = annotations_to_boxes(annotation_path)
    #bins = np.arange(0, 51060, 50)
    bins = np.logspace(0, 5, 5160//50)
    diff = np.diff(bboxes[:,0])
    hist, bin_edges = np.histogram(bboxes[:,0], bins)
    log_bin_edges = np.log(bin_edges[:-1])
    sos = signal.butter(3, 0.15, btype='highpass', output='sos')
    filtered = signal.sosfilt(sos, hist)
    filtered_best_inds = np.argsort(filtered)[-6:]
    
    peak_inds, _ = signal.find_peaks(filtered, distance=9)
    filter_peaks = filtered[peak_inds]
    filtered_peak_inds = peak_inds[np.argsort(filter_peaks)[-6:]]
    
    sizes = np.exp(log_bin_edges[filtered_peak_inds])
    
    debug = True
    if debug == True:
        print("Sizes:", sizes)
        plt.figure()
        plt.plot(log_bin_edges, filtered)
        plt.scatter(log_bin_edges[filtered_peak_inds], filtered[filtered_peak_inds], c='r')
        plt.figure()
        plt.plot(log_bin_edges, hist)
    
    bins = np.sort(sizes)
    centers = (bins[1:]+bins[:-1]) / 2
    digitize_inds = np.digitize(bboxes[:,0], centers)
    bboxes_per_size = [np.squeeze(bboxes[np.where(digitize_inds == idx),1:]) for idx in range(len(bins))]
    
    aspect_ratios_per_size = []
    avg_iou_perc_per_size = []
    for idx, b in enumerate(bboxes_per_size):
        aspect_ratios, avg_iou_perc =  kmeans_aspect_ratios(
                                            bboxes=b,
                                            kmeans_max_iter=kmeans_max_iter,
                                            num_aspect_ratios=num_aspect_ratios)
        aspect_ratios_per_size.append(sorted(aspect_ratios))
        avg_iou_perc_per_size.append(avg_iou_perc)

    #aspect_ratios = sorted(aspect_ratios)

    print('Sizes:', bins)
    print('Aspect ratios generated:', [[round(ar,2) for ar in aspect_ratios_per_size[i]] for i in range(len(aspect_ratios))])
    print('Average IOU with anchors:', avg_iou_perc_per_size)
    
    return sizes, aspect_ratios_per_size

class AnchorBoxesCustom(object):
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
                ax.scatter(size, r, s=70, facecolors='dimgray', edgecolors='k')
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

plt.figure()
with open('figures/box_size_and_ar.pkl', 'rb') as file:
    ax = pickle.load(file)
#ax = pickle.load('dataset_exploration/box_size_and_ar_zoom_zoom.pickle')

# anchors = AnchorBoxesTester2(
#     feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
#     # Strides is the number of pixels (in image space) between each spatial position in the feature map
#     strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
#     min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
#     aspect_ratios=[[0.5, 0.75, 1.3, 1.5, 2, 3, 4, 5, 6], [0.75, 1.3, 1.5, 1.8, 2, 2.5, 3, 4, 5], [2, 3], [1.5, 2], [1.5, 2], [1.5, 1.7]],
#     image_shape=(128,1024),
#     scale_center_variance=0.1,
#     scale_size_variance=0.2,
#     ax=ax
# )

anchors = AnchorBoxesCustom(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    aspect_ratios=[[0.5, 0.75, 1.3, 1.5, 2, 3, 4, 5, 6], [0.75, 1.3, 1.5, 1.8, 2, 2.5, 3, 4, 5], [2, 3], [1.5, 2], [1.5, 2], [1.5, 1.7]],
    image_shape=(128,1024),
    scale_center_variance=0.1,
    scale_size_variance=0.2,
    ax=ax
)

plt.show()
exit()
ax.set_ylim([0, 10])
ax.set_xlim([0, 40000])
ax.set_ylabel("Aspect ratio")
ax.set_xlabel("Box size")

plt.savefig('figures/anchor_box_coverage_improved.png')
plt.savefig('figures/anchor_box_coverage_improved.svg')
ax.set_ylim([0, 6])
ax.set_xlim([0, 5000])
plt.savefig('figures/anchor_box_coverage_improved_zoom.png')
plt.savefig('figures/anchor_box_coverage_improved_zoom.svg')

