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
    #file = open(filename)
    file = open('data/tdt4265_2022/train_annotations.json')
    annot_data = json.load(file)
    for box in annot_data['annotations']:
        bbox = box['bbox']
        bbox_width = float(bbox[2])
        bbox_height = float(bbox[3])
        size = float(box['area'])
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

    aspect_ratios = [h/w for w,h in ar]

    return aspect_ratios, avg_iou_perc


def analyze_ratios(annotation_path, num_aspect_ratios=6):
    
    # Tune the iterations based on the size and distribution of your dataset
    # You can check avg_iou_prec every 100 iterations to see how centroids converge
    kmeans_max_iter = 500

    # Get the ground-truth bounding boxes for our dataset
    bboxes = annotations_to_boxes(annotation_path)
    #bins = np.arange(0, 51060, 50)
    bins = np.logspace(0, 5, 5160//50)
    hist, bin_edges = np.histogram(bboxes[:,0], bins)
    log_bin_edges = np.log(bin_edges[:-1])
    sos = signal.butter(3, 0.15, btype='highpass', output='sos')
    filtered = signal.sosfilt(sos, hist)
    
    peak_inds, _ = signal.find_peaks(filtered, distance=10)
    filter_peaks = filtered[peak_inds]
    filtered_peak_inds = peak_inds[np.argsort(filter_peaks)[-6:]]
    
    sizes = np.exp(log_bin_edges[filtered_peak_inds])
    
    debug = False
    if debug == True:
        print("Sizes:", sizes)
        plt.figure()
        plt.plot(log_bin_edges, filtered)
        plt.scatter(log_bin_edges[filtered_peak_inds], filtered[filtered_peak_inds], c='r')
        plt.title("Highpass filtered distribution of box sizes (logarithmic)")
        plt.xlabel("Logarithm of bbox sizes ( log(size) )")
        plt.ylabel("Number of bboxes - Highpass filtered")
        plt.savefig("figures/distribution_of_log_sizes_hp.png")
        plt.savefig("figures/svgs/distribution_of_log_sizes_hp.svg")
        plt.figure()
        plt.plot(log_bin_edges, hist)
        plt.title("Distribution of box sizes (logarithmic)")
        plt.xlabel("Logarithm of bbox sizes ( log(size) )")
        plt.ylabel("Number of bboxes")
        plt.savefig("figures/distribution_of_log_sizes.png")
        plt.savefig("figures/svgs/distribution_of_log_sizes.svg")
        plt.show()
    
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
    
    return bins, aspect_ratios_per_size

def main():
    annotation_path = "data/tdt4265_2022/train_annotations.json"

    analyze_ratios(annotation_path)


if __name__ == '__main__':
    main()
