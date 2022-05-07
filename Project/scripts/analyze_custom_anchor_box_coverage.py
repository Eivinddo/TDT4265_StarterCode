import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil
import sys
import os
import json
from sklearn.cluster import KMeans
from scipy import signal


def annotations_to_boxes(filename, rescale_width=None, rescale_height=None):
    dimensions_list = []
    file = open(filename)
    
    annot_data = json.load(file)
    for box in annot_data['annotations']:
        bbox = box['bbox']
        bbox_width = float(bbox[2])
        bbox_height = float(bbox[3])
        size = float(box['area'])
        dimensions_list.append([size, bbox_width, bbox_height])
    
    bboxes = np.array(dimensions_list)
    return bboxes


def kmeans_aspect_ratios(bboxes, kmeans_max_iter, num_aspect_ratios):
    normalized_bboxes = bboxes / np.sqrt(bboxes.prod(axis=1, keepdims=True))
  
    # Using kmeans to find centroids of the width/height clusters
    kmeans = KMeans(
        init='random', n_clusters=num_aspect_ratios, random_state=0, max_iter=kmeans_max_iter)
    kmeans.fit(X=normalized_bboxes)
    aspect_ratios_wh = kmeans.cluster_centers_

    # The aspect ratio is height/width which is opposite of common definition, but we made a 
    # mistake early on, so throughout the code, h/w is aspect ratio.
    aspect_ratios = [h/w for w,h in aspect_ratios_wh]

    return aspect_ratios


def analyze_sizes(bboxes):
    # bboxes are Nx3 ([size, bbox_width, bbox_height])
    
    # Divide the range of possible sizes into logarithmic bins
    bins = np.logspace(0, 5, 5160//50)
    # Divide the different bbox sizes into the bins and find edges
    hist, bin_edges = np.histogram(bboxes[:,0], bins)
    log_bin_edges = np.log(bin_edges[:-1])
    
    # High-pass filter in order to highlight the peaks
    sos = signal.butter(3, 0.15, btype='highpass', output='sos')
    filtered = signal.sosfilt(sos, hist)
    
    # Find peaks in the high-passed distribution
    peak_inds, _ = signal.find_peaks(filtered, distance=10)
    filter_peaks = filtered[peak_inds]
    # Find top 6 peaks
    filtered_peak_inds = peak_inds[np.argsort(filter_peaks)[-6:]]
    
    # Get the top 6 sizes based on the exponential of the bins and the peaks
    sizes = np.exp(log_bin_edges[filtered_peak_inds])
    
    # Debug printing and plotting
    debug = True
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
    
    return sizes


def analyze_ratios(annotation_path, num_aspect_ratios=6):
    # Get the gt bounding boxes for our dataset
    bboxes = annotations_to_boxes(annotation_path)
    
    # Get the different dominant box sizes
    sizes = analyze_sizes(bboxes)
    print("sizes",sizes)
    
    # Sort the sizes from small to largest "feature_map" and find center of bins
    bins = np.sort(sizes)
    centers = (bins[1:]+bins[:-1]) / 2
    digitize_inds = np.digitize(bboxes[:,0], centers)
    # Find all the bboxes that correspond to each of the sizes.
    # This is in order to afterwards find the ideal aspect ratios of the
    # bboxes at each size-level.
    bboxes_per_size = [np.squeeze(bboxes[np.where(digitize_inds == idx),1:]) for idx in range(len(bins))]
    
    max_iter = 500  # Could maybe be found in a better way than hardcoding
    
    aspect_ratios_per_size = []
    for b in bboxes_per_size:
        aspect_ratios =  kmeans_aspect_ratios(
            bboxes=b,
            kmeans_max_iter=max_iter,
            num_aspect_ratios=num_aspect_ratios)
        aspect_ratios_per_size.append(sorted(aspect_ratios))
    
    return bins, aspect_ratios_per_size


class AnchorBoxesCustom(object):
    def __init__(self, 
            image_shape: tuple,
            aspect_ratios_per_feature_map: int,
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

        annotation_path = os.path.join(os.getcwd(), Path('data/tdt4265_2022/train_annotations.json'))
        
        sizes, aspect_ratios_per_size = analyze_ratios(annotation_path, aspect_ratios_per_feature_map)
        self.num_boxes_per_fmap = [2 + len(ratios) for ratios in aspect_ratios_per_size]
        self.aspect_ratios = [[round(ar, 3) for ar in aspect_ratios] for aspect_ratios in aspect_ratios_per_size]
        self.feature_sizes = []
        strides = []

        anchors = []
        # size of feature and number of feature
        for sidx, size in enumerate(sizes):
            bbox_sizes = []
            
            wh_side = sqrt(size)
            h = wh_side / image_shape[0]
            w = wh_side / image_shape[1]
            bbox_sizes.append((w, h))
            ax.scatter(w*h*image_shape[0]*image_shape[1], 1, s=70, facecolors='dimgray', edgecolors='k')
            bbox_sizes.append((w*sqrt(2), h*sqrt(2)))
            ax.scatter(w*h*2*image_shape[0]*image_shape[1], 1, s=70, facecolors='dimgray', edgecolors='k')
            
            for r in self.aspect_ratios[sidx]:
                w = sqrt(size / r)
                h = r * w
                h = h / image_shape[0]
                w = w / image_shape[1]
                bbox_sizes.append((w, h))
                
                ax.scatter(w*h*image_shape[0]*image_shape[1], r, s=70, facecolors='dimgray', edgecolors='k')
            
            square = sqrt(size)
            fH = round(image_shape[0] / square)
            fW = round(image_shape[1] / square)
            self.feature_sizes.append([fH, fW])
            stride_y = image_shape[0] / fH
            stride_x = image_shape[1] / fW
            strides.append([round(stride_y, 3), round(stride_x, 3)])
            scale_y = image_shape[0] / stride_y
            scale_x = image_shape[1] / stride_x
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
    
        print("\n\nCONFIGS:")
        print("feature_sizes:", self.feature_sizes)
        print("strides:", strides)
        print("aspect_ratios:", self.aspect_ratios)
        print("image_shape:", image_shape)

plt.figure()
with open('figures/box_size_and_ar.pkl', 'rb') as file:
    ax = pickle.load(file)

anchors = AnchorBoxesCustom(
    image_shape=(128,1024),
    aspect_ratios_per_feature_map=5,
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

