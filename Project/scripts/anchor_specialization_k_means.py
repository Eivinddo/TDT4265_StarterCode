import numpy as np
import matplotlib.pyplot as plt
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
    
    return sizes


def analyze_ratios(annotation_path, num_aspect_ratios=6):
    # Get the gt bounding boxes for our dataset
    bboxes = annotations_to_boxes(annotation_path)
    
    # Get the different dominant box sizes
    sizes = analyze_sizes(bboxes)
    
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

def main():
    annotation_path = "data/tdt4265_2022/train_annotations.json"

    analyze_ratios(annotation_path)


if __name__ == '__main__':
    main()
