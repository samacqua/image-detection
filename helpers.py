import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json 
import numpy as np
import scipy.stats as stats

def create_data_split(data_dir, splits):
    """
    partition the data from a single directory with all images + labels

    data_dir: name of directory with data
    splits: fraction each new directory should have of data
    """
    assert sum(splits) == 1, 'partitions must sum to 1'
    assert len(splits) > 1, 'need to split into more than 1'

    img_fnames = [fname for fname in os.listdir(data_dir) if fname.endswith('.JPG')]
    imgs_left = img_fnames
    split_data = []

    for split in splits[:-1]:

        # splits out of 1.0, but train test removes some
        # so renormalize
        perc_left = len(imgs_left) / len(img_fnames)
        new_split = split / perc_left

        # split further
        split_fnames, imgs_left = train_test_split(imgs_left, train_size=new_split)
        split_data.append(split_fnames)
        perc = round(len(split_fnames) / len(img_fnames), 2)

    split_data.append(imgs_left)    # out of loop to avoid rounding errors + bc train_test_split accepts 0<x<1

    return split_data

def plot_dt_gt_counts(coco_dt_path, coco_gt_path, conf_thresh, save_path=None, plot=True):
    """
    plot the counts of detected objects vs. num actual objects, per image at the given level of confidence
    Params:
        coco_dt_path: path to COCO detections file
        coco_gt_path: path to COCO ground truth file
        conf_thresh: the minimum confidence at which you will count a detection
        save_path: where to save the plot, do not save if save_path is None
        plot: show the plot if True
    Returns:
        (pearson correlation coefficient, p-value of ataining >= R without any correlation)
    """

    with open(coco_gt_path, 'r') as coco_gt_f:
        coco_gt = json.load(coco_gt_f)
        n_imgs = len(coco_gt['images'])
        ground_truth_counts = np.zeros(n_imgs)
        for ground_truth in coco_gt['annotations']:
            ground_truth_counts[ground_truth['image_id']] += 1
    
    with open(coco_dt_path, 'r') as coco_dt_f:
        detected_counts = np.zeros(n_imgs)
        detections = [d for d in json.load(coco_dt_f) if d['score'] > conf_thresh]
        for detection in detections:
            detected_counts[detection['image_id']] += 1

    pearson_r, p_value = stats.pearsonr(ground_truth_counts, detected_counts)
    
    # add slight noise so duplicate points not right on top of eachother
    n = len(ground_truth_counts)
    jittered_gt_counts = ground_truth_counts + np.random.rand(n) / 3
    jittered_dt_counts = detected_counts + np.random.rand(n) / 3

    fig, ax = plt.subplots()
    ax.scatter(jittered_gt_counts, jittered_dt_counts, alpha=0.5)
    _ = ax.set(title=f'detected count v. ground truth count (thresh={conf_thresh}, r={round(pearson_r, 2)})',
            xlabel="ground truth count", ylabel="detected count")
    
    # plot 1 to 1 line
    max_val = int(np.max(ground_truth_counts))
    ax.plot([])    # hack to use secondary color
    ax.plot(range(max_val+1), alpha=0.7, label='perfect accuracy')

    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if plot:
        plt.show()
    else:
        _ = plt.clf()

    return pearson_r, p_value


def confidence_interval(data, confidence=0.95):
    """
    compute the confidence interval for data using a t-distrbution
    Args:
        data: the 1-d data to calculate the confidence interval for
        confidence: the confidence level to construct the interval
    Returns:
        (interval start, interval end)
    """
    mean = np.mean(data)
    # standard error of the mean * z-score of 2-tailed confidence
    interval = stats.sem(data) * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean-interval, mean+interval


if __name__ == '__main__':
    coco_detection_fpath = 'data/coco_examples/coco_instances_results.json'
    coco_ground_truth_fpath = 'data/coco_examples/seed_test_coco_format.json'
    r, p = plot_dt_gt_counts(coco_detection_fpath, coco_ground_truth_fpath, 0.8, save_path=None, plot=True)