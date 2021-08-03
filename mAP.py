import json
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_average_precision(recall, precision):
    """
    Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
            precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.
    Using method in VOC2012
    """

    recall = recall.copy()
    recall = np.concatenate([[0], recall, [1]]) # insert 0 at beginning and end of list

    precision = precision.copy()
    precision = np.concatenate([[0], precision, [0]])

    # make precision monotonically increasing
    mono_increase_prec = np.maximum.accumulate(precision[::-1])[::-1]

    # find indices where recall changes
    i_list = np.where(recall[:-1] != recall[1:])[0] + 1

    # compute Average Precision (AP) -- the area under the curve
    ap = 0.0
    for i in i_list:
        ap += ((recall[i]-recall[i-1])*mono_increase_prec[i])
    return ap, list(recall), list(mono_increase_prec)


def compute_IoU(bbox_a, bbox_b):
    """
    compute the intersection between 2 bounding boxes
    Args:
        bbox_a: bounding box of form x1, y1, x2, y2
        bbox_b: bounding box of form x1, y1, x2, y2
    """

    x1a, y1a, x2a, y2a = bbox_a
    x1b, y1b, x2b, y2b = bbox_b

    # calculate intersection area
    xA = max(x1a, x1b)
    yA = max(y1a, y1b)
    xB = min(x2a, x2b)
    yB = min(y2a, y2b)
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # calculate union area
    bbox_a_area = (x2a - x1a + 1) * (y2a - y1a + 1)
    bbox_b_area = (x2b - x1b + 1) * (y2b - y1b + 1)
    union_area = bbox_a_area + bbox_b_area - inter_area

    IoU = inter_area / union_area

    return IoU

def summarize_coco(coco_detection_fpath, 
                coco_ground_truth_fpath, 
                plot_dir=None, min_IoU=0.5):
    """
    compute object detection stats

    returns dict for each class containing:
        - AP, prediction objects (with detection results (i.e. fp vs. tp)), recall list, 
            ground truth: [{'file_name': filename, 'annotations': [annotation]}] where index is image_id, precision list
            (prediction objects, recall list, and precision list all in same order (sorted by confidence))
    """
    
    # load detections and ground truth
    coco_detections = json.load(open(coco_detection_fpath))
    coco_ground_truth = json.load(open(coco_ground_truth_fpath))

    # get list of classes
    gt_classes = [cat['name'] for cat in sorted(coco_ground_truth['categories'], key=lambda x: x['id'])]

    # map category id to category name
    category_id_to_name = {}
    for category in coco_ground_truth['categories']:
        category_id_to_name[category['id']] = category['name']

    # format ground truth (gt) annotations for calculation
    sorted_gt_images = sorted(coco_ground_truth['images'], key=lambda im: im['id'])
    ground_truth = [{'file_name': im_gt['file_name'], 'annotations': []} for im_gt in sorted_gt_images]
    for gt in coco_ground_truth['annotations']:
        formatted_gt = gt.copy()
        x, y, w, h = gt['bbox']
        formatted_gt['bbox'] = x, y, x+w, y+h   # convert to x1,y1,x2,y2
        formatted_gt['used'] = False    # so that multiple detections aren't assigned to same ground truth
        formatted_gt['class_name'] = category_id_to_name[gt['category_id']]
        ground_truth[gt['image_id']]['annotations'].append(formatted_gt)

    # for fast lookup of detections (dt) by class
    detection_results = {}
    for dt in coco_detections:
        formatted_dt = dt.copy()
        x, y, w, h = dt['bbox']
        formatted_dt['bbox'] = x, y, x+w, y+h 
        class_name = gt_classes[formatted_dt['category_id']]
        detection_results.setdefault(class_name, []).append(formatted_dt)

    # sort by confidence
    for class_name in detection_results:
        detection_results[class_name].sort(key=lambda x:float(x['score']), reverse=True)

    # Create an "output/" directory
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    # count number ground truth objects per class
    gt_counter_per_class = {}
    for gt_obj in ground_truth:
        for obj in gt_obj['annotations']:
            class_name = obj['class_name']
            gt_counter_per_class[class_name] = gt_counter_per_class.get(class_name, 0) + 1

    # calculate AP per class
    average_precision = {}
    predictions = {class_name: [] for class_name in gt_classes}
    recall_lists = {}
    precision_lists = {}

    for class_index, class_name in enumerate(gt_classes):

        # assign detection-results to ground-truth objects
        
        nd = len(detection_results.get(class_name, []))
        tp = np.zeros(nd)   # true positives
        fp = np.zeros(nd)   # false positives
        for idx, detection in enumerate(detection_results.get(class_name, [])):

            # assign detection-results to ground truth object if any
            img_ground_truth = ground_truth[detection["image_id"]]['annotations']

            best_overlap = best_gt_match = best_match_idx = -1

            # load detected object bounding-box
            bb = detection["bbox"]
            for gt_idx, obj in enumerate(img_ground_truth):
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = obj["bbox"]  # ground truth bounding box
                    IoU = compute_IoU(bb, bbgt)

                    if IoU > best_overlap and IoU > 0:  # only update if best and any overlap
                        best_overlap = IoU
                        best_gt_match = obj
                        best_match_idx = gt_idx
                
            # assign detection as true positive/false positive
            if best_overlap >= min_IoU:
                if not best_gt_match["used"]:
                    # true_positives_per_class.setdefault(class_name, []).append((bb, best_gt_match['bbox']))
                    tp[idx] = 1     # true positive
                    img_ground_truth[best_match_idx]['used'] = True
                else:
                    # false_positives_per_class.setdefault(class_name, []).append(bb)
                    fp[idx] = 1     # false positive (multiple detection)
            else:
                # false_positives_per_class.setdefault(class_name, []).append(bb)
                fp[idx] = 1     # false positive

            # save detection object with detection result
            detection_w_result = detection.copy()
            detection_w_result['match'] = best_gt_match if tp[idx] == 1 else None
            predictions[class_name].append(detection_w_result)

        # compute total tp and fp, precision/recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        recall = tp / gt_counter_per_class[class_name]
        precision = tp / (fp + tp)
        recall_lists[class_name] = list(recall)
        precision_lists[class_name] = list(precision)

        ap, padded_recall, mono_increase_prec = calculate_average_precision(recall, precision)
        average_precision[class_name] = ap

        # plot and save
        if plot_dir is not None:
            plt.plot(recall, precision, '-o')

            area_under_curve_x = padded_recall[:-1] + [padded_recall[-2]] + [padded_recall[-1]]
            area_under_curve_y = mono_increase_prec[:-1] + [0.0] + [mono_increase_prec[-1]]
            plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
            fig = plt.gcf()
            fig.canvas.set_window_title('AP ' + class_name)

            text = f'class: {class_name} AP={round(ap*100, 2)}% (IoU={min_IoU})'
            plt.title(text)
            plt.xlabel('Recall')
            plt.ylabel('Precision')

            axes = plt.gca() # gca - get current axes
            axes.set_xlim([0.0,1.0])
            axes.set_ylim([0.0,1.05]) # .05 to give some extra space

            fig.savefig(os.path.join(plot_dir, f'{class_name}_{min_IoU}.png'))
            plt.cla()

    return average_precision, predictions, ground_truth, recall_lists, precision_lists

def main():
    coco_detection_fpath = 'data/coco_examples/coco_instances_results.json'
    coco_ground_truth_fpath = 'data/coco_examples/seed_test_coco_format.json'
    res = summarize_coco(coco_detection_fpath=coco_detection_fpath,
                        coco_ground_truth_fpath=coco_ground_truth_fpath,
                         plot_dir='output')
    print(res[0], res[1], res[3], res[4])
    average_precision, predictions, ground_truth, recall_lists, precision_lists = [x['seed'] if 'seed' in x else x for x in res]
    print('average precision:', str(round(100*average_precision, 2)) + '%')

if __name__ == '__main__':
    main()