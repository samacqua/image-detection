import cv2
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from enum import Enum, auto

from mAP import summarize_coco

class BBox(Enum):
    """
    enumeration of different bbox types
    """
    XYWH = auto()   # [top left x, top left y, width, height]
    X1Y1X2Y2 = auto()   # [left x, top y, right x, bottom y]
    XMIDYMIDWH_NORM = auto()    # [center x (normalized), center y (noramlized), width (normalized), height (normalized)]

def convert_bbox(bbox, from_format, to_format, image_size=(800, 600)):
    """convert bounding box between any two formats"""

    if from_format == BBox.XYWH:
        x,y,w,h = bbox
    elif from_format == BBox.X1Y1X2Y2:
        x,y,x2,y2 = bbox
        w, h = x2-x, y2-y
    elif from_format == BBox.XMIDYMIDWH_NORM:
        x_mid_n, y_mid_n, w_n, h_n = bbox
        x_mid, w = x_mid_n * image_size[0], w_n * image_size[0]
        y_mid, h = y_mid_n * image_size[1], h_n * image_size[1]
        x = x_mid - w/2
        y = y_mid - h/2
    
    if to_format == BBox.XYWH:
        return [x,y,w,h]
    elif to_format == BBox.X1Y1X2Y2:
        return [x,y,x+w,y+h]
    elif to_format == BBox.XMIDYMIDWH_NORM:
        x_n, w_n = x/image_size[0], w/image_size[0]
        y_n, h_n = y/image_size[1], h/image_size[1]
        x_mid_n = x_n + w_n/2
        y_mid_n = y_n + h_n/2
        return [x_mid_n, y_mid_n, w_n, h_n]

def bbox_center(bbox, from_format, image_size=(800, 600)):
    """calculate the center of the bounding box"""
    x_center, y_center, *_ = convert_bbox(bbox, from_format, BBox.XMIDYMIDWH_NORM, image_size=image_size)
    return x_center, y_center

class VisImage:
    """
    trimmed copy of https://github.com/facebookresearch/detectron2/blob/9eb0027d795bb9a38098bb05e2ceb273bfc9cf41/detectron2/utils/visualizer.py
    easier than installing all of detectron
    """
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.
        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath, bbox_inches = 'tight', pad_inches = 0)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

class Visualizer:
    """
    class to draw bounding boxes on images
    contains symbolic representations of bounding boxes, so can render
    bounding boxes at different confidence scores
    """
    def __init__(self, img, scale=1.0, color_transform=None):
        """
        Args:
            img: the image to draw the bounding boxes on top of
            scale: size which to scale the image
            color_transform: cv2 color transform function (i.e. COLOR_BGR2RGB)
                to originally apply to the image
        """
        
        # set up VisImage
        self.img = img
        if color_transform is not None:
            self.img = cv2.cvtColor(self.img, color_transform)

        self.output = VisImage(self.img, scale=scale)

        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 100, 8 // scale
        )

        self._stack = []    # represents the functions to draw the bounding boxes
        self._categories = set()    # set of (category name, category color) to put on legend

    def get_output(self):
        """
        Returns:
            output (VisImage): the image output containing the visualizations added
            to the image.
        """
        return self.output

    def _draw_text(self, text, x, y, font_size=None, bg_color="g", alpha=1.0, **kwargs):
        """internal function to draw text onto the output VisImage at position x, y"""

        if font_size is None:
            font_size = self._default_font_size

        self.output.ax.text(x, y, text, size=font_size * self.output.scale,
            bbox={"facecolor": bg_color, "alpha": alpha, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="bottom", horizontalalignment="left",
            color='w', zorder=10, **kwargs)

    def _draw_box(self, x, y, w, h, **kwargs):
        """internal function to draw a bounding box [x,y,w,h] onto the output VisImage"""

        linewidth = max(self._default_font_size / 5, 1.5)

        self.output.ax.add_patch(
            mpl.patches.Rectangle((x, y), w, h, fill=False,
                linewidth=linewidth * self.output.scale, **kwargs
            )
        )

    def draw_bbox(self, bbox, label=None, color='g', bbox_from=BBox.XYWH, bbox_args=None, text_args=None):
        """external method for adding function to stack for drawing bounding box + label"""
        self._stack.append(lambda thresh: self._draw_bbox(bbox, label, color, bbox_from, bbox_args, text_args))
    def _draw_bbox(self, bbox, label, color, bbox_from, bbox_args, text_args):
        """
        method for drawing bounding box + label
        Args:
            bbox: the bounding box to draw
            label: the label of the bounding box
            color: the color of the label background and bounding box
            bbox_from: the BBox format that bbox is in (i.e. X1Y1X2Y2)
            bbox_args: dictionary of named arguments to be passed to _draw_box
            text_args: dictionary of named arguments to be passed to _draw_text
        """
        x, y, w, h = convert_bbox(bbox, bbox_from, BBox.XYWH, image_size=(self.output.width, self.output.height)) # standardize bbox format
        bbox_args = bbox_args if bbox_args is not None else {}
        self._draw_box(x, y, w, h, edgecolor=color, **bbox_args)

        text_args = text_args if text_args is not None else {}
        self._draw_text(label, x, y, bg_color=color, **text_args)

    def draw_ground_truth(self, annotations, missed_only=False,
                          predictions=None, bbox_from=BBox.X1Y1X2Y2, **bbox_args):
        """external method for adding function to draw ground truth labels to stack"""
        self._stack.append(lambda thresh: self._draw_ground_truth(annotations, missed_only,
                          predictions, thresh, bbox_from, **bbox_args))
    def _draw_ground_truth(self, annotations, missed_only, predictions, thresh, bbox_from, **bbox_args):
        """
        draw the ground truth bounding boxes
        Args:
            annotations: 'annotations' of COCO ground truth file
            missed_only: flag whether to draw all ground truth bboxes or only ones with no 
                detections, at the given level of confidence
            predictions: list of predictions on self.img, only necessary if missed_only==True
            thresh: confidence threshold for using predictions, only necessary if missed_only==True
            bbox_from: the BBox format of which the bounding boxes are in the form of 
            kwargs: arguments to be passed to _draw_bbox
        """

        bboxes = set([tuple(a['bbox']) for a in annotations])

        if missed_only:
            conf_preds = [p for p in predictions if p['score'] > thresh]    # preds with high enough confidence

            # remove all ground truth bboxes which were detected
            for dt in conf_preds:
                if dt['match'] is not None:
                    bboxes.remove(dt['match']['bbox'])

            color = 'orange'
            self._categories.add(('false negatives', color))
        else:
            color = 'g'
            self._categories.add(('ground truth', color))

        for bbox in bboxes:
            self._draw_bbox(bbox, color=color, label=None, bbox_from=bbox_from, bbox_args=bbox_args, text_args=None)

    def draw_predictions(self, predictions, bbox_from=BBox.X1Y1X2Y2, bbox_args=None, text_args=None):
        self._stack.append(lambda thresh: self._draw_predictions(predictions, thresh, bbox_from, bbox_args, text_args))
    def _draw_predictions(self, predictions, thresh, bbox_from, bbox_args, text_args):
        """
        draw a list of COCO detections
        Args:
            predictions: list of predictions on self.img in the COCO format
            thresh: the confidence threshold that predictions must exceed to be drawn
            bbox_from: the BBox format that each prediction's bbox is in
            bbox_args: dictionary of named arguments to be passed to _draw_box
            text_args: dictionary of named arguments to be passed to _draw_text
        """
        # draw most confident first
        conf_preds = [p for p in predictions if p['score'] > thresh]
        sorted_preds = sorted(conf_preds, key=lambda p: p['score'], reverse=True)

        for p in conf_preds:
            color = 'r' if p['match'] is None else 'b'
            self._draw_bbox(p['bbox'], label=round(p['score'], 2), 
            color=color, bbox_from=bbox_from, bbox_args=bbox_args, text_args=text_args)
        
        self._categories |= {('true positives', 'b'), ('false positives', 'r')}

    def show(self, conf_thresh=0.5, show=True):
        """
        go through the stack of drawing functions and draw each set of bounding boxes at the 
        given confidence level and show the image.
        Args:
            conf_thresh: minimum confidence to draw detections
            show: whether to run plt.show()
        """

        # reset output
        self.output = VisImage(self.img, scale=self.output.scale)

        # actually add all the things
        for f in self._stack:
            f(conf_thresh)

        fig, ax = plt.subplots()
        if len(self._categories) > 0:
            categories, colors = zip(*self._categories)
            custom_lines = [Line2D([0], [0], color=c, lw=4) for c in colors]
            ax.legend(custom_lines, categories)

        fig.set_size_inches(18, 10)
        ax.axis("off")
        ax.imshow(self.get_output().get_image())
        self.output.fig = fig
        self.output.ax = ax
        if show:
            plt.show()

    def save(self, save_path, conf_thresh=0.5):
        """save the output at conf_thresh to save_path"""
        self.show(conf_thresh, show=False)
        self.output.save(save_path)

class COCOVisualizer:
    """class that controls Visualizer objects to make showing entire dataset easier"""
    def __init__(self, coco_ground_truth_fpath, coco_detection_fpath, min_IoU=0.5):
        """
        Args:
            coco_ground_truth_fpath: path to COCO gt file
            coco_detection_fpath: path to COCO dt file
            min_IoU: the minimum IoU that is considered a detection
        """
        res = summarize_coco(coco_detection_fpath=coco_detection_fpath,
                        coco_ground_truth_fpath=coco_ground_truth_fpath,
                         plot_dir=None, min_IoU=min_IoU)
        _, self.predictions, self.ground_truth, *_ = [x['seed'] if 'seed' in x else x for x in res]

        # sort by confidence
        self.predictions.sort(key=lambda x:float(x['score']), reverse=True)

        # pre-calculate for O(1) look up
        self.predictions_by_imid = {im_id: [p for p in self.predictions if p['image_id'] == im_id] for im_id in range(len(self.ground_truth))}

    def show_image(self, im_id, ground_truth=True, false_negatives_only=True, detections=True, conf_thresh=0.5, save_path=None):
        """
        show the image with the given id
        Args:
            ground_truth: True to show ground truth bboxes
            false_negatives_only: if True (and ground_truth is True), then will only show ground 
                truth bboxes that were not matched from any prediction
            detection: True to show detection bboxes
            conf_thresh: minimum confidence threshold to show detections / use detection to
                calculate if ground truth label is detected
            save_path: where to save the image visualization, will not save if None
        """
        gt_obj = self.ground_truth[im_id]
        predictions = self.predictions_by_imid[im_id]
        # im_name = gt_obj['file_name']
        im_name = 'data/dataset/' + gt_obj['file_name'].split('/')[-1]
        
        visualizer = Visualizer(cv2.imread(im_name), color_transform=cv2.COLOR_BGR2RGB)
        if ground_truth:
            visualizer.draw_ground_truth(gt_obj['annotations'], missed_only=false_negatives_only, predictions=predictions)

        if detections:
            visualizer.draw_predictions(predictions)

        show(conf_thresh)
        if save_path is not None:
            visualizer.save(save_path, conf_thresh)

    def show_highest_conf_misses(self, n=1, save_dir=None):
        """
        show the highest confidence detections that were false positives
        Args:
            n: the number of high confidence false positives to show
            save_dir: directory to save the visualizations in, None not to save
        """

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        already_shown = set()
        for conf_i in range(n):
            i, first_wrong_prec = next((i, d) for i, d in enumerate(self.predictions) if d['match'] is None and i not in already_shown)
            already_shown.add(i)

            first_wrong_im_obj = self.ground_truth[first_wrong_prec['image_id']]
            # im_path = first_wrong_im_obj['file_name']
            im_path = 'data/dataset/' + first_wrong_im_obj['file_name'].split('/')[-1]
            im = cv2.imread(im_path)
            bbox = list(first_wrong_prec['bbox'])
            visualizer = Visualizer(im, color_transform=cv2.COLOR_BGR2RGB)
            visualizer.draw_ground_truth(first_wrong_im_obj['annotations'])
            visualizer.draw_bbox(bbox, label=round(first_wrong_prec['score'], 2), color='r', bbox_from=BBox.X1Y1X2Y2)
            visualizer.show()

            if save_dir is not None:
                visualizer.save(os.path.join(save_dir, f'high_conf_miss_{conf_i}.png'))

    def show_unfound(self, conf_thresh=0, draw_predictions=False, save_dir=None):
        """
        show all bounding boxes that were not detected at the given level of confidence (false negatives)
        Args:
            conf_thresh: the threshold at which to a predictions must be to be counted
            save_dir: directory to save the visualizations in, None not to save
        """

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        for im_id, gt_obj in enumerate(self.ground_truth):

            # remove all detected bboxes (at given threshold) to get a set of undetected bboxes
            im_unfound_bboxes = set([ann['id'] for ann in gt_obj['annotations']])
            for dt in self.predictions_by_imid[im_id]:
                gt_match = dt['match']
                if gt_match is not None and dt['score'] > conf_thresh:
                    im_unfound_bboxes.remove(gt_match['id'])

            # show the undetected bboxes
            if len(im_unfound_bboxes) > 0 and im_id==10:
                # im_path = gt_obj['file_name']
                im_path = 'data/dataset/' + gt_obj['file_name'].split('/')[-1]
                im = cv2.imread(im_path)
                visualizer = Visualizer(im, color_transform=cv2.COLOR_BGR2RGB)
                visualizer.draw_ground_truth(gt_obj['annotations'], missed_only=True, predictions=self.predictions_by_imid[im_id])
                if draw_predictions:
                    visualizer.draw_predictions(self.predictions_by_imid[im_id])
                visualizer.show(conf_thresh)

                if save_dir is not None:
                    im_name = im_path.split('/')[-1][:-4]
                    visualizer.save(os.path.join(save_dir, f'missed_seeds_{im_name}_conf={conf_thresh}.png'))
    

if __name__ == '__main__':

    import os
    from mAP import summarize_coco

    # get ground truth and predictions
    coco_detection_fpath = 'data/coco_examples/coco_instances_results.json'
    coco_ground_truth_fpath = 'data/coco_examples/seed_test_coco_format.json'
    res = summarize_coco(coco_detection_fpath=coco_detection_fpath,
                        coco_ground_truth_fpath=coco_ground_truth_fpath,
                         plot_dir=None)
    average_precision, predictions, ground_truth, recall_lists, precision_lists = [x['seed'] if 'seed' in x else x for x in res]

    # get random image
    data_loc = 'data/dataset/'
    random_id = np.random.choice(range(len(ground_truth)))
    print(random_id)
    random_img = ground_truth[random_id]
    random_img['file_name'] = os.path.join(data_loc, random_img['file_name'].split('/')[-1])
    random_img_dts = [p for p in predictions if p['image_id'] == random_id]

    print('gt:', [gt['bbox'] for gt in random_img['annotations']])
    print('dt:', [dt['bbox'] for dt in random_img_dts])

    img = cv2.imread(random_img['file_name'])

    vis = Visualizer(img, color_transform=cv2.COLOR_BGR2RGB)
    vis.draw_ground_truth(random_img['annotations'], missed_only=True, 
                        predictions=random_img_dts, bbox_from=BBox.X1Y1X2Y2)
    vis.draw_predictions(random_img_dts, bbox_from=BBox.X1Y1X2Y2, bbox_args={'linestyle': ":"}, text_args={'fontfamily': 'cursive'})
    vis.draw_bbox([100, 100, 110, 110], label='bbox', color='pink', bbox_from=BBox.X1Y1X2Y2)
    
    os.makedirs('output', exist_ok=True)
    for i in [0.1, 0.5, 0.99]:
        vis.show(i)
        
        vis.save(f'output/im{i}.png', conf_thresh=i)

    coco_vis = COCOVisualizer(coco_ground_truth_fpath, coco_detection_fpath, min_IoU=0.25)

    for i in range(5):
        coco_vis.show_image(i, detections=False, false_negatives_only=False, conf_thresh=0.5)

    coco_vis.show_highest_conf_misses(3, save_dir='output')
    for i in [0, 0.5]:
        coco_vis.show_unfound(i, draw_predictions=True, save_dir='output')
