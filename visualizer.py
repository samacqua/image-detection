import cv2
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from enum import Enum, auto
import os

from .mAP import summarize_coco
from .bbox import convert_bbox, BBox

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

    def add_category(self, name, color):
        """add a category to the list of categories so that, when showing image, category shows up in legend"""
        self._categories.add((name, color))

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

    def draw_ground_truth(self, annotations, mode='a',
                          predictions=None, bbox_from=BBox.X1Y1X2Y2, **bbox_args):
        """
        draw the ground truth bounding boxes
        Args:
            annotations: 'annotations' of COCO ground truth file
            mode: 'a'=draw all predictions, 'tp'=only true positives, 'fn'=only false negatives, 
                'tpfn'=draw all predictions, but distinguished by true positives v. false negatives
            predictions: list of predictions on self.img,  unnecessary if mode=='a'
            bbox_from: the BBox format of which the bounding boxes are in the form of 
            kwargs: arguments to be passed to _draw_bbox
        Returns:
            None
        """
        modes = {'a', 'fn', 'tp', 'tpfn'}
        if mode not in modes:
            raise ValueError(f'mode must be one of {modes}')
        self._stack.append(lambda thresh: self._draw_ground_truth(annotations, mode,
                          predictions, thresh, bbox_from, **bbox_args))
    def _draw_ground_truth(self, annotations, mode, predictions, thresh, bbox_from, **bbox_args):
        """internal method for drawing ground truth labels to stack. see draw_ground_truth."""

        bboxes = set([tuple(a['bbox']) for a in annotations])

        if mode == 'a':     # just draw all bounding boxes
            color = 'g'
            self._categories.add(('ground truth', color))
            for bbox in bboxes:
                self._draw_bbox(bbox, color=color, label=None, bbox_from=bbox_from, bbox_args=bbox_args, text_args=None)
        
        else:   # mode == 'tp', 'fn', or 'tpfn'

            conf_preds = [p for p in predictions if p['score'] > thresh]    # preds with high enough confidence
            
            # get false negative bounding boxes
            fn_bboxes = bboxes.copy()
            for dt in conf_preds:
                if dt['match'] is not None:
                    fn_bboxes.remove(dt['match']['bbox'])
            
            # true positive bounding boxes
            tp_bboxes = bboxes - fn_bboxes

            if 'tp' in mode:    # mode == 'tp' or 'tpfn'
                color = 'g'
                self._categories.add(('true positive ground truth', color))
                for bbox in tp_bboxes:
                    self._draw_bbox(bbox, color=color, label=None, bbox_from=bbox_from, bbox_args=bbox_args, text_args=None)
            if 'fn' in mode:    # mode == 'fn' or 'tpfn'
                color = 'orange'
                self._categories.add(('false negative ground truth', color))
                for bbox in fn_bboxes:
                    self._draw_bbox(bbox, color=color, label=None, bbox_from=bbox_from, bbox_args=bbox_args, text_args=None)
            
    def draw_predictions(self, predictions, mode='tpfp', bbox_from=BBox.X1Y1X2Y2, bbox_args=None, text_args=None):
        """
        draw a list of COCO detections
        Args:
            predictions: list of predictions on self.img in the COCO format
            mode: 'a'=draw all predictions, 'tp'=draw only true positive predictions,
                'fp'=draw only false positives, 'tpfp'=draw all, but distinguish between 
                true positives and false positives
            bbox_from: the BBox format that each prediction's bbox is in
            bbox_args: dictionary of named arguments to be passed to _draw_box
            text_args: dictionary of named arguments to be passed to _draw_text
        """
        modes = {'a', 'fp', 'tp', 'tpfp'}
        if mode not in modes:
            raise ValueError(f'mode must be one of {modes}')
        self._stack.append(lambda thresh: self._draw_predictions(predictions, mode, thresh, bbox_from, bbox_args, text_args))
    def _draw_predictions(self, predictions, mode, thresh, bbox_from, bbox_args, text_args):
        """internal method for drawing predictions. see draw_predictions."""

        # draw most confident last
        conf_preds = [p for p in predictions if p['score'] > thresh]
        sorted_preds = sorted(conf_preds, key=lambda p: p['score'], reverse=False)

        preds = set()
        if mode == 'a':
            self._categories.add(('predictions', 'b'))
            preds = set((tuple(p['bbox']), 'b', p['score']) for p in sorted_preds)
        else:
            if 'tp' in mode:    # mode == 'tp' or 'tpfp'
                self._categories.add(('true positive predictions', 'b'))
                preds |= set((tuple(p['bbox']), 'b', p['score']) for p in sorted_preds if p['match'] is not None)
            if 'fp' in mode:    # mode == 'fp' or 'tpfp'
                self._categories.add(('false positives', 'firebrick'))
                preds |= set((tuple(p['bbox']), 'firebrick', p['score']) for p in sorted_preds if p['match'] is None)

        for bbox, color, score in preds:
            self._draw_bbox(bbox, label=round(score, 2), color=color, bbox_from=bbox_from,
                            bbox_args=bbox_args, text_args=text_args)
        
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

    def show_image(self, im_id, ground_truth_mode='tpfn', prediction_mode='tpfp', conf_thresh=0.5, save_path=None):
        """
        show the image with the given id
        Args:
            ground_truth_mode: 'a'=draw all ground truth, 'tp'=draw only true positive ground truth, 
                'fn'=draw only false negative ground truth, 'tpfn'=draw all ground truth, but 
                differentiate true positive from false negatives, 'n'=don't draw ground truth
            prediction_mode: 'a'=draw all predictions, 'tp'=draw only true positive predictions, 
                'fp'=draw only false positives, 'tpfp'=draw all predictions, but 
                differentiate true positive from false positives, 'n'=don't draw predictions
            conf_thresh: minimum confidence threshold to show detections / use detection to
                calculate if ground truth label is detected
            save_path: where to save the image visualization, will not save if None
        Returns:
            ground truth object at im_id
            all predictions on im_id, regardless of confidence level
        """

        gt_modes = {'a', 'tp', 'fn', 'tpfn', 'n'}
        if ground_truth_mode not in gt_modes:
            raise ValueError(f'ground_truth_mode must have value from: {gt_modes}')

        dt_modes = {'a', 'tp', 'fp', 'tpfp', 'n'}
        if prediction_mode not in dt_modes:
            raise ValueError(f'prediction_mode must have value from: {dt_modes}')

        gt_obj = self.ground_truth[im_id]
        predictions = self.predictions_by_imid[im_id]
        im_path = gt_obj['file_name']
        
        visualizer = Visualizer(cv2.imread(im_path), color_transform=cv2.COLOR_BGR2RGB)
        if ground_truth_mode != 'n':
            visualizer.draw_ground_truth(gt_obj['annotations'], mode=ground_truth_mode, predictions=predictions)

        if prediction_mode != 'n':
            visualizer.draw_predictions(predictions, mode=prediction_mode)

        visualizer.show(conf_thresh)
        if save_path is not None:
            visualizer.save(save_path, conf_thresh)

        return gt_obj, predictions

    def show_highest_conf_misses(self, n=1, im_id=None, ground_truth_mode='a', prediction_mode='n', save_dir=None):
        """
        show the highest confidence detections that were false positives
        Args:
            n: the number of high confidence false positives to show
            im_id: image id to show miss for. if None, will show highest confidence miss on all images
            ground_truth_mode: 'a'=draw all ground truth, 'tp'=draw only true positive ground truth, 
                'fn'=draw only false negative ground truth, 'tpfn'=draw all ground truth, but 
                differentiate true positive from false negatives, 'n'=don't draw ground truth
            prediction_mode: 'a'=draw all predictions, 'tp'=draw only true positive predictions, 
                'fp'=draw only false positives, 'tpfp'=draw all predictions, but 
                differentiate true positive from false positives, 'n'=don't draw predictions
            save_dir: directory to save the visualizations in, None not to save
        Returns:
            list of n tuples ordered by descending confidence: (prediction, ground truth object)
        """

        gt_modes = {'a', 'tp', 'fn', 'tpfn', 'n'}
        if ground_truth_mode not in gt_modes:
            raise ValueError(f'ground_truth_mode must have value from: {gt_modes}')

        dt_modes = {'a', 'tp', 'fp', 'tpfp', 'n'}
        if prediction_mode not in dt_modes:
            raise ValueError(f'prediction_mode must have value from: {dt_modes}')

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # restrict predictions to image if im_id is not None
        predictions = self.predictions if im_id is None else [p for p in self.predictions if p['image_id'] == im_id]

        # show the n highest confidence false positives
        already_shown = set()
        shown_data = []
        for conf_i in range(n):
            i, first_wrong_pred = next(((i, d) for i, d in enumerate(predictions) if d['match'] is None and i not in already_shown), (-1, None))
            if first_wrong_pred is None:
                print("NO MORE FALSE POSITIVES")
                return shown_data
            already_shown.add(i)

            # load the image and false positive bbox
            im_id = first_wrong_pred['image_id']
            first_wrong_im_obj = self.ground_truth[im_id]
            im_path = first_wrong_im_obj['file_name']
            im = cv2.imread(im_path)
            bbox = list(first_wrong_pred['bbox'])

            # draw ground truth, predictions
            visualizer = Visualizer(im, color_transform=cv2.COLOR_BGR2RGB)
            im_predictions = [p for p in predictions if p['image_id'] == im_id]
            if ground_truth_mode != 'n':
                visualizer.draw_ground_truth(first_wrong_im_obj['annotations'], mode=ground_truth_mode, predictions=im_predictions)
            if prediction_mode != 'n':
                visualizer.draw_predictions(im_predictions, mode=prediction_mode)

            # draw false positive + show
            visualizer.draw_bbox(bbox, label=round(first_wrong_pred['score'], 2), color='orangered', bbox_from=BBox.X1Y1X2Y2)
            visualizer.add_category('highest confidence false positive', 'orangered')
            visualizer.show()

            shown_data.append((first_wrong_pred, first_wrong_im_obj))

            if save_dir is not None:
                visualizer.save(os.path.join(save_dir, f'high_conf_miss_{conf_i}.png'))

        return shown_data

    def show_unfound(self, conf_thresh=0, ground_truth_mode='fn', prediction_mode='n', save_dir=None):
        """
        show all bounding boxes that were not detected at the given level of confidence (false negatives)
        Args:
            conf_thresh: the threshold at which to a predictions must be to be counted
            ground_truth_mode: 'fn'=draw only false negative ground truth, 'tpfn'=draw all ground truth, but 
                differentiate true positive from false negatives
            prediction_mode: 'a'=draw all predictions, 'tp'=draw only true positive predictions, 
                'fp'=draw only false positives, 'tpfp'=draw all predictions, but 
                differentiate true positive from false positives, 'n'=don't draw predictions
            save_dir: directory to save the visualizations in, None not to save
        Returns:
            list of sets of unfound bounding boxes, where index into list is the image id
        """

        gt_modes = {'fn', 'tpfn'}
        if ground_truth_mode not in gt_modes:
            raise ValueError(f'ground_truth_mode must have value from: {gt_modes}')

        dt_modes = {'a', 'tp', 'fp', 'tpfp', 'n'}
        if prediction_mode not in dt_modes:
            raise ValueError(f'prediction_mode must have value from: {dt_modes}')

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        unfound_bboxes = []
        for im_id, gt_obj in enumerate(self.ground_truth):

            # remove all detected bboxes (at given threshold) to get a set of undetected bboxes
            im_unfound_bboxes = set([tuple(ann['bbox']) for ann in gt_obj['annotations']])
            for dt in self.predictions_by_imid[im_id]:
                gt_match = dt['match']
                if gt_match is not None and dt['score'] > conf_thresh:
                    im_unfound_bboxes.remove(tuple(gt_match['bbox']))

            unfound_bboxes.append(im_unfound_bboxes)

            # show the undetected bboxes
            if len(im_unfound_bboxes) > 0:
                im_path = gt_obj['file_name']
                im = cv2.imread(im_path)
                visualizer = Visualizer(im, color_transform=cv2.COLOR_BGR2RGB)
                visualizer.draw_ground_truth(gt_obj['annotations'], mode=ground_truth_mode, predictions=self.predictions_by_imid[im_id])
                if prediction_mode != 'n':
                    visualizer.draw_predictions(self.predictions_by_imid[im_id], mode=prediction_mode)
                visualizer.show(conf_thresh)

                if save_dir is not None:
                    im_name = im_path.split('/')[-1][:-4]
                    visualizer.save(os.path.join(save_dir, f'missed_seeds_{im_name}_conf={conf_thresh}.png'))

        return unfound_bboxes

if __name__ == '__main__':

    # get ground truth and predictions
    coco_detection_fpath = 'data/coco_examples/coco_instances_results.json'
    coco_ground_truth_fpath = 'data/coco_examples/seed_test_coco_format.json'
    res = summarize_coco(coco_detection_fpath=coco_detection_fpath,
                        coco_ground_truth_fpath=coco_ground_truth_fpath,
                         plot_dir=None)
    average_precision, predictions, ground_truth, recall_lists, precision_lists = [x['seed'] if 'seed' in x else x for x in res]

    # get random image
    # data_loc = 'data/dataset/'
    # random_id = np.random.choice(range(len(ground_truth)))
    # random_id = 8
    # print(random_id)
    # random_img = ground_truth[random_id]
    # random_img['file_name'] = os.path.join(data_loc, random_img['file_name'].split('/')[-1])
    # random_img_dts = [p for p in predictions if p['image_id'] == random_id]

    # print('gt:', [gt['bbox'] for gt in random_img['annotations']])
    # print('dt:', [[round(c, 2) for c in dt['bbox']] for dt in random_img_dts])

    # img = cv2.imread(random_img['file_name'])

    # vis = Visualizer(img, color_transform=cv2.COLOR_BGR2RGB)
    # vis.draw_ground_truth(random_img['annotations'], mode='tpfn', 
    #                     predictions=random_img_dts, bbox_from=BBox.X1Y1X2Y2)
    # vis.draw_predictions(random_img_dts, mode='tpfp', bbox_from=BBox.X1Y1X2Y2, bbox_args={'linestyle': ":"}, text_args={'fontfamily': 'cursive'})
    # vis.draw_bbox([100, 100, 110, 110], label='bbox', color='pink', bbox_from=BBox.X1Y1X2Y2)
    
    # vis.show(0)

    # os.makedirs('output', exist_ok=True)
    # for i in [0.1, 0.5, 0.99]:
    #     vis.show(i)
        
    #     vis.save(f'output/im{i}.png', conf_thresh=i)

    coco_vis = COCOVisualizer(coco_ground_truth_fpath, coco_detection_fpath, min_IoU=0.25)

    # for i in range(5):
    #     coco_vis.show_image(i, ground_truth_mode='fn', prediction_mode='tp', conf_thresh=0.5)

    # coco_vis.show_highest_conf_misses(1, im_id=1, ground_truth_mode='tpfn', prediction_mode='fp', save_dir=None)
    for i in [0]:
        coco_vis.show_unfound(i, ground_truth_mode='tpfn', prediction_mode='tpfp', save_dir='output')
