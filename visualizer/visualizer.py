import cv2
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class VisImage:
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
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

class Visualizer:
    def __init__(self, img, scale=1.0, color_transform=None):
        
        # set up VisImage
        self.img = img
        if color_transform is not None:
            self.img = cv2.cvtColor(self.img, color_transform)

        self.output = VisImage(self.img, scale=scale)

        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )

        self.scale = scale
        self._stack = []
        self._categories = set()

    def get_output(self):
        """
        Returns:
            output (VisImage): the image output containing the visualizations added
            to the image.
        """
        return self.output

    def draw_text(self, text, x, y, font_size=None, color="w", bg_color="g",
                alpha=1.0, horizontal_alignment="left", rotation=0):

        if not font_size:
            font_size = self._default_font_size

        self.output.ax.text(x, y, text, size=font_size * self.output.scale, family="sans-serif",
            bbox={"facecolor": bg_color, "alpha": alpha, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="bottom", horizontalalignment=horizontal_alignment,
            color='w', zorder=10, rotation=rotation)

    def draw_box(self, box_coord, text=None, alpha=1.0, color="g", line_style="-"):

        x, y, w, h = box_coord

        linewidth = max(self._default_font_size / 3, 1.5)

        self.output.ax.add_patch(
            mpl.patches.Rectangle((x, y), w, h, fill=False, edgecolor=color,
                linewidth=linewidth * self.output.scale, alpha=alpha,
                linestyle=line_style,
            )
        )

        if text is not None:
            self.draw_text(text, x, y, bg_color=color, alpha=alpha)

    def draw_bounding_boxes(self, bboxes, labels, colors, from_corners=False):
        if from_corners:
            bboxes = [[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in bboxes]
        for bbox, label, color in zip(bboxes, labels, colors):
            x, y, w, h = bbox
            self.draw_box(bbox, color=color, text=label, alpha=1)

    def draw_ground_truth(self, annotations, from_corners=True, missed_only=False,
                          predictions=None, image_id=None):
        self._stack.append(lambda thresh: self._draw_ground_truth(annotations, from_corners, missed_only,
                          predictions, image_id, thresh))

    def _draw_ground_truth(self, annotations, from_corners=True, missed_only=False,
                          predictions=None, image_id=None, thresh=0.5):

        bboxes = set([tuple(a['bbox']) for a in annotations])

        if missed_only:

            if len(predictions) == 0:
                return
            
            sorted_preds = sorted(predictions, key=lambda p: p['score'], reverse=True)

            best_score = sorted_preds[0]['score']
            for i in range(len(sorted_preds)):
                # if conf out of 100, make out of 1
                if best_score > 1:
                    sorted_preds[i]['score'] /= 100

                # filter out too low confidence
                if sorted_preds[i]['score'] < thresh:
                    sorted_preds = sorted_preds[:i]
                    break

            annot_ids = set([a['id'] for a in annotations])
            for dt in sorted_preds:
                if dt['match'] is not None and dt['match']['id'] in annot_ids and dt['score'] > thresh:
                    bboxes.remove(dt['match']['bbox'])

            color = 'orange'
            self._categories.add(('false negatives', color))
        else:
            color = 'g'
            self._categories.add(('ground truth', color))

        labels = [None] * len(bboxes)
        colors = [color] * len(bboxes)

        self.draw_bounding_boxes(list(bboxes), labels, colors, from_corners=from_corners)

    def draw_predictions(self, predictions, from_corners=True):
        self._stack.append(lambda thresh: self._draw_predictions(predictions, from_corners, thresh))
    
    def _draw_predictions(self, predictions, from_corners=True, thresh=0.5):

        if len(predictions) == 0:
            return
        
        # draw most confident first
        sorted_preds = sorted(predictions, key=lambda p: p['score'], reverse=True)

        best_score = sorted_preds[0]['score']
        for i in range(len(sorted_preds)):
            # if conf out of 100, make out of 1
            if best_score > 1:
                sorted_preds[i]['score'] /= 100

            # filter out too low confidence
            if sorted_preds[i]['score'] < thresh:
                sorted_preds = sorted_preds[:i]
                break

        bboxes, scores = [], []
        for p in sorted_preds:
            bboxes.append(p['bbox'])
            scores.append(round(p['score'], 2))

        colors = ['r' if p['match'] is None else 'b' for p in sorted_preds]
        
        self._categories |= set([('true positives', 'b'), ('false positives', 'r')])

        self.draw_bounding_boxes(bboxes, scores, colors, from_corners=from_corners)

    def show(self, conf_thresh=0.5):

        # reset output
        self.output = VisImage(self.img, scale=self.scale)

        # actually add all the things
        for f in self._stack:
            f(conf_thresh)

        fig, ax = plt.subplots()
        categories, colors = zip(*self._categories)
        
        custom_lines = [Line2D([0], [0], color=c, lw=4) for c in colors]
        ax.legend(custom_lines, categories)

        fig.set_size_inches(18, 10)
        ax.axis("off")
        ax.imshow(self.get_output().get_image())
        plt.show()

if __name__ == '__main__':

    import os
    dirname = os.path.dirname(__file__)
    mAP_path = os.path.join(dirname, '../mAP')

    import sys
    sys.path.insert(1, mAP_path)
    from mAP import summarize_coco

    # get ground truth and predictions
    coco_detection_fpath = os.path.join(dirname, '../data/coco_examples/coco_instances_results.json')
    coco_ground_truth_fpath = os.path.join(dirname, '../data/coco_examples/seed_test_coco_format.json')
    res = summarize_coco(coco_detection_fpath=coco_detection_fpath,
                        coco_ground_truth_fpath=coco_ground_truth_fpath,
                         plot_dir=None)
    average_precision, predictions, ground_truth, recall_lists, precision_lists = [x['seed'] if 'seed' in x else x for x in res]

    # get random image
    data_loc = os.path.join(dirname, '../data/dataset/')
    random_id = np.random.choice(range(len(ground_truth)))
    random_img = ground_truth[random_id]
    random_img['file_name'] = os.path.join(data_loc, random_img['file_name'].split('/')[-1])
    random_img_dts = [p for p in predictions if p['image_id'] == random_id]

    print('gt:', [gt['bbox'] for gt in random_img['annotations']])
    print('dt:', [dt['bbox'] for dt in random_img_dts])

    img = cv2.imread(random_img['file_name'])

    vis = Visualizer(img, color_transform=cv2.COLOR_BGR2RGB)
    vis.draw_ground_truth(random_img['annotations'], missed_only=True, 
                        predictions=random_img_dts)
    vis.draw_predictions(random_img_dts)
    vis.show(0.5)