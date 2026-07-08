"""Figure 3 — GT vs segmentation comparison for excluded classes (guiding_line, safety_zone, bicycle_lane).

For each class it builds an [original + GT | original + prediction] horizontal pair, and stacks
them vertically when several classes have enough instances.
Only classes with at least 3 GT instances are considered.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
import figure_render as fr
from figure_base import FigureGenerator


class ExcludedClassFigure(FigureGenerator):
    """Compares the GT and segmentation results of classes excluded from evaluation side by side."""

    name = "Figure_3"
    target_ids = (8, 10, 11)   # guiding_line, safety_zone, bicycle_lane
    min_instances = 3

    def build_figure(self, image_id, path):
        base = cv2.imread(path)
        if base is None:
            return None
        seg = self.read_prediction(image_id)
        gt = self.gt_annotations(image_id)
        rows = [self.class_pair(base, seg, gt, class_id) for class_id in self.target_ids]
        rows = [row for row in rows if row is not None]
        if not rows:
            return None
        return fr.concat_vertical(rows), ""

    def class_pair(self, base, seg, gt, class_id):
        """A single class's [GT | prediction] horizontal pair (None if there are too few instances)."""
        anns = [a for a in gt if a.get("category_id") == class_id]
        if len(anns) < self.min_instances:
            return None
        gt_panel = fr.draw_annotations_on_image(base.copy(), anns, [])
        return fr.concat_horizontal([gt_panel, self.pred_panel(base, seg, class_id)])

    def pred_panel(self, base, seg, class_id):
        """Overlays the segmentation prediction's pixels for the class on the original in render color."""
        out = base.copy()
        color = cfg.ID2BGR.get(class_id)
        if seg is None or color is None:
            return out
        render = cfg.RENDER_ID2BGR.get(class_id, color)
        out[np.all(seg == color, axis=-1)] = render
        return out


if __name__ == "__main__":
    ExcludedClassFigure().run()
