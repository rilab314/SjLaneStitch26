"""Figure 4 — vectorization: curvature-aware tracing + residual re-extraction (1x3 horizontal collage).

Demo class = center_line (single color). Panels:
(a) original image + segmentation blobs | (b) ordered sample points (no extrapolation) | (c) residual
re-extraction result (first pass = gray, recovered = orange).
Only outputs frames where center_line was recovered by residual re-extraction (so panel (c) is meaningful).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
import figure_render as fr
import figure_metrics as fm
from figure_base import FigureGenerator


class VectorizationFigure(FigureGenerator):
    """Shows center_line's blob -> sample points -> residual recovery vectorization process as a 1x3 collage."""

    name = "Figure_4"
    cls = fm.CENTER_LINE_ID
    render = cfg.RENDER_ID2BGR.get(fm.CENTER_LINE_ID, (255, 77, 77))
    gray = (170, 170, 170)
    orange = (0, 140, 255)
    min_residual_len = 50.0   # only when the total length of center_line recovered by residual >= this value

    def build_figure(self, image_id, path):
        pred_img = self.read_prediction(image_id)
        if not fm.has_color(pred_img, cfg.ID2BGR.get(self.cls)):
            return None
        stage = self._detector.stage_linestrings(path, do_merge=False)
        residual = self._center_lines(stage["res"])
        if sum(fm.arc_length(s.points) for s in residual) < self.min_residual_len:
            return None
        first = self._center_lines(stage["first"])
        return self.compose(stage["image"], pred_img, first, residual), ""

    def _center_lines(self, strands):
        return [s for s in strands if s.class_id == self.cls]

    def compose(self, image, pred_img, first, residual):
        """Combines the three panels horizontally with black gaps."""
        height, width = self._detector._img_shape
        panels = [
            self.blob_panel(image, pred_img, height, width),
            self.sample_panel(first, height, width),
            self.residual_panel(first, residual, height, width),
        ]
        return fr.concat_horizontal(panels)

    def blob_panel(self, image, pred_img, height, width):
        """(a) Paints the segmentation blobs in render color on the original image."""
        canvas = image.copy()
        canvas[np.all(pred_img == cfg.ID2BGR.get(self.cls), axis=-1)] = self.render
        return canvas

    def sample_panel(self, first, height, width):
        """(b) Ordered sample-point polyline + endpoint dots (extrapolation not shown)."""
        canvas = fr.make_white_canvas(height, width)
        for strand in first:
            fr.draw_strand(canvas, strand.points, self.render, thickness=3, draw_dots=True)
        return canvas

    def residual_panel(self, first, residual, height, width):
        """(c) Overlays the residual-recovered lines (orange) on the first-pass lines (gray), with endpoint dots."""
        canvas = fr.make_white_canvas(height, width)
        for strand in first:
            fr.draw_strand(canvas, strand.points, self.gray, thickness=3, draw_dots=True)
        for strand in residual:
            fr.draw_strand(canvas, strand.points, self.orange, thickness=3, draw_dots=True)
        return canvas


if __name__ == "__main__":
    VectorizationFigure().run()
