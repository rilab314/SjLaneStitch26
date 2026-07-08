"""Figure 5 — refinement stage: before/after comparison of center_line parallel-overlap trimming (1x3 horizontal collage). *key new*

Panels: (a) original image + segmentation overlay (scene context) | (b) center_line before refinement
      (overlapping duplicate bodies) | (c) after trimming (duplicate bodies cut, branch spurs preserved).
Only outputs frames where the center_line trimming drop >= 20px.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
import figure_render as fr
import figure_metrics as fm
from figure_base import FigureGenerator


class RefinementFigure(FigureGenerator):
    """Compares scene context (original + segmentation) | before refinement | after refinement center_line as a 1x3 collage."""

    name = "Figure_5"
    cls = fm.CENTER_LINE_ID
    min_trim_drop = 20.0

    def build_figure(self, image_id, path):
        pred_img = self.read_prediction(image_id)
        if not fm.has_color(pred_img, cfg.ID2BGR.get(self.cls)):
            return None
        stage = self._detector.stage_linestrings(path, do_merge=False)
        trim = fm.measure_trim(stage)
        if trim["len_drop"] < self.min_trim_drop:
            return None
        return self.compose(stage), ""

    def compose(self, stage):
        """Combines the scene-context + before/after center_line panels horizontally with black gaps."""
        height, width = stage["img_shape"]
        context = fr.overlay_segmentation(
            stage["image"], stage["pred_img"], cfg.EXCLUDE_IDS, alpha=1.0)
        before = fr.draw_strands(
            fr.make_white_canvas(height, width), stage["combined"], only_class=self.cls)
        after = fr.draw_strands(
            fr.make_white_canvas(height, width), stage["refined"], only_class=self.cls)
        return fr.concat_horizontal([context, before, after])


if __name__ == "__main__":
    RefinementFigure().run()
