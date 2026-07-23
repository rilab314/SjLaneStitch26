"""Figure 1 — final lane extraction result (prediction) showcase.

Overlays the final linestrings (extracted and merged with the best parameters) on the
original satellite image using class colors + endpoint dots.
Only outputs well-performing frames whose per-frame F1@IoU0.5 >= 0.50.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import figure_render as fr
import figure_metrics as fm
from figure_base import FigureGenerator


class ResultShowcaseFigure(FigureGenerator):
    """Selects only high frame-F1 (well-extracted) results and saves them as prediction overlays."""

    name = "Figure_1"
    f1_min = 0.50

    def build_figure(self, image_id, path):
        stage = self._detector.stage_linestrings(
            path, do_merge=True, merge_iters=self._detector.num_merges)
        final = self.final_merge(stage)
        f1 = self.frame_f1(image_id, final)
        if f1 is None or f1 < self.f1_min:
            return None
        canvas = stage["image"].copy()
        fr.draw_strands(canvas, final, dots=True)
        return canvas, ""

    def frame_f1(self, image_id, final):
        """Converts the final linestrings to RLE predictions and computes the frame F1@0.5."""
        pred_anns = self._detector.convert_to_json(final, image_id)
        return fm.measure_frame_f1(self.gt_annotations(image_id), pred_anns)


if __name__ == "__main__":
    ResultShowcaseFigure().run()
