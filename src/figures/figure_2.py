"""Figure 2 — pipeline overview (1x5 horizontal collage).

Panels: (a) original + GT | (b) segmentation (eval classes) | (c) initial linestrings (before
refinement) | (d) after refinement | (e) final merge.
Only outputs frames with pronounced center_line parallel-overlap trimming (drop >= 50px).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
import figure_render as fr
import figure_metrics as fm
from figure_base import FigureGenerator


class PipelineFigure(FigureGenerator):
    """Shows the flow of a scene through the segmentation -> vectorization -> refinement -> merge stages as a 1x5 collage."""

    name = "Figure_2"
    min_trim_drop = 50.0   # only frames with strong center_line trimming (drop >= 50px)
    ap20_min = 0.50

    def build_figure(self, image_id, path):
        pred_img = self.read_prediction(image_id)
        if not fm.has_color(pred_img, cfg.ID2BGR.get(fm.CENTER_LINE_ID)):
            return None
        stage = self._detector.stage_linestrings(
            path, do_merge=True, merge_iters=self._detector.num_merges)
        trim = fm.measure_trim(stage)
        if trim["len_drop"] < self.min_trim_drop:
            return None
        if not self.is_good_frame(stage, image_id):
            return None
        return self.compose(stage, image_id), ""

    def is_good_frame(self, stage, image_id):
        """Decides whether this is a clean example whose frame AP20 is at or above the threshold."""
        pred_anns = self._detector.convert_to_json(self.final_merge(stage), image_id)
        ap20 = fm.measure_frame_ap20(self.gt_annotations(image_id), pred_anns, image_id)
        return ap20 is not None and ap20 > self.ap20_min

    def compose(self, stage, image_id):
        """Combines the five panels horizontally with white gaps."""
        height, width = stage["img_shape"]
        panels = [
            self.gt_panel(stage, image_id),
            fr.recolor_segmentation(stage["pred_img"], cfg.EXCLUDE_IDS),
            fr.draw_strands(fr.make_white_canvas(height, width), stage["combined"]),
            fr.draw_strands(fr.make_white_canvas(height, width), stage["refined"]),
            fr.draw_strands(fr.make_white_canvas(height, width), self.final_merge(stage)),
        ]
        return fr.concat_horizontal(panels)

    def gt_panel(self, stage, image_id):
        """Panel overlaying GT annotations on the original image."""
        return fr.draw_annotations_on_image(
            stage["image"].copy(), self.gt_annotations(image_id), cfg.EXCLUDE_IDS)


if __name__ == "__main__":
    PipelineFigure().run()
