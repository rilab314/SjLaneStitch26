"""Figure 6 — merging: extrapolation intersection + parallel rejection + serial chaining (1x3 horizontal collage).

Panels: (a) refined fragment bodies (so the broken state is visible) | (b) bodies + outward endpoint
      extrapolation (light gray) | (c) serially connected final result.
Only outputs frames where fragments are actually joined (joined >= 3) and center_line parallel rejection occurs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import figure_render as fr
import figure_metrics as fm
from figure_base import FigureGenerator
from lane_stitcher import bodies_parallel


class MergingFigure(FigureGenerator):
    """Shows fragments before merge | extrapolation | after merge as a 1x3 collage (end-to-end chaining and avoiding parallel mis-merges)."""

    name = "Figure_6"
    cls = fm.CENTER_LINE_ID
    ext_color = (153, 136, 119)   # LightSlateGray (BGR) — light blue-gray, does not clash with any class render color
    min_joined = 3

    def build_figure(self, image_id, path):
        stage = self._detector.stage_linestrings(
            path, do_merge=True, merge_iters=self._detector.num_merges)
        if fm.measure_merge(stage)["joined"] < self.min_joined:
            return None
        self._detector._img_shape = stage["img_shape"]
        if self.count_parallel_rejections(stage["refined"]) < 1:
            return None
        return self.compose(stage), ""

    def count_parallel_rejections(self, refined):
        """Number of center_line pairs that would be rejected as parallel double lines during merging (the key scene of figure 6)."""
        group = self.center_candidates(refined)
        if len(group) < 2:
            return 0
        by_id = {strand.id: strand for strand in group}
        return sum(self._rejections_from(strand, group, by_id) for strand in group)

    def center_candidates(self, refined):
        """List of center_line candidates that have extension endpoints."""
        return [s for s in refined if s.class_id == self.cls
                and s.points is not None and len(s.points) >= 2 and s.ext_points is not None]

    def _rejections_from(self, strand, group, by_id):
        """Number of candidates overlapping strand whose bodies are parallel (counts only the larger-id side to avoid duplicates)."""
        detector = self._detector
        count = 0
        for other_id in detector._find_overlap(group, strand):
            other = by_id.get(int(other_id))
            if other is None or other.id <= strand.id:
                continue
            if bodies_parallel(strand.points, other.points,
                               detector.parallel_overlap, detector.parallel_lateral):
                count += 1
        return count

    def compose(self, stage):
        """Combines the three panels before merge | extrapolation | after merge horizontally with black gaps."""
        height, width = stage["img_shape"]
        refined = stage["refined"]
        final = self._detector._smoothed_copies(self.final_merge(stage))
        panels = [
            fr.draw_strands(fr.make_white_canvas(height, width), refined, dots=True),
            self.extension_panel(refined, height, width),
            fr.draw_strands(fr.make_white_canvas(height, width), final, dots=True),
        ]
        return fr.concat_horizontal(panels)

    def extension_panel(self, refined, height, width):
        """(b) On top of the fragment bodies (class color + endpoint dots), draws only the outward endpoint extrapolation as light-gray lines."""
        canvas = fr.draw_strands(fr.make_white_canvas(height, width), refined, dots=True)
        for strand in refined:
            fr.draw_extension(canvas, strand.ext_points, strand.src_range, self.ext_color)
        return canvas


if __name__ == "__main__":
    MergingFigure().run()
