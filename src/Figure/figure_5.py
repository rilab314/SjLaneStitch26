"""Figure 5 — 정제 단계: center_line 평행 겹침 트리밍 전후 비교 (1×3 가로 콜라주). ★핵심 신규★

패널: (a) 원본 영상 + 분할 오버레이(장면 맥락) | (b) 정제 전 center_line(겹친 중복 본체)
      | (c) 트리밍 후(중복 본체 절단, 분기 가지 보존).
center_line 트리밍 제거량 ≥ 20px 인 프레임만 출력한다.
"""
import os
import sys

_CUR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_CUR, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as cfg
import figure_render as fr
import figure_metrics as fm
from figure_base import FigureGenerator


class RefinementFigure(FigureGenerator):
    """장면 맥락(원본+분할) | 정제 전 | 정제 후 center_line을 1×3으로 비교한다."""

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
        """장면 맥락 + 정제 전/후 center_line 패널을 검은 여백으로 가로 결합한다."""
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
