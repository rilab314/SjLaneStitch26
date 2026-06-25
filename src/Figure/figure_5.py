"""Figure 5 — 정제 단계: center_line 평행 겹침 트리밍 (1×2 가로 콜라주). ★핵심 신규★

패널: (a) 정제 전 center_line(이중선 겹침+지그재그) | (b) 트리밍 후(겹친 본체 절단, 분기 가지 보존).
center_line 트리밍이 뚜렷한(제거 ≥ 50px) 프레임만 출력하며, 파일명에 제거 길이를 기록한다.
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
    """center_line 평행 겹침 트리밍 전/후를 1×2로 비교한다."""

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
        return self.compose(stage), f"_drop{int(trim['len_drop'])}"

    def compose(self, stage):
        """정제 전/후 center_line 패널을 흰색 간격으로 가로 결합한다."""
        height, width = stage["img_shape"]
        before = fr.draw_strands(
            fr.make_white_canvas(height, width), stage["combined"], only_class=self.cls)
        after = fr.draw_strands(
            fr.make_white_canvas(height, width), stage["refined"], only_class=self.cls)
        return fr.concat_horizontal([before, after])


if __name__ == "__main__":
    RefinementFigure().run()
