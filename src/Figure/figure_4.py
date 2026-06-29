"""Figure 4 — 벡터화: 곡률 인지 추적 + 잔여 재추출 (1×3 가로 콜라주).

데모 클래스 = center_line(한 색). 패널:
(a) 원본 영상 + 분할 블롭 | (b) 정렬 샘플점(외삽 없음) | (c) 잔여 재추출 결과(1차=회색, 복원=주황).
잔여 재추출로 center_line이 복원된 프레임만 출력한다(패널 (c)가 의미를 갖도록).
"""
import os
import sys

import numpy as np

_CUR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_CUR, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as cfg
import figure_render as fr
import figure_metrics as fm
from figure_base import FigureGenerator


class VectorizationFigure(FigureGenerator):
    """center_line의 블롭→샘플점→잔여복원 벡터화 과정을 1×3으로 보인다."""

    name = "Figure_4"
    cls = fm.CENTER_LINE_ID
    render = cfg.RENDER_ID2BGR.get(fm.CENTER_LINE_ID, (255, 77, 77))
    gray = (170, 170, 170)
    orange = (0, 140, 255)
    min_residual_len = 50.0   # 잔여로 복원된 center_line 총 길이 ≥ 이 값일 때만

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
        """세 패널을 검은 여백으로 가로 결합한다."""
        height, width = self._detector._img_shape
        panels = [
            self.blob_panel(image, pred_img, height, width),
            self.sample_panel(first, height, width),
            self.residual_panel(first, residual, height, width),
        ]
        return fr.concat_horizontal(panels)

    def blob_panel(self, image, pred_img, height, width):
        """(a) 원본 영상 위에 분할 블롭을 렌더색으로 칠한다."""
        canvas = image.copy()
        canvas[np.all(pred_img == cfg.ID2BGR.get(self.cls), axis=-1)] = self.render
        return canvas

    def sample_panel(self, first, height, width):
        """(b) 정렬 샘플점 폴리라인 + 끝점 점(외삽은 표시하지 않음)."""
        canvas = fr.make_white_canvas(height, width)
        for strand in first:
            fr.draw_strand(canvas, strand.points, self.render, thickness=3, draw_dots=True)
        return canvas

    def residual_panel(self, first, residual, height, width):
        """(c) 1차 선(회색) 위에 잔여로 복원된 선(주황)을 오버레이(끝점 점 포함)."""
        canvas = fr.make_white_canvas(height, width)
        for strand in first:
            fr.draw_strand(canvas, strand.points, self.gray, thickness=3, draw_dots=True)
        for strand in residual:
            fr.draw_strand(canvas, strand.points, self.orange, thickness=3, draw_dots=True)
        return canvas


if __name__ == "__main__":
    VectorizationFigure().run()
