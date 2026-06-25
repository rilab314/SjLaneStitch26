"""Figure 4 — 벡터화: 세선화 + 곡률 인지 추적 + 잔여 재추출 (1×4 가로 콜라주).

데모 클래스 = center_line(한 색). 패널:
(a) 분할 블롭 | (b) 1픽셀 골격 | (c) 정렬 샘플점(외삽 없음) | (d) 잔여 재추출 결과(1차=회색, 복원=주황).
잔여 재추출로 center_line이 복원된 프레임만 출력한다(패널 (d)가 의미를 갖도록).
"""
import os
import sys

import cv2
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
    """center_line의 블롭→골격→샘플점→잔여복원 벡터화 과정을 1×4로 보인다."""

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
        self._detector._img_shape = pred_img.shape[:2]
        blob, line_map, _ = self._detector.class_skeleton(pred_img, self.cls)
        stage = self._detector.stage_linestrings(path, do_merge=False)
        residual = self._center_lines(stage["res"])
        if sum(fm.arc_length(s.points) for s in residual) < self.min_residual_len:
            return None
        first = self._center_lines(stage["first"])
        return self.compose(pred_img, line_map, first, residual), ""

    def _center_lines(self, strands):
        return [s for s in strands if s.class_id == self.cls]

    def compose(self, pred_img, line_map, first, residual):
        """네 패널을 검은 여백으로 가로 결합한다."""
        height, width = self._detector._img_shape
        panels = [
            self.blob_panel(pred_img, height, width),
            self.skeleton_panel(line_map, height, width),
            self.sample_panel(first, height, width),
            self.residual_panel(first, residual, height, width),
        ]
        return fr.concat_horizontal(panels)

    def blob_panel(self, pred_img, height, width):
        """(a) 분할 블롭을 렌더색으로."""
        canvas = fr.make_white_canvas(height, width)
        canvas[np.all(pred_img == cfg.ID2BGR.get(self.cls), axis=-1)] = self.render
        return canvas

    def skeleton_panel(self, line_map, height, width):
        """(b) 1픽셀 골격(가시성 위해 약간 dilate)."""
        canvas = fr.make_white_canvas(height, width)
        skeleton = cv2.dilate((line_map > 0).astype(np.uint8), np.ones((2, 2), np.uint8))
        canvas[skeleton > 0] = self.render
        return canvas

    def sample_panel(self, first, height, width):
        """(c) 정렬 샘플점 폴리라인 + 끝점 점(외삽은 표시하지 않음)."""
        canvas = fr.make_white_canvas(height, width)
        for strand in first:
            fr.draw_strand(canvas, strand.points, self.render, thickness=3, draw_dots=True)
        return canvas

    def residual_panel(self, first, residual, height, width):
        """(d) 1차 선(회색) 위에 잔여로 복원된 선(주황)을 오버레이(끝점 점 포함)."""
        canvas = fr.make_white_canvas(height, width)
        for strand in first:
            fr.draw_strand(canvas, strand.points, self.gray, thickness=3, draw_dots=True)
        for strand in residual:
            fr.draw_strand(canvas, strand.points, self.orange, thickness=3, draw_dots=True)
        return canvas


if __name__ == "__main__":
    VectorizationFigure().run()
