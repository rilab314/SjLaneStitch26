"""Figure 3 — 제외 클래스(guiding_line·safety_zone)의 GT vs 분할 비교.

클래스마다 [원본+GT | 원본+예측] 가로쌍을 만들고, 두 클래스가 모두 충분하면 세로로 합친다.
GT 인스턴스가 3개 이상인 클래스만 대상으로 한다.
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
from figure_base import FigureGenerator


class ExcludedClassFigure(FigureGenerator):
    """평가에서 제외한 클래스의 GT와 분할 결과를 나란히 비교한다."""

    name = "Figure_3"
    target_ids = (8, 10)   # guiding_line, safety_zone
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
        """한 클래스의 [GT | 예측] 가로쌍(인스턴스가 부족하면 None)."""
        anns = [a for a in gt if a.get("category_id") == class_id]
        if len(anns) < self.min_instances:
            return None
        gt_panel = fr.draw_annotations_on_image(base.copy(), anns, [])
        return fr.concat_horizontal([gt_panel, self.pred_panel(base, seg, class_id)])

    def pred_panel(self, base, seg, class_id):
        """분할 예측에서 해당 클래스 픽셀을 원본 위에 렌더색으로 오버레이."""
        out = base.copy()
        color = cfg.ID2BGR.get(class_id)
        if seg is None or color is None:
            return out
        render = cfg.RENDER_ID2BGR.get(class_id, color)
        out[np.all(seg == color, axis=-1)] = render
        return out


if __name__ == "__main__":
    ExcludedClassFigure().run()
