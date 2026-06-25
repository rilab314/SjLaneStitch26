"""Figure 1 — 최종 차선 추출 결과(예측) 쇼케이스.

원본 위성영상 위에 best 파라미터로 추출·병합한 최종 linestring을 클래스색 + 끝점 점으로 오버레이.
프레임 단위 AP@IoU0.20 ≥ 0.70 인 잘 나온 프레임만 출력한다.
"""
import os
import sys

_CUR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_CUR, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import figure_render as fr
import figure_metrics as fm
from figure_base import FigureGenerator


class ResultShowcaseFigure(FigureGenerator):
    """프레임 AP20가 높은(잘 추출된) 결과만 골라 예측 오버레이로 저장."""

    name = "Figure_1"
    ap20_min = 0.70

    def build_figure(self, image_id, path):
        stage = self._detector.stage_linestrings(
            path, do_merge=True, merge_iters=self._detector.num_merges)
        final = self.final_merge(stage)
        ap20 = self.frame_ap20(image_id, final)
        if ap20 is None or ap20 < self.ap20_min:
            return None
        canvas = stage["image"].copy()
        fr.draw_strands(canvas, final, dots=True)
        return canvas, f"_ap{ap20:.2f}"

    def frame_ap20(self, image_id, final):
        """최종 linestring을 RLE 예측으로 변환해 프레임 AP20을 계산."""
        pred_anns = self._detector.convert_to_json(final, image_id)
        return fm.measure_frame_ap20(self.gt_annotations(image_id), pred_anns, image_id)


if __name__ == "__main__":
    ResultShowcaseFigure().run()
