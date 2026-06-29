"""Figure 2 — 파이프라인 개요 (1×5 가로 콜라주).

패널: (a) 원본+GT | (b) 분할(평가 클래스) | (c) 초기 linestring(정제 전) | (d) 정제 후 | (e) 최종 병합.
center_line 평행 겹침 트리밍이 뚜렷한(제거 ≥ 50px) 프레임만 출력한다.
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


class PipelineFigure(FigureGenerator):
    """한 장면이 분할→벡터화→정제→병합 단계를 거치는 흐름을 1×5로 보인다."""

    name = "Figure_2"
    min_trim_drop = 50.0   # center_line 트리밍이 강한(제거 ≥ 50px) 프레임만
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
        """프레임 AP20이 기준 이상인 깔끔한 예시인지 판정."""
        pred_anns = self._detector.convert_to_json(self.final_merge(stage), image_id)
        ap20 = fm.measure_frame_ap20(self.gt_annotations(image_id), pred_anns, image_id)
        return ap20 is not None and ap20 > self.ap20_min

    def compose(self, stage, image_id):
        """다섯 패널을 흰색 간격으로 가로 결합한다."""
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
        """원본 영상에 GT 어노테이션을 오버레이한 패널."""
        return fr.draw_annotations_on_image(
            stage["image"].copy(), self.gt_annotations(image_id), cfg.EXCLUDE_IDS)


if __name__ == "__main__":
    PipelineFigure().run()
