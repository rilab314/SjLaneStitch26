"""Figure 8 — 클래스별 실패 사례 (per-class, 원본|GT|분할|TP·FP·FN 1×4, 원본 해상도).

대상 클래스의 GT·예측을 IoU 0.2로 매칭해 실패 시그니처를 보인다. 4종 케이스:
  edge_line(인접 융합+near-miss) / no_parking(과소검출) / bus_only(오검출) / bicycle(희소·느슨).
(a) 원본 | (b) 원본+GT linestring(클래스 렌더색, 끝점 점) | (c) 원본+분할 맵(불투명)
| (d) TP(녹)·FP(빨강)=예측선, FN(파랑)=놓친 GT선.
한 프레임에서 4종을 모두 검사해 매칭되는 케이스마다 클래스 서브폴더에 저장한다.
sparse 클래스(bus_only·bicycle)는 기준을 완화했다. 매칭·렌더는 figure_match 공유.
"""
import os
import sys

import cv2

_CUR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_CUR, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as cfg
import figure_render as fr
import figure_match as fmatch
from figure_base import FigureGenerator


def cond_edge(s):        # 인접 융합 + near-miss
    return s["merge"] >= 1 and s["near"] >= 1


def cond_no_parking(s):  # 장척 과소검출(미검출)
    return s["miss"] >= 1 or (s["recall"] <= 0.5 and s["near"] >= 1)


def cond_bus_only(s):    # 오검출(FP) — 대폭 완화: bus_only가 GT/예측에 등장하는 프레임
    return s["nG"] >= 1 or s["nP"] >= 1


def cond_bicycle(s):     # 희소·느슨 — 대폭 완화: bicycle이 GT/예측에 등장하는 프레임
    return s["nG"] >= 1 or s["nP"] >= 1


class FailureCaseFigure(FigureGenerator):
    """대상 클래스의 IoU 매칭 실패 양상을 원본|GT|분할|TP·FP·FN 4패널로 보인다."""

    name = "Figure_8"
    MAX_PER_CLASS = 100   # 클래스별 저장 상한(초과 시 저장 중단)
    CASES = [
        {"class_id": 5, "name": "edge_line", "condition": cond_edge},
        {"class_id": 7, "name": "no_parking_stopping_line", "condition": cond_no_parking},
        {"class_id": 4, "name": "bus_only_lane", "condition": cond_bus_only},
        {"class_id": 11, "name": "bicycle_lane", "condition": cond_bicycle},
    ]

    def __init__(self):
        super().__init__()
        self._saved = {case["name"]: 0 for case in self.CASES}

    def save_if_match(self, path):
        if all(n >= self.MAX_PER_CLASS for n in self._saved.values()):
            return 0
        image_id = os.path.basename(path)[:-4]
        stage = self._detector.stage_linestrings(
            path, do_merge=True, merge_iters=self._detector.num_merges)
        final = self.final_merge(stage)
        image, pred_img, (h, w) = stage["image"], stage["pred_img"], stage["img_shape"]
        gt = self.gt_annotations(image_id)
        saved = 0
        for case in self.CASES:
            if self._saved[case["name"]] >= self.MAX_PER_CLASS:
                continue
            match = fmatch.match_class(self._detector, final, gt, case["class_id"], h, w)
            if match is None or not case["condition"](match["signals"]):
                continue
            self._save_case(image, pred_img, match, case, image_id)
            self._saved[case["name"]] += 1
            saved += 1
        return saved

    def _save_case(self, image, pred_img, match, case, image_id):
        out_dir = os.path.join(self._out_dir, case["name"])
        os.makedirs(out_dir, exist_ok=True)
        panel = self._build_panel(image, pred_img, match, case["class_id"])
        cv2.imwrite(os.path.join(out_dir, f"{image_id}.png"), panel)

    def _build_panel(self, image, pred_img, match, class_id):
        """원본 | GT(렌더색) | 분할(불투명) | TP녹·FP빨강·FN파랑을 원본 해상도로 결합."""
        gt_strand_lists = [fmatch.mask_strands(self._detector, m, class_id) for m in match["gt_masks"]]
        gt_panel = image.copy()
        render = cfg.RENDER_ID2BGR.get(class_id, (255, 255, 255))
        for strands in gt_strand_lists:
            fmatch.draw(gt_panel, strands, render)
        seg = fr.overlay_segmentation(image, pred_img, cfg.EXCLUDE_IDS, alpha=1.0)
        match_panel = image.copy()
        for j, strands in enumerate(gt_strand_lists):
            if j not in match["matched_gt"]:
                fmatch.draw(match_panel, strands, fmatch.FN)
        for i, strand in enumerate(match["pred_strands"]):
            fr.draw_strand(match_panel, strand.points,
                           fmatch.TP if i in match["matched_pred"] else fmatch.FP,
                           thickness=3, draw_dots=True)
        return fr.concat_horizontal([image.copy(), gt_panel, seg, match_panel])


if __name__ == "__main__":
    FailureCaseFigure().run()
