"""Figure 8 — 클래스별 실패 사례 (per-class, 좌[GT] | 중[원본+분할] | 우[TP·FP·FN] 1×3, 원본 해상도).

대상 클래스의 GT·예측을 IoU 0.2로 매칭해 실패 시그니처를 보인다. 4종 케이스:
  edge_line(인접 융합+near-miss) / no_parking(과소검출) / bus_only(오검출) / bicycle(희소·느슨).
좌: 원본 + GT linestring(클래스 렌더색, 끝점 점). 중: 원본 + 분할 맵 오버레이(장면 맥락).
우: TP(녹)·FP(빨강)=예측선, FN(파랑)=놓친 GT선.
한 프레임에서 4종을 모두 검사해 매칭되는 케이스마다 클래스 서브폴더에 저장한다.
sparse 클래스(bus_only·bicycle)는 기준을 완화했다.
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
from evaluator import ann_to_mask

IOU_THR = 0.2


def cond_edge(s):        # 인접 융합 + near-miss
    return s["merge"] >= 1 and s["near"] >= 1


def cond_no_parking(s):  # 장척 과소검출(미검출)
    return s["miss"] >= 1 or (s["recall"] <= 0.5 and s["near"] >= 1)


def cond_bus_only(s):    # 오검출(FP) — 대폭 완화: bus_only가 GT/예측에 등장하는 프레임
    return s["nG"] >= 1 or s["nP"] >= 1


def cond_bicycle(s):     # 희소·느슨 — 대폭 완화: bicycle이 GT/예측에 등장하는 프레임
    return s["nG"] >= 1 or s["nP"] >= 1


class FailureCaseFigure(FigureGenerator):
    """대상 클래스의 IoU 매칭 실패 양상을 좌[GT] | 우[TP·FP·FN] 2패널로 보인다."""

    name = "Figure_8"
    TP, FP, FN = (0, 255, 0), (0, 0, 255), (255, 0, 0)  # BGR: 녹/빨강/파랑
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
            match = self._match_class(final, gt, case["class_id"], h, w)
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

    def _match_class(self, final, gt, class_id, h, w):
        """한 프레임·한 클래스의 예측↔GT IoU 매칭과 진단 시그니처를 구한다."""
        gt_masks = [ann_to_mask(a, h, w) for a in gt if a.get("category_id") == class_id]
        pred_strands = [s for s in final if s.class_id == class_id]
        if not gt_masks and not pred_strands:
            return None
        pred_masks = [self._strand_mask(s, h, w) for s in pred_strands]
        iou = self._iou_matrix(pred_masks, gt_masks)
        matched_pred, matched_gt = self._greedy_match(iou)
        return {"gt_masks": gt_masks, "pred_strands": pred_strands,
                "matched_pred": matched_pred, "matched_gt": matched_gt,
                "signals": self._signals(iou, len(pred_masks), len(gt_masks), len(matched_pred))}

    def _strand_mask(self, strand, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.rint(np.asarray(strand.points)).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(mask, [pts], False, 1, self._detector.thickness)
        elif len(pts) == 1:
            cv2.circle(mask, (int(pts[0, 0, 0]), int(pts[0, 0, 1])), self._detector.thickness, 1, -1)
        return mask

    def _iou_matrix(self, pred_masks, gt_masks):
        iou = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)
        for i, pm in enumerate(pred_masks):
            pa = pm > 0
            for j, gm in enumerate(gt_masks):
                ga = gm > 0
                inter = int(np.logical_and(pa, ga).sum())
                if inter:
                    iou[i, j] = inter / int(np.logical_or(pa, ga).sum())
        return iou

    def _greedy_match(self, iou):
        nP, nG = iou.shape
        pairs = sorted(((iou[i, j], i, j) for i in range(nP) for j in range(nG)
                        if iou[i, j] >= IOU_THR), reverse=True)
        matched_pred, matched_gt = set(), set()
        for _, i, j in pairs:
            if i in matched_pred or j in matched_gt:
                continue
            matched_pred.add(i)
            matched_gt.add(j)
        return matched_pred, matched_gt

    def _signals(self, iou, nP, nG, M):
        gt_best = iou.max(axis=0) if nP else np.zeros(nG)
        pred_best = iou.max(axis=1) if nG else np.zeros(nP)
        merge = int(np.sum((iou > 0).sum(axis=1) >= 2)) if nP and nG else 0
        return {"nG": nG, "nP": nP, "M": M,
                "miss": int(np.sum(gt_best == 0.0)),
                "near": int(np.sum((gt_best > 0.0) & (gt_best < IOU_THR))),
                "fp": int(np.sum(pred_best == 0.0)),
                "merge": merge,
                "precision": M / nP if nP else 0.0,
                "recall": M / nG if nG else 0.0}

    def _build_panel(self, image, pred_img, match, class_id):
        """좌(GT 렌더색) | 중(원본+분할 오버레이) | 우(FN 파랑 GT + TP 녹/FP 빨강 예측)를 원본 해상도로 결합."""
        gt_strand_lists = [self._mask_strands(m, class_id) for m in match["gt_masks"]]
        left = image.copy()
        render = cfg.RENDER_ID2BGR.get(class_id, (255, 255, 255))
        for strands in gt_strand_lists:
            self._draw(left, strands, render)
        middle = fr.overlay_segmentation(image, pred_img, cfg.EXCLUDE_IDS)
        right = image.copy()
        for j, strands in enumerate(gt_strand_lists):
            if j not in match["matched_gt"]:
                self._draw(right, strands, self.FN)
        for i, strand in enumerate(match["pred_strands"]):
            fr.draw_strand(right, strand.points,
                           self.TP if i in match["matched_pred"] else self.FP,
                           thickness=3, draw_dots=True)
        return fr.concat_horizontal([left, middle, right])

    def _mask_strands(self, mask, class_id):
        """GT 인스턴스 마스크를 세선화·샘플링해 끝점 점을 찍을 수 있는 polyline으로 변환."""
        self._detector._id_count = self._detector.id_offset
        line_map, raw = self._detector._thin_image(mask.astype(np.uint8), class_id)
        return self._detector._extend_lines(line_map, raw)

    def _draw(self, image, strands, color):
        for strand in strands:
            fr.draw_strand(image, strand.points, color, thickness=3, draw_dots=True)


if __name__ == "__main__":
    FailureCaseFigure().run()
