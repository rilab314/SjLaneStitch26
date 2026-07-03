"""예측 strand ↔ GT 마스크 IoU 매칭(TP/FP/FN)과 매칭 패널 렌더링 공유 모듈.

figure_7(전 클래스 통합 패널)·figure_8(클래스별)이 공유한다.
매칭: 같은 이미지·같은 클래스 안에서 예측 마스크(두께 렌더)와 GT 마스크의 IoU를
greedy로 1:1 매칭(IoU ≥ 0.2). TP=매칭된 예측, FP=미매칭 예측, FN=미매칭 GT.
"""
import os
import sys

import cv2
import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Figure/ → src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as cfg
import figure_render as fr
from evaluator import ann_to_mask

IOU_THR = 0.2
TP, FP, FN = (0, 255, 0), (0, 0, 255), (255, 0, 0)  # BGR: 녹(매칭 예측)/빨강(오검출)/파랑(놓친 GT)


def match_class(detector, final, gt, class_id, h, w):
    """한 프레임·한 클래스의 예측↔GT IoU 매칭과 진단 시그니처(없으면 None)."""
    gt_masks = [ann_to_mask(a, h, w) for a in gt if a.get("category_id") == class_id]
    pred_strands = [s for s in final if s.class_id == class_id]
    if not gt_masks and not pred_strands:
        return None
    pred_masks = [strand_mask(s, h, w, detector.thickness) for s in pred_strands]
    iou = iou_matrix(pred_masks, gt_masks)
    matched_pred, matched_gt = greedy_match(iou)
    return {"gt_masks": gt_masks, "pred_strands": pred_strands,
            "matched_pred": matched_pred, "matched_gt": matched_gt,
            "signals": signals(iou, len(pred_masks), len(gt_masks), len(matched_pred))}


def strand_mask(strand, h, w, thickness):
    """폴리라인을 두께 thickness의 이진 마스크로 래스터화한다."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.rint(np.asarray(strand.points)).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(mask, [pts], False, 1, thickness)
    elif len(pts) == 1:
        cv2.circle(mask, (int(pts[0, 0, 0]), int(pts[0, 0, 1])), thickness, 1, -1)
    return mask


def iou_matrix(pred_masks, gt_masks):
    iou = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)
    for i, pm in enumerate(pred_masks):
        pa = pm > 0
        for j, gm in enumerate(gt_masks):
            ga = gm > 0
            inter = int(np.logical_and(pa, ga).sum())
            if inter:
                iou[i, j] = inter / int(np.logical_or(pa, ga).sum())
    return iou


def greedy_match(iou):
    """IoU ≥ 임계에서 높은 순으로 1:1 greedy 매칭. (matched_pred, matched_gt) 인덱스 집합."""
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


def signals(iou, nP, nG, M):
    """프레임·클래스 진단 지표(near-miss·과병합·미검출·오검출·precision·recall)."""
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


def mask_strands(detector, mask, class_id):
    """GT 인스턴스 마스크를 세선화·샘플링해 끝점 점을 찍을 polyline으로 변환."""
    detector._id_count = detector.id_offset
    line_map, raw = detector._thin_image(mask.astype(np.uint8), class_id)
    return detector._extend_lines(line_map, raw)


def draw(image, strands, color):
    for strand in strands:
        fr.draw_strand(image, strand.points, color, thickness=3, draw_dots=True)


def tpfpfn_panel(detector, image, final, gt, class_ids=None):
    """전 클래스 통합 TP/FP/FN 패널: TP녹·FP빨강=예측선, FN파랑=놓친 GT선."""
    class_ids = cfg.EVAL_CLASS_IDS if class_ids is None else class_ids
    h, w = image.shape[:2]
    panel = image.copy()
    for cid in class_ids:
        match = match_class(detector, final, gt, cid, h, w)
        if match is None:
            continue
        for j, gt_mask in enumerate(match["gt_masks"]):
            if j not in match["matched_gt"]:
                draw(panel, mask_strands(detector, gt_mask, cid), FN)
        for i, strand in enumerate(match["pred_strands"]):
            fr.draw_strand(panel, strand.points,
                           TP if i in match["matched_pred"] else FP, thickness=3, draw_dots=True)
    return panel
