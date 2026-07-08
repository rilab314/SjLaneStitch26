"""Shared module for prediction strand <-> GT mask IoU matching (TP/FP/FN) and matching-panel rendering.

Shared by figure_7 (all-class combined panel) and figure_8 (per-class).
Matching: within the same image and same class, greedily 1:1 matches prediction masks (thickness render)
and GT masks by IoU (IoU >= 0.2). TP = matched prediction, FP = unmatched prediction, FN = unmatched GT.
"""
import os
import sys

import cv2
import numpy as np

import config as cfg
import figure_render as fr
from evaluator import ann_to_mask

IOU_THR = 0.2
TP, FP, FN = (0, 255, 0), (0, 0, 255), (255, 0, 0)  # BGR: green (matched prediction)/red (false detection)/blue (missed GT)


def match_class(detector, final, gt, class_id, h, w):
    """Prediction<->GT IoU matching and diagnostic signature for one frame and one class (None if absent)."""
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
    """Rasterizes the polyline into a binary mask of the given thickness."""
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
    """Greedy 1:1 matching in descending IoU order above the threshold. (matched_pred, matched_gt) index sets."""
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
    """Frame/class diagnostic metrics (near-miss, over-merge, missed detection, false detection, precision, recall)."""
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
    """Thins and samples a GT instance mask into a polyline for placing endpoint dots."""
    detector._id_count = detector.id_offset
    line_map, raw = detector._thin_image(mask.astype(np.uint8), class_id)
    return detector._extend_lines(line_map, raw)


def draw(image, strands, color):
    for strand in strands:
        fr.draw_strand(image, strand.points, color, thickness=3, draw_dots=True)


def tpfpfn_panel(detector, image, final, gt, class_ids=None):
    """All-class combined TP/FP/FN panel: TP green/FP red = prediction lines, FN blue = missed GT lines."""
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
