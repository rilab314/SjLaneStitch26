"""Quantitative metrics for figure selection conditions.

Computes trimming/merging/branching/frame F1 from LaneStitcher stage results (stage_linestrings)
and GT/prediction annotations, used to select only the frames where each figure's intended scene
actually appears.
"""
import cv2
import numpy as np

import config as cfg
from evaluator import greedy_match, iou_matrix

CENTER_LINE_ID = 1        # target class for trimming/parallel rejection
MIN_TRIM_DROP_PX = 15.0   # minimum removed length to judge trimming as "occurred"
F1_IOU = cfg.F1_IOUS[0]   # frame-F1 matching threshold (same operating point as the tables)
BRANCH_NEIGHBORS = 3      # branch point: at least this many 8-neighbor skeleton pixels


def arc_length(points):
    """Cumulative Euclidean length of a polyline."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def measure_trim(stage, trim_class_id=CENTER_LINE_ID):
    """center_line parallel-overlap trimming strength. Returns dict(n_in, n_out, len_drop, happened)."""
    before = [s for s in stage["combined"] if s.class_id == trim_class_id]
    after = [s for s in stage["refined"] if s.class_id == trim_class_id]
    len_before = sum(arc_length(s.points) for s in before)
    len_after = sum(arc_length(s.points) for s in after)
    drop = len_before - len_after
    happened = bool(before) and (len(after) != len(before) or drop > MIN_TRIM_DROP_PX)
    return {"n_in": len(before), "n_out": len(after), "len_drop": drop, "happened": happened}


def measure_merge(stage):
    """Number of fragments joined by merging. Returns dict(n_refined, n_merged, joined)."""
    n_refined = len(stage["refined"])
    n_merged = len(stage["merges"][-1]) if stage["merges"] else n_refined
    return {"n_refined": n_refined, "n_merged": n_merged, "joined": n_refined - n_merged}


def has_color(image, color):
    """Whether the image has any pixel of a specific BGR color (for a cheap pre-filter)."""
    if image is None or color is None:
        return False
    return bool(np.any(np.all(image == color, axis=-1)))


def find_branch_points(skeleton):
    """List of branch-point (y, x) coordinates in a 1-pixel skeleton (8-neighbors >= BRANCH_NEIGHBORS)."""
    binary = (np.asarray(skeleton) > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.nonzero((binary > 0) & (neighbors >= BRANCH_NEIGHBORS))
    return list(zip(ys.tolist(), xs.tolist()))


def measure_frame_f1(gt_anns, pred_anns, eval_class_ids=None, exclude_ids=None):
    """Frame-level macro F1 at IoU F1_IOU. None if there is nothing to evaluate.

    Same matching rule as the evaluator (evaluate_f1_per_class) restricted to one frame:
    per class, greedy 1:1 RLE-mask matching at IoU >= F1_IOU, F1 = 2PR/(P+R); macro-averaged
    over the classes present in the frame's GT or prediction. Returns None when either the
    GT or the prediction list is empty (callers skip such frames)."""
    eval_class_ids = cfg.EVAL_CLASS_IDS if eval_class_ids is None else eval_class_ids
    exclude_ids = cfg.EXCLUDE_IDS if exclude_ids is None else exclude_ids
    gt_list = [a for a in gt_anns if a.get("category_id") not in exclude_ids]
    dt_list = [a for a in pred_anns if a.get("category_id") not in exclude_ids]
    if not gt_list or not dt_list:
        return None
    f1s = [_class_f1([a for a in dt_list if a.get("category_id") == cid],
                     [a for a in gt_list if a.get("category_id") == cid])
           for cid in eval_class_ids]
    f1s = [v for v in f1s if v is not None]
    return float(np.mean(f1s)) if f1s else None


def _class_f1(prs, gts):
    """F1 of one frame and one class (None if the class is absent from both GT and prediction)."""
    if not gts and not prs:
        return None
    matched = len(greedy_match(iou_matrix(prs, gts), F1_IOU))
    precision = matched / len(prs) if prs else 0.0
    recall = matched / len(gts) if gts else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
