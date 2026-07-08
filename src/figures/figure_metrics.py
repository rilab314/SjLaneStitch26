"""Quantitative metrics for figure selection conditions.

Computes trimming/merging/branching/frame AP from LaneStitcher stage results (stage_linestrings)
and GT/prediction annotations, used to select only the frames where each figure's intended scene
actually appears.
"""
import os
import sys
import contextlib

import cv2
import numpy as np
from pycocotools import mask as mask_util

import config as cfg

CENTER_LINE_ID = 1        # target class for trimming/parallel rejection
MIN_TRIM_DROP_PX = 15.0   # minimum removed length to judge trimming as "occurred"
AP20_IOU = 0.20
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


def measure_frame_ap20(gt_anns, pred_anns, image_id, eval_class_ids=None, exclude_ids=None):
    """COCO AP@IoU0.20 (segm) for one frame. None if there is nothing to evaluate."""
    eval_class_ids = cfg.EVAL_CLASS_IDS if eval_class_ids is None else eval_class_ids
    exclude_ids = cfg.EXCLUDE_IDS if exclude_ids is None else exclude_ids
    gt_list = [a for a in gt_anns if a.get("category_id") not in exclude_ids]
    dt_list = [_with_score(a, image_id) for a in pred_anns
               if a.get("category_id") not in exclude_ids]
    if not gt_list or not dt_list:
        return None
    gt_dataset = build_frame_dataset(gt_list, image_id, eval_class_ids)
    return run_segm_ap(gt_dataset, dt_list, image_id, eval_class_ids)


def _with_score(ann, image_id):
    """Augments a copy of the prediction annotation with score (default 1.0) and image_id."""
    out = dict(ann)
    out.setdefault("score", 1.0)
    out["image_id"] = image_id
    return out


def build_frame_dataset(gt_list, image_id, eval_class_ids):
    """Builds a single-image COCO GT dictionary (including the fields loadRes requires)."""
    height, width = _detect_size(gt_list)
    annotations = []
    for index, ann in enumerate(gt_list):
        annotations.append({
            "id": index + 1, "image_id": image_id, "category_id": ann["category_id"],
            "segmentation": ann["segmentation"], "iscrowd": 0,
            "area": ann.get("area", _ann_area(ann["segmentation"])),
        })
    return {
        "info": {}, "licenses": [],
        "images": [{"id": image_id, "height": height, "width": width}],
        "categories": [{"id": int(c), "name": str(c)} for c in eval_class_ids],
        "annotations": annotations,
    }


def _detect_size(anns, default=(768, 768)):
    """Extracts (height, width) from the RLE segmentation's size (default if absent)."""
    for ann in anns:
        seg = ann.get("segmentation")
        if isinstance(seg, dict) and "size" in seg:
            return int(seg["size"][0]), int(seg["size"][1])
    return default


def _ann_area(seg):
    """Area of an RLE segmentation (0 for polygons, etc.)."""
    if isinstance(seg, dict):
        return float(mask_util.area(seg))
    return 0.0


def run_segm_ap(gt_dataset, dt_list, image_id, eval_class_ids):
    """Computes the mean precision with COCOeval (segm, IoU0.20)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with _suppress_stdout():
        coco_gt = COCO()
        coco_gt.dataset = gt_dataset
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt_list)
        evaluation = COCOeval(coco_gt, coco_dt, iouType="segm")
        evaluation.params.imgIds = [image_id]
        evaluation.params.catIds = list(eval_class_ids)
        evaluation.params.iouThrs = np.array([AP20_IOU], dtype=np.float32)
        evaluation.evaluate()
        evaluation.accumulate()
    precision = evaluation.eval["precision"][0, :, :, 0, -1]
    precision = precision[precision > -1]
    return float(np.mean(precision)) if precision.size else 0.0


@contextlib.contextmanager
def _suppress_stdout():
    """Temporarily suppresses console output from COCO creation/evaluation."""
    with open(os.devnull, "w") as devnull:
        saved = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = saved
