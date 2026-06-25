"""Figure 선별 조건을 위한 정량 지표.

LaneStitcher 단계별 결과(stage_linestrings)와 GT/예측 어노테이션에서 트리밍·병합·분기·프레임 AP를
계산해, 각 figure가 의도한 장면이 실제로 나타난 프레임만 고르는 데 쓴다.
"""
import os
import sys
import contextlib

import cv2
import numpy as np
from pycocotools import mask as mask_util

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Figure/ → src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as cfg

CENTER_LINE_ID = 1        # 트리밍·평행거부 대상 클래스
MIN_TRIM_DROP_PX = 15.0   # 트리밍 '발생' 판정 최소 제거 길이
AP20_IOU = 0.20
BRANCH_NEIGHBORS = 3      # 분기점: 8-이웃 골격 픽셀이 이 수 이상


def arc_length(points):
    """폴리라인의 누적 유클리드 길이."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def measure_trim(stage, trim_class_id=CENTER_LINE_ID):
    """center_line 평행 겹침 트리밍 강도. 반환 dict(n_in, n_out, len_drop, happened)."""
    before = [s for s in stage["combined"] if s.class_id == trim_class_id]
    after = [s for s in stage["refined"] if s.class_id == trim_class_id]
    len_before = sum(arc_length(s.points) for s in before)
    len_after = sum(arc_length(s.points) for s in after)
    drop = len_before - len_after
    happened = bool(before) and (len(after) != len(before) or drop > MIN_TRIM_DROP_PX)
    return {"n_in": len(before), "n_out": len(after), "len_drop": drop, "happened": happened}


def measure_merge(stage):
    """병합으로 이어진 단편 수. 반환 dict(n_refined, n_merged, joined)."""
    n_refined = len(stage["refined"])
    n_merged = len(stage["merges"][-1]) if stage["merges"] else n_refined
    return {"n_refined": n_refined, "n_merged": n_merged, "joined": n_refined - n_merged}


def has_color(image, color):
    """이미지에 특정 BGR 색 픽셀이 하나라도 있는지(값싼 사전필터용)."""
    if image is None or color is None:
        return False
    return bool(np.any(np.all(image == color, axis=-1)))


def find_branch_points(skeleton):
    """1픽셀 골격에서 분기점(8-이웃이 BRANCH_NEIGHBORS 이상) 좌표 (y, x) 리스트."""
    binary = (np.asarray(skeleton) > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.nonzero((binary > 0) & (neighbors >= BRANCH_NEIGHBORS))
    return list(zip(ys.tolist(), xs.tolist()))


def measure_frame_ap20(gt_anns, pred_anns, image_id, eval_class_ids=None, exclude_ids=None):
    """한 프레임의 COCO AP@IoU0.20(segm). 평가 대상이 없으면 None."""
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
    """예측 어노테이션 사본에 score(기본 1.0)와 image_id를 보강."""
    out = dict(ann)
    out.setdefault("score", 1.0)
    out["image_id"] = image_id
    return out


def build_frame_dataset(gt_list, image_id, eval_class_ids):
    """단일 이미지 COCO GT 딕셔너리를 만든다(loadRes가 요구하는 필드 포함)."""
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
    """RLE segmentation의 size에서 (height, width)를 추출(없으면 기본값)."""
    for ann in anns:
        seg = ann.get("segmentation")
        if isinstance(seg, dict) and "size" in seg:
            return int(seg["size"][0]), int(seg["size"][1])
    return default


def _ann_area(seg):
    """RLE segmentation의 면적(폴리곤 등은 0)."""
    if isinstance(seg, dict):
        return float(mask_util.area(seg))
    return 0.0


def run_segm_ap(gt_dataset, dt_list, image_id, eval_class_ids):
    """COCOeval(segm, IoU0.20)로 precision 평균을 계산한다."""
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
    """COCO 생성/평가의 콘솔 출력을 잠시 억제한다."""
    with open(os.devnull, "w") as devnull:
        saved = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = saved
