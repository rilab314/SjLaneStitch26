import os
import glob
import json
import sys
from typing import Dict, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


IOUs=[0.10, 0.20, 0.50]


def main():
    coco_gt_json = os.path.join(cfg.RESULT_PATH, "merged_annotations.json")
    label_path = os.path.join(cfg.DATASET_PATH, "annotations", "validation")
    
    from util import find_best_pred_json_path, find_model_path
    csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
    model_name, _, _ = find_best_pred_json_path(csv_path)

    if model_name is None:
        model_name = "internimage_large"

    model_path = find_model_path(model_name)

    evaluate_all(coco_gt_json, label_path, model_path, cfg.RESULT_PATH)


def evaluate_all(coco_gt_json, label_path, model_path, result_path):
    json_files = sorted(glob.glob(os.path.join(result_path, "coco_pred_*.json")))
    results = []
    result = {"merge_count": None}
    result.update(evaluate_segm_pred_miou(model_path, label_path))
    results.append(result)

    for json_file in json_files:
        merge_count = _filename_to_merge_count(json_file)
        result = {"merge_count": merge_count}
        result.update(evaluate_coco_ap(coco_gt_json, json_file))
        result.update(evaluate_miou_json(json_file, label_path))
        results.append(result)

    df = pd.DataFrame(results)
    filename = os.path.join(result_path, "eval_result.csv")
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"\n{'Evaluation Results':^100}\n")
    print(df.to_string(index=False))


def _filename_to_merge_count(json_file: str) -> int:
    name = os.path.basename(json_file).replace(".json", "").replace("coco_pred_instances_", "")
    if name == "origin":
        return 0
    if name.startswith("merge"):
        return int(name[5:])
    return name  # fallback: 알 수 없는 형식은 이름 그대로 반환


def evaluate_segm_pred_miou(model_path: str, label_path: str) -> Dict[str, float]:
    print(f"===== [evaluate_segm_pred_miou] model_path: {model_path}, label_path: {label_path}")
    metrics = evaluate_segm_pred_metrics(model_path, label_path)
    res = {"mIoU": metrics["mIoU"]}
    print(f"===== [evaluate_segm_pred_miou] res: {res}")
    return res


def evaluate_segm_pred_metrics(model_path: str, label_path: str) -> Dict[str, Any]:
    """순수 segmentation 예측의 mIoU와 클래스별 IoU를 계산한다 (metrics.json 캐시 사용)"""
    metrics_json = os.path.join(model_path, "metrics.json")
    if os.path.exists(metrics_json):
        data = load_json(metrics_json)
        if "per_class_iou" in data:  # 클래스별 IoU가 없는 구버전 캐시는 재계산
            return data

    files = glob.glob(os.path.join(label_path, "*.png"))
    intersections = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
    unions = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
    for file in tqdm(files, desc="Segm IoU"):
        grtr_label = to_label_index_image(cv2.imread(file, cv2.IMREAD_UNCHANGED), True)
        if grtr_label is None:
            continue
        pred_img = cv2.imread(os.path.join(model_path, "prediction", os.path.basename(file)))
        pred_label = to_label_index_image(pred_img, False)
        for cid in cfg.EVAL_CLASS_IDS:
            intersections[cid] += int(np.sum((grtr_label == cid) & (pred_label == cid)))
            unions[cid] += int(np.sum((grtr_label == cid) | (pred_label == cid)))

    per_class_iou = {str(cid): (intersections[cid] / unions[cid] if unions[cid] > 0 else 0.0)
                     for cid in cfg.EVAL_CLASS_IDS}
    valid_ious = [intersections[cid] / unions[cid] for cid in cfg.EVAL_CLASS_IDS if unions[cid] > 0]
    res = {"mIoU": float(np.mean(valid_ious)) if valid_ious else 0.0, "per_class_iou": per_class_iou}
    with open(metrics_json, 'w') as f:
        json.dump(res, f)
    return res


def evaluate_coco_ap(gt_json: str, pred_json: str):
    print(f"===== [evaluate_coco_ap] pred_json: {pred_json}")
    selected_gt_path = _get_selected_annotation(gt_json)
    coco_gt = COCO(selected_gt_path)
    dt_data = load_json(pred_json)
    if isinstance(dt_data, list):
        dt_data = [d for d in dt_data if d.get('category_id') not in cfg.EXCLUDE_IDS]
    coco_dt = coco_gt.loadRes(dt_data)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.params.catIds = cfg.EVAL_CLASS_IDS
    coco_eval.params.iouThrs = np.array(IOUs, dtype=np.float32)
    coco_eval.evaluate()
    coco_eval.accumulate()

    res = {"instances": str(len(coco_dt.getAnnIds()))}
    for i, iou in enumerate(IOUs):
        p = coco_eval.eval['precision'][i, :, :, 0, -1]
        p = p[p > -1]
        res[f"AP{int(iou*100)}"] = float(np.mean(p)) if p.size else 0.0
    print(f"===== [evaluate_coco_ap] res: {res}")
    return res


def _get_selected_annotation(gt_json: str) -> str:
    save_path = os.path.join(os.path.dirname(gt_json), "selected_annotation.json")
    # 캐시가 원본(gt_json)보다 최신일 때만 재사용. 원본이 갱신되면 캐시를 무효화한다.
    if os.path.exists(save_path) and os.path.getmtime(save_path) >= os.path.getmtime(gt_json):
        return save_path

    gt_data = load_json(gt_json)
    gt_data['annotations'] = [
        a for a in gt_data.get('annotations', [])
        if a.get('category_id') not in cfg.EXCLUDE_IDS
    ]
    for i, ann in enumerate(gt_data['annotations']):
        ann['id'] = ann.get('id', i + 1)
        ann['iscrowd'] = ann.get('iscrowd', 0)
        if 'area' not in ann and 'segmentation' in ann:
            ann['area'] = float(
                maskUtils.area(ann['segmentation'])
                if isinstance(ann['segmentation'], dict) else 0.0
            )

    with open(save_path, 'w') as f:
        json.dump(gt_data, f)
    print(f"selected_annotation.json 저장: {save_path}")
    return save_path


def evaluate_miou_json(json_path: str, label_path: str) -> Dict[str, float]:
    print(f"===== [evaluate_miou_json] json_path: {json_path}, label_path: {label_path}")
    data = load_json(json_path)
    anns = data["annotations"] if isinstance(data, dict) else data
    ann_idx = {}
    for a in anns:
        if int(a.get("category_id", 0)) in cfg.EXCLUDE_IDS:
            continue
        ann_idx.setdefault(str(a.get("image_id")), []).append(a)

    files = glob.glob(os.path.join(label_path, "*.png"))
    intersections = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
    unions = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
    for file in tqdm(files, desc="JSON mIoU"):
        grtr_label = to_label_index_image(cv2.imread(file, cv2.IMREAD_UNCHANGED), True)
        if grtr_label is None:
            continue
        H, W = grtr_label.shape
        pred_label = json_to_label_image(ann_idx, H, W, file)
        for cid in cfg.EVAL_CLASS_IDS:
            intersections[cid] += int(np.sum((grtr_label == cid) & (pred_label == cid)))
            unions[cid] += int(np.sum((grtr_label == cid) | (pred_label == cid)))

    ious = [intersections[cid] / unions[cid] for cid in cfg.EVAL_CLASS_IDS if unions[cid] > 0]
    val = float(np.mean(ious)) if ious else 0.0
    res = {"mIoU": val}
    print(f"===== [evaluate_miou_json] res: {res}")
    return res


def json_to_label_image(anns: Dict[str, Any], H: int, W: int, file: str) -> np.ndarray:
    pr_lab = np.zeros((H, W), dtype=np.int32)
    cur_anns = anns.get(os.path.basename(file).replace(".png", ""))
    if cur_anns:
        for ann in cur_anns:
            pr_lab[ann_to_mask(ann, H, W) > 0] = int(ann.get("category_id", 0))
    return pr_lab


def ann_to_mask(ann: Dict[str, Any], H: int, W: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if not seg:
        return np.zeros((H, W), dtype=np.uint8)
    if isinstance(seg, dict) and "counts" in seg:
        m = maskUtils.decode(seg)
        if m.ndim == 3:
            m = m[..., 0]
        return (cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                if m.shape != (H, W) else m.astype(np.uint8))
    if isinstance(seg, list):
        if not seg:
            return np.zeros((H, W), dtype=np.uint8)
        rles = maskUtils.frPyObjects(seg, H, W)
        m = maskUtils.decode(rles)
        return np.any(m, axis=2).astype(np.uint8) if m.ndim == 3 else (m > 0).astype(np.uint8)
    return np.zeros((H, W), dtype=np.uint8)


def to_label_index_image(img: np.ndarray, is_gt: bool) -> np.ndarray:
    if img is None:
        return None
    h, w = img.shape[:2]
    lab = np.zeros((h, w), dtype=np.int32)
    if img.ndim == 2:
        lab = img.astype(np.int32)
    else:
        for cid, bgr in cfg.ID2BGR.items():
            mask = np.all(img == np.array(bgr, dtype=np.uint8), axis=-1)
            lab[mask] = int(cid)
    if is_gt:
        lab = lab - 1
        lab[lab < 0] = 0
    return lab


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
