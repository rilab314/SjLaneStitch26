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

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg


IOUs=[0.10, 0.20, 0.50]


def main():
    """Smoke test for the evaluator -- this is NOT the real pipeline entry point.

    The real evaluation runs inside experiment/run_experiments.py, which calls
    evaluate_all() for each parameter folder. This test only re-evaluates the best
    config's existing validation predictions (coco_pred_val_*.json under the best
    parameter folder) to confirm evaluate_all / evaluate_coco_ap / evaluate_miou_json
    still work end to end.
    """
    from util import find_best_pred_json_path, find_model_path
    split = 'validation'
    csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
    model_name, _, pred_json_path = find_best_pred_json_path(csv_path)
    if model_name is None:
        print(f"[evaluator test] no predictions found under {cfg.RESULT_PATH}; nothing to evaluate")
        return
    model_path = find_model_path(model_name)
    # find_best_pred_json_path points at the best param folder's prediction json;
    # evaluate_all globs coco_pred_{split}_*.json in that same folder.
    result_path = os.path.dirname(pred_json_path)
    evaluate_all(cfg.coco_anno_path(split), cfg.label_dir(split), model_path, result_path, split)


def evaluate_all(coco_gt_json, label_path, model_path, result_path, split='validation'):
    # val/test coexist in the same param folder. Read only the prediction JSON of the current split,
    # append a (val)/(test) suffix to the metric columns, and merge into eval_result.csv.
    label = cfg.split_label(split)
    json_files = sorted(glob.glob(os.path.join(result_path, f"coco_pred_{label}_*.json")))
    results = []
    result = {"merge_count": None}
    result.update(_suffix(evaluate_segm_pred_miou(model_path, label_path, split), label))
    results.append(result)

    for json_file in json_files:
        merge_count = _filename_to_merge_count(json_file)
        result = {"merge_count": merge_count}
        result.update(_suffix(evaluate_coco_ap(coco_gt_json, json_file), label))
        result.update(_suffix(evaluate_miou_json(json_file, label_path), label))
        results.append(result)

    df = pd.DataFrame(results)
    filename = os.path.join(result_path, "eval_result.csv")
    df = _merge_split_eval(filename, df)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"\n{'Evaluation Results [' + split + ']':^100}\n")
    print(df.to_string(index=False))


def _suffix(metrics: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Append a (label) suffix to each key of the metrics dict. e.g. AP20 -> AP20(val)."""
    return {f"{k}({label})": v for k, v in metrics.items()}


def _order_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    """After merge_count, order the metric columns by split (val->test) and then by metric."""
    metrics = ["instances", "AP10", "AP20", "AP50", "mIoU"]
    ordered = ["merge_count"]
    for lbl in ("val", "test"):
        for m in metrics:
            col = f"{m}({lbl})"
            if col in df.columns and col not in ordered:
                ordered.append(col)
    ordered += [c for c in df.columns if c not in ordered]
    return df[ordered]


def _merge_split_eval(csv_path: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge with an existing eval_result.csv (columns of the other split) on the merge_count key to keep val and test columns in one file."""
    if not os.path.exists(csv_path):
        return _order_eval_columns(new_df)
    old = pd.read_csv(csv_path)
    sentinel = -999  # temporary value for merging the NaN merge_count (pure seg row)

    def norm(df):
        df = df.copy()
        df["merge_count"] = pd.to_numeric(df["merge_count"], errors="coerce").fillna(sentinel)
        return df

    old_n, new_n = norm(old), norm(new_df)
    dup = [c for c in new_n.columns if c in old_n.columns and c != "merge_count"]
    old_n = old_n.drop(columns=dup)  # prefer new values when re-evaluating the same split
    merged = pd.merge(old_n, new_n, on="merge_count", how="outer").sort_values("merge_count")
    merged["merge_count"] = merged["merge_count"].replace(sentinel, pd.NA)
    return _order_eval_columns(merged)


def _filename_to_merge_count(json_file: str):
    # The last token of coco_pred_{val|test}_{origin|merge1|merge2}.json is the stage.
    stage = os.path.basename(json_file).replace(".json", "").split("_")[-1]
    if stage == "origin":
        return 0
    if stage.startswith("merge"):
        return int(stage[5:])
    return stage  # fallback: return unknown formats as is


def evaluate_segm_pred_miou(model_path: str, label_path: str, split: str = 'validation') -> Dict[str, float]:
    print(f"===== [evaluate_segm_pred_miou] model_path: {model_path}, label_path: {label_path}, split: {split}")
    metrics = evaluate_segm_pred_metrics(model_path, label_path, split)
    res = {"mIoU": metrics["mIoU"]}
    print(f"===== [evaluate_segm_pred_miou] res: {res}")
    return res


def evaluate_segm_pred_metrics(model_path: str, label_path: str, split: str = 'validation') -> Dict[str, Any]:
    """Compute the mIoU and per-class IoU of the pure segmentation prediction (uses a per-split metrics cache).

    Prediction masks are read from <model_path>/pred_val|pred_test, and GT labels from label_path (per split).
    The cache is stored in a per-split metrics_{split}.json."""
    metrics_json = os.path.join(model_path, f"metrics_{split}.json")
    if os.path.exists(metrics_json):
        data = load_json(metrics_json)
        if "per_class_iou" in data:  # recompute old-version caches that lack per-class IoU
            return data

    pred_dir = cfg.pred_path(model_path, split)
    files = glob.glob(os.path.join(label_path, "*.png"))
    intersections = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
    unions = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
    for file in tqdm(files, desc=f"Segm IoU[{split}]"):
        grtr_label = to_label_index_image(cv2.imread(file, cv2.IMREAD_UNCHANGED), True)
        if grtr_label is None:
            continue
        pred_img = cv2.imread(os.path.join(pred_dir, os.path.basename(file)))
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
    # A separate cache per GT file (..._selected.json). Since val/test GT are in the same folder (RESULT_PATH),
    # using a fixed name would cause cache collisions between splits.
    root, ext = os.path.splitext(gt_json)
    save_path = f"{root}_selected{ext}"
    # Reuse the cache only when it is newer than the source (gt_json). Invalidate the cache when the source is updated.
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
    print(f"selected_annotation.json saved: {save_path}")
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
