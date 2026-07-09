"""Table 2 — per-class performance of the best model, 9 rows.

Columns: class_name | gt_count | pred_count | AP20 | mIoU
Source: best combo, merge×1 prediction JSON (table_common). Based on the best model (Mask2Former Swin-L).
"""
import os
import sys
import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg
from evaluator import load_json, ann_to_mask, to_label_index_image, _get_selected_annotation
import table_common as tc


class Table2Builder:
    """Aggregate per-class gt/pred counts + mIoU + AP20."""

    def __init__(self, pred_json_path, save_name):
        self.pred_json = pred_json_path
        self.save_name = save_name
        self.gt_json = cfg.COCO_MERGED_ANNO_PATH
        self.label_dir = cfg.label_dir("validation")

    def build(self):
        print(f"pred_json: {self.pred_json}")
        gt_counts, pred_counts = self._count_objects()
        ap20 = self._evaluate_ap20_per_class()
        miou = self._evaluate_miou_per_class()
        rows = [{"class_name": cfg.ID2NAME.get(cid, str(cid)),
                 "gt_count": gt_counts.get(cid, 0),
                 "pred_count": pred_counts.get(cid, 0),
                 "AP20": tc.pct(ap20.get(cid, 0.0)),
                 "mIoU": tc.pct(miou.get(cid, 0.0))}
                for cid in cfg.EVAL_CLASS_IDS]
        tc.save_csv(pd.DataFrame(rows), self.save_name)

    def _count_objects(self):
        gt = load_json(_get_selected_annotation(self.gt_json))["annotations"]
        pred = self._pred_annotations()
        gt_counts = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
        pred_counts = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
        for ann in gt:
            if ann["category_id"] in gt_counts:
                gt_counts[ann["category_id"]] += 1
        for ann in pred:
            if ann.get("category_id") in pred_counts:
                pred_counts[ann["category_id"]] += 1
        return gt_counts, pred_counts

    def _pred_annotations(self):
        data = load_json(self.pred_json)
        return data["annotations"] if isinstance(data, dict) else data

    def _evaluate_ap20_per_class(self):
        coco_gt = COCO(_get_selected_annotation(self.gt_json))
        pred = [d for d in self._pred_annotations() if d.get("category_id") not in cfg.EXCLUDE_IDS]
        coco_eval = COCOeval(coco_gt, coco_gt.loadRes(pred), iouType="segm")
        coco_eval.params.catIds = cfg.EVAL_CLASS_IDS
        coco_eval.params.iouThrs = np.array([0.20], dtype=np.float32)
        coco_eval.evaluate()
        coco_eval.accumulate()
        return {int(cid): self._class_ap(coco_eval, idx)
                for idx, cid in enumerate(coco_eval.params.catIds)}

    def _class_ap(self, coco_eval, class_idx):
        p = coco_eval.eval["precision"][0, :, class_idx, 0, -1]
        p = p[p > -1]
        return float(np.mean(p)) if p.size else 0.0

    def _evaluate_miou_per_class(self):
        ann_idx = self._index_pred_by_image()
        inter = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
        union = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
        for file in tqdm(glob.glob(os.path.join(self.label_dir, "*.png")), desc="per-class mIoU"):
            gt_label = to_label_index_image(cv2.imread(file, cv2.IMREAD_UNCHANGED), True)
            if gt_label is None:
                continue
            h, w = gt_label.shape
            pred_label = self._pred_label(ann_idx, h, w, file)
            for cid in cfg.EVAL_CLASS_IDS:
                inter[cid] += int(np.sum((gt_label == cid) & (pred_label == cid)))
                union[cid] += int(np.sum((gt_label == cid) | (pred_label == cid)))
        return {cid: (inter[cid] / union[cid] if union[cid] else 0.0) for cid in cfg.EVAL_CLASS_IDS}

    def _index_pred_by_image(self):
        idx = {}
        for a in self._pred_annotations():
            if int(a.get("category_id", 0)) in cfg.EXCLUDE_IDS:
                continue
            idx.setdefault(str(a.get("image_id")), []).append(a)
        return idx

    def _pred_label(self, ann_idx, h, w, file):
        label = np.zeros((h, w), dtype=np.int32)
        for ann in ann_idx.get(os.path.basename(file).replace(".png", ""), []):
            label[ann_to_mask(ann, h, w) > 0] = int(ann.get("category_id", 0))
        return label


def main():
    combo = tc.best_combo(pd.read_csv(tc.total_csv_path()))
    Table2Builder(tc.pred_json_path(combo, tc.MERGE_COUNT), "table_2.csv").build()


if __name__ == "__main__":
    main()
