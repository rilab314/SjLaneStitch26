"""Table 2 — per-class performance of the best model, 9 rows.

Columns: class_name | gt_count | pred_count | F1@0.5 | mIoU
F1@0.5 is the harmonic mean of the Table 3 precision/recall (same IoU 0.5 matching).
Source: best combo, merge×1 prediction JSON (table_common). Based on the best model (Mask2Former Swin-L).
"""
import os
import sys
import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg
from evaluator import load_json, ann_to_mask, to_label_index_image, evaluate_f1_per_class
import table_common as tc


class Table2Builder:
    """Aggregate per-class gt/pred counts + F1@0.5 + mIoU."""

    def __init__(self, pred_json_path, save_name):
        self.pred_json = pred_json_path
        self.save_name = save_name
        self.gt_json = cfg.COCO_MERGED_ANNO_PATH
        self.label_dir = cfg.label_dir("validation")

    def build(self):
        print(f"pred_json: {self.pred_json}")
        f1 = evaluate_f1_per_class(self.gt_json, self.pred_json)
        miou = self._evaluate_miou_per_class()
        rows = [{"class_name": cfg.ID2NAME.get(cid, str(cid)),
                 "gt_count": f1[cid]["n_gt"],
                 "pred_count": f1[cid]["n_pred"],
                 cfg.F1_PRIMARY: tc.pct(f1[cid][cfg.F1_PRIMARY]),
                 "mIoU": tc.pct(miou.get(cid, 0.0))}
                for cid in cfg.EVAL_CLASS_IDS]
        tc.save_csv(pd.DataFrame(rows), self.save_name)

    def _pred_annotations(self):
        data = load_json(self.pred_json)
        return data["annotations"] if isinstance(data, dict) else data

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
