"""Table 3 — per-class diagnostic breakdown of the best model, 9 rows.

Columns: class_name | precision | recall | near_miss_gt | merge_ratio | miou_match
All based on IoU 0.5 matching (the F1@0.5 operating point; F1 = 2PR/(P+R) of these
precision/recall backs the Table 2 values). merge_ratio>1=merging, near_miss_gt=0<IoU<0.5,
miou_match=average IoU of matched pairs. (Non-manuscript columns such as near_miss_pix/frag_ratio/fp_* are excluded)
Source: best combo, merge×1 prediction JSON.
"""
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg
from evaluator import (load_json, _get_selected_annotation,
                       group_anns_by_image_class, iou_matrix, greedy_match)
import table_common as tc

MATCH_IOU = cfg.F1_IOUS[0]  # 0.5 — same operating point as F1@0.5


class Table3Builder:
    """Aggregate diagnostic metrics via prediction<->GT IoU matching within the same image and class."""

    def __init__(self, pred_json_path, save_name):
        self.pred_json = pred_json_path
        self.save_name = save_name
        self.gt_json = cfg.COCO_MERGED_ANNO_PATH

    def build(self):
        print(f"pred_json: {self.pred_json}")
        gt_idx = group_anns_by_image_class(load_json(_get_selected_annotation(self.gt_json))["annotations"])
        pred_idx = group_anns_by_image_class(self._pred_annotations())
        acc = {cid: self._new_acc() for cid in cfg.EVAL_CLASS_IDS}
        for img_id in tqdm(set(gt_idx) | set(pred_idx), desc="diagnostic metrics"):
            for cid in cfg.EVAL_CLASS_IDS:
                self._accumulate(acc[cid], gt_idx.get(img_id, {}).get(cid, []),
                                 pred_idx.get(img_id, {}).get(cid, []))
        rows = [{"class_name": cfg.ID2NAME.get(cid, str(cid)), **self._finalize(acc[cid])}
                for cid in cfg.EVAL_CLASS_IDS]
        tc.save_csv(pd.DataFrame(rows), self.save_name)

    def _pred_annotations(self):
        data = load_json(self.pred_json)
        return data["annotations"] if isinstance(data, dict) else data

    def _new_acc(self):
        return {"n_gt": 0, "n_pred": 0, "matched": 0, "matched_iou": 0.0,
                "near_miss": 0, "merge_sum": 0}

    def _accumulate(self, acc, gts, prs):
        nG, nP = len(gts), len(prs)
        acc["n_gt"] += nG
        acc["n_pred"] += nP
        iou = iou_matrix(prs, gts)
        for j in range(nG):
            best = float(iou[:, j].max()) if nP else 0.0
            if 0.0 < best < MATCH_IOU:
                acc["near_miss"] += 1
        for i in range(nP):
            acc["merge_sum"] += int((iou[i, :] > 0).sum())
        for v in greedy_match(iou, MATCH_IOU):
            acc["matched"] += 1
            acc["matched_iou"] += v

    def _finalize(self, a):
        nP, nG, M = a["n_pred"], a["n_gt"], a["matched"]
        return {"precision": tc.pct(M / nP if nP else 0.0),
                "recall": tc.pct(M / nG if nG else 0.0),
                "near_miss_gt": tc.pct(a["near_miss"] / nG if nG else 0.0),
                "merge_ratio": round(a["merge_sum"] / nP if nP else 0.0, 3),
                "miou_match": tc.pct(a["matched_iou"] / M if M else 0.0)}


def main():
    combo = tc.best_combo(pd.read_csv(tc.total_csv_path()))
    Table3Builder(tc.pred_json_path(combo, tc.MERGE_COUNT), "table_3.csv").build()


if __name__ == "__main__":
    main()
