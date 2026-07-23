"""Table 4 — stage-wise performance improvement of the best model, 5 rows.

Columns: Stage | Instances | F1@0.5 | mIoU
Cumulative stages: Baseline (pure vectorization) -> + residual re-extraction -> + refinement -> merge×1 -> merge×2.
Refinement/merge1/merge2 (stages 3-5) are read directly from total_performance.csv,
and only the baseline/residual (stages 1-2), which are absent from total_performance.csv, are freshly evaluated via stage_linestrings.
"""
import os
import sys
import glob
import json

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg
from lane_stitcher import LaneStitcher
from stitch_config import load_stitch_config
from evaluator import evaluate_f1, evaluate_miou_json
import table_common as tc


class Table4Builder:
    """Summarize stage-wise (baseline/residual/refinement/merge1/merge2) F1/mIoU/instance counts in 5 rows."""

    def __init__(self, total_csv_path, save_name):
        self.df = tc.with_val_aliases(pd.read_csv(total_csv_path))
        self.save_name = save_name
        self.combo = tc.best_combo(self.df)

    def build(self):
        fresh = self._evaluate_first_and_combined()
        rows = [
            self._row("Baseline", fresh["first"]),
            self._row("+ Residual re-extraction", fresh["combined"]),
            self._row("+ Refinement", self._csv_stage(0)),
            self._row("+ Merge ×1", self._csv_stage(1)),
            self._row("+ Merge ×2", self._csv_stage(2)),
        ]
        tc.save_csv(pd.DataFrame(rows), self.save_name)

    def _row(self, stage, metrics):
        return {"Stage": stage, "Instances": int(metrics["instances"]),
                cfg.F1_PRIMARY: tc.pct(metrics[cfg.F1_PRIMARY]),
                "mIoU": tc.pct(metrics["mIoU"])}

    def _csv_stage(self, merge_count):
        """Read the metrics of the best combo, merge_count row from total_performance.csv."""
        c = self.combo
        mask = ((self.df["model_name"] == c["model_name"]) &
                (self.df["thicknesses"] == c["thicknesses"]) &
                (self.df["sample_strides"] == c["sample_strides"]) &
                (self.df["extend_lens"] == c["extend_lens"]) &
                (self.df["turn_penalties"] == c["turn_penalties"]) &
                (self.df["merge_count"] == merge_count))
        row = self.df[mask].iloc[0]
        return {"instances": row["instances"], cfg.F1_PRIMARY: row[cfg.F1_PRIMARY],
                "mIoU": row["mIoU"]}

    def _evaluate_first_and_combined(self):
        """Build baseline(first)/residual stage predictions via stage_linestrings and freshly evaluate them."""
        first_json, combined_json = self._build_stage_predictions()
        return {"first": self._evaluate(first_json),
                "combined": self._evaluate(combined_json)}

    def _build_stage_predictions(self):
        det = self._build_detector()
        files = sorted(glob.glob(os.path.join(cfg.image_dir("validation"), "*.png")))
        first_preds, combined_preds = [], []
        for f in tqdm(files, desc="first/residual stage extraction"):
            image_id = os.path.basename(f)[:-4]
            stage = det.stage_linestrings(f, do_merge=False)
            first_preds += det.convert_to_json(det._smoothed_copies(stage["first"]), image_id)
            combined_preds += det.convert_to_json(det._smoothed_copies(stage["combined"]), image_id)
        return (self._dump(first_preds, "_stage_first.json"),
                self._dump(combined_preds, "_stage_combined.json"))

    def _build_detector(self):
        sc = load_stitch_config()
        det = LaneStitcher(cfg.DATASET_PATH, sc.model_path, cfg.RESULT_PATH,
                           thickness=sc.thickness, sample_stride=sc.sample_stride,
                           extend_len=sc.extend_len, visualize=False)
        det.turn_penalty = sc.turn_penalty
        return det

    def _dump(self, preds, name):
        os.makedirs(tc.tables_dir(), exist_ok=True)
        path = os.path.join(tc.tables_dir(), name)
        with open(path, "w") as fp:
            json.dump(preds, fp)
        return path

    def _evaluate(self, pred_json):
        f1 = evaluate_f1(cfg.COCO_MERGED_ANNO_PATH, pred_json)
        # Same mIoU label basis as the total_performance sweep (both read ade20k/annotations/validation).
        miou = evaluate_miou_json(pred_json, cfg.label_dir("validation"))
        return {"instances": f1["instances"], cfg.F1_PRIMARY: f1[cfg.F1_PRIMARY],
                "mIoU": miou["mIoU"]}


def main():
    Table4Builder(tc.total_csv_path(), "table_4.csv").build()


if __name__ == "__main__":
    main()
