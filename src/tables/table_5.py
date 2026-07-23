"""Table 5 — parameter impact (ablation).

Columns: sample_stride | extend_len | turn_penalty | Instances | F1@0.5 | mIoU
Best model, merge×1, thickness=3 fixed. Sensitivity per sample_stride/extend_len/turn_penalty combination.
Source: total_performance.csv.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg
import table_common as tc

PARAMS = ["sample_strides", "extend_lens", "turn_penalties"]
RENAME = {"sample_strides": "sample_stride", "extend_lens": "extend_len",
          "turn_penalties": "turn_penalty", "instances": "Instances"}


class Table5Builder:
    """Summarize performance per stride/extend/turn combination for the best model, merge×1."""

    def __init__(self, total_csv_path, save_name):
        self.df = tc.with_val_aliases(pd.read_csv(total_csv_path))
        self.save_name = save_name

    def build(self):
        combo = tc.best_combo(self.df)
        mask = ((self.df["model_name"] == combo["model_name"]) &
                (self.df["merge_count"] == tc.MERGE_COUNT) &
                (self.df["instances"] > 0))
        sub = self.df[mask].sort_values(PARAMS).reset_index(drop=True)
        result = sub[PARAMS + ["instances", cfg.F1_PRIMARY, "mIoU"]].copy()
        result[PARAMS + ["instances"]] = result[PARAMS + ["instances"]].astype(int)
        result[cfg.F1_PRIMARY] = result[cfg.F1_PRIMARY].map(tc.pct)
        result["mIoU"] = result["mIoU"].map(tc.pct)
        result = result.rename(columns=RENAME)
        print(f"fixed: model={combo['model_name']}, merge×{tc.MERGE_COUNT}, thickness=3")
        tc.save_csv(result, self.save_name)


def main():
    Table5Builder(tc.total_csv_path(), "table_5.csv").build()


if __name__ == "__main__":
    main()
