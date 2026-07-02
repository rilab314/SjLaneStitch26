"""Table 5 — 파라미터 영향 (ablation).

열: sample_stride | extend_len | turn_penalty | Instances | AP20 | mIoU
최고 모델·merge×1·thickness=3 고정. sample_stride·extend_len·turn_penalty 조합별 민감도.
출처: total_performance.csv.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import table_common as tc

PARAMS = ["sample_strides", "extend_lens", "turn_penalties"]
RENAME = {"sample_strides": "sample_stride", "extend_lens": "extend_len",
          "turn_penalties": "turn_penalty", "instances": "Instances"}


class Table5Builder:
    """최고 모델·merge×1에서 stride·extend·turn 조합별 성능을 정리한다."""

    def __init__(self, total_csv_path, save_name):
        self.df = pd.read_csv(total_csv_path)
        self.save_name = save_name

    def build(self):
        combo = tc.best_combo(self.df)
        mask = ((self.df["model_name"] == combo["model_name"]) &
                (self.df["merge_count"] == tc.MERGE_COUNT) &
                (self.df["instances"] > 0))
        sub = self.df[mask].sort_values(PARAMS).reset_index(drop=True)
        result = sub[PARAMS + ["instances", "AP20", "mIoU"]].copy()
        result[PARAMS + ["instances"]] = result[PARAMS + ["instances"]].astype(int)
        result["AP20"] = result["AP20"].map(tc.pct)
        result["mIoU"] = result["mIoU"].map(tc.pct)
        result = result.rename(columns=RENAME)
        print(f"고정: model={combo['model_name']}, merge×{tc.MERGE_COUNT}, thickness=3")
        tc.save_csv(result, self.save_name)


def main():
    Table5Builder(tc.total_csv_path(), "table_5.csv").build()


if __name__ == "__main__":
    main()
