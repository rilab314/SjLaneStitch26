"""Table 1 — 모델 비교 (segmentation vs merge×1), 6줄.

열: Model | Params(M) | Stage | Instances | AP20 | mIoU
각 모델마다 Segmentation 행(인스턴스·AP20 공란, mIoU만)과 Merge×1 행을 출력한다.
출처: total_performance.csv(best 파라미터 고정) + num_params.csv.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
import table_common as tc


class Table1Builder:
    """3모델의 segmentation mIoU와 merge×1 성능을 6줄로 정리한다."""

    def __init__(self, total_csv_path, num_params_path, save_name):
        self.df = pd.read_csv(total_csv_path)
        self.params_map = pd.read_csv(num_params_path).set_index("model")["total_params_M"]
        self.save_name = save_name

    def build(self):
        combo = tc.best_combo(self.df)
        at_best = self._filter_best_params(combo)
        rows = []
        for model in tc.MODEL_ORDER:
            sub = at_best[at_best["model_name"] == model]
            if sub.empty:
                continue
            rows.append(self._segmentation_row(model, sub))
            rows.append(self._merge_row(model, sub))
        tc.save_csv(pd.DataFrame(rows), self.save_name)

    def _filter_best_params(self, combo):
        mask = ((self.df["thicknesses"] == combo["thicknesses"]) &
                (self.df["sample_strides"] == combo["sample_strides"]) &
                (self.df["extend_lens"] == combo["extend_lens"]) &
                (self.df["turn_penalties"] == combo["turn_penalties"]))
        return self.df[mask]

    def _segmentation_row(self, model, sub):
        seg = sub[sub["merge_count"].isna()].iloc[0]
        return self._row(model, "Segmentation", tc.BLANK, tc.BLANK, seg["mIoU"])

    def _merge_row(self, model, sub):
        m1 = sub[sub["merge_count"] == tc.MERGE_COUNT].iloc[0]
        return self._row(model, f"Merge×{tc.MERGE_COUNT}",
                         int(m1["instances"]), tc.pct(m1["AP20"]), m1["mIoU"])

    def _row(self, model, stage, instances, ap20, miou):
        return {"Model": tc.MODEL_DISPLAY.get(model, model),
                "Params(M)": self.params_map.get(model),
                "Stage": stage, "Instances": instances,
                "AP20": ap20, "mIoU": tc.pct(miou)}


def main():
    Table1Builder(tc.total_csv_path(),
                  os.path.join(cfg.RESULT_PATH, "num_params.csv"),
                  "table_1.csv").build()


if __name__ == "__main__":
    main()
