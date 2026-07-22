"""Table 1 — model comparison (segmentation vs merge×1), val/test side by side.

Columns: Model | Params(M) | Stage | Instances | F1@0.5(val) | F1@0.5(test) | mIoU(val) | mIoU(test)
Instances is the validation prediction count.
For each model, output a Segmentation row (instances/F1 blank, mIoU only) and a Merge×1 row.
Source: total_performance.csv (best params fixed, val/test columns) + num_params.csv.
If the test columns do not exist yet (i.e. only validation was run), those values are shown as blank.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg
import table_common as tc


class Table1Builder:
    """Summarize segmentation mIoU and merge×1 performance for the 3 models, val/test side by side."""

    def __init__(self, total_csv_path, num_params_path, save_name):
        self.df = tc.with_val_aliases(pd.read_csv(total_csv_path))
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
        # pure segmentation reports mIoU only (instances/F1 are blank)
        seg = sub[sub["merge_count"].isna()].iloc[0]
        return self._row(model, "Segmentation",
                         {sp: {"instances": None, cfg.F1_PRIMARY: None,
                               "mIoU": seg.get(cfg.mcol("mIoU", sp))}
                          for sp in cfg.EVAL_SPLITS})

    def _merge_row(self, model, sub):
        m1 = sub[sub["merge_count"] == tc.MERGE_COUNT].iloc[0]
        return self._row(model, f"Merge×{tc.MERGE_COUNT}",
                         {sp: {"instances": m1.get(cfg.mcol("instances", sp)),
                               cfg.F1_PRIMARY: m1.get(cfg.mcol(cfg.F1_PRIMARY, sp)),
                               "mIoU": m1.get(cfg.mcol("mIoU", sp))}
                          for sp in cfg.EVAL_SPLITS})

    def _row(self, model, stage, per_split):
        # one Instances column (validation count), then F1 val/test and mIoU val/test side by side
        row = {"Model": tc.MODEL_DISPLAY.get(model, model),
               "Params(M)": self.params_map.get(model),
               "Stage": stage,
               "Instances": _int_or_blank(per_split["validation"]["instances"])}
        for metric in (cfg.F1_PRIMARY, "mIoU"):
            for sp in cfg.EVAL_SPLITS:
                row[cfg.mcol(metric, sp)] = _pct_or_blank(per_split[sp][metric])
        return row


def _pct_or_blank(value):
    return tc.BLANK if value is None or pd.isna(value) else tc.pct(value)


def _int_or_blank(value):
    return tc.BLANK if value is None or pd.isna(value) else int(value)


def main():
    Table1Builder(tc.total_csv_path(),
                  os.path.join(cfg.RESULT_PATH, "num_params.csv"),
                  "table_1.csv").build()


if __name__ == "__main__":
    main()
