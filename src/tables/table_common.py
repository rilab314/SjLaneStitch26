"""Shared helpers for Tables 1-5: best-combo lookup, prediction JSON paths, model display names, output formatting."""
import os

import pandas as pd

import config as cfg

# Operating point for the main tables: merge×1 (merge×2 is shown separately only in the Table 4 stage-wise table).
MERGE_COUNT = 1

# Internal model name in the CSV -> paper display name.
MODEL_DISPLAY = {
    "mask2former_large": "Mask2Former (Swin-L)",
    "mask2former_small": "Mask2Former (Swin-S)",
    "internimage_large": "InternImage-L",
}
# Row order of models in the table output.
MODEL_ORDER = ["mask2former_large", "mask2former_small", "internimage_large"]

BLANK = "–"  # en dash, blank-cell marker


def total_csv_path():
    return os.path.join(cfg.RESULT_PATH, "total_performance.csv")


def tables_dir():
    return os.path.join(cfg.RESULT_PATH, "Tables")


def with_val_aliases(df):
    """Add suffix-less aliases for the (val) metric columns of total_performance/eval.
    A backward-compatibility shim so existing table scripts can read validation values directly via df['F1@0.5'], etc.
    (Test columns are accessed directly via df['F1@0.5(test)'].)"""
    for m in ("instances", *cfg.F1_METRICS, "mIoU"):
        v = cfg.mcol(m, "validation")
        if v in df.columns and m not in df.columns:
            df[m] = df[v]
    return df


def load_total_performance():
    """Read total_performance.csv and return a DataFrame with (val) aliases attached."""
    return with_val_aliases(pd.read_csv(total_csv_path()))


def best_combo(df):
    """Return the model and hyperparameters (dict) of the row with the highest F1(val)."""
    metric = cfg.primary_metric_col(df.columns)
    best = df.sort_values(metric, ascending=False, na_position="last").iloc[0]
    return {
        "model_name": str(best["model_name"]),
        "thicknesses": int(best["thicknesses"]),
        "sample_strides": int(best["sample_strides"]),
        "extend_lens": int(best["extend_lens"]),
        "turn_penalties": int(best["turn_penalties"]),
    }


def param_dir_name(combo):
    return (f"thick={combo['thicknesses']},stride={combo['sample_strides']},"
            f"extend={combo['extend_lens']},turn={combo['turn_penalties']}")


def pred_json_path(combo, merge_count, split="validation"):
    """Prediction JSON path for the best combo and merge stage (validation by default)."""
    model_dir = cfg.MODEL_PREFIX + combo["model_name"]
    name = "origin" if merge_count == 0 else f"merge{merge_count}"
    return os.path.join(cfg.RESULT_PATH, model_dir, param_dir_name(combo),
                        f"coco_pred_{cfg.split_label(split)}_{name}.json")


def pct(value):
    """Ratio (0-1) -> percentage rounded to two decimal places."""
    return round(float(value) * 100, 2)


def save_csv(df, name):
    out_dir = tables_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"saved: {path}")
    print(df.to_string(index=False))
    return path
