"""Loader for the best combination (model and hyperparameters).

Reads the highest-AP20 row of total_performance.csv and provides it as a StitchConfig.
Shared by figure generation (Figure/figure_base) and single-run experiments (run_best_experiment).
"""
import os
import sys
from dataclasses import dataclass

import pandas as pd

import config as cfg

DEFAULT_MODEL = "mask2former_large"


@dataclass
class StitchConfig:
    """Model and hyperparameter combination used to configure LaneStitcher."""

    model_name: str
    model_path: str
    pred_dir: str
    thickness: int
    sample_stride: int
    extend_len: int
    turn_penalty: float
    merge_count: int


def load_stitch_config(prefer_model=DEFAULT_MODEL):
    """Read the highest-AP20 combination from total_performance.csv and return it as a StitchConfig (defaults if absent)."""
    csv_path = os.path.join(cfg.RESULT_PATH, "total_performance.csv")
    if os.path.exists(csv_path):
        return build_config_from_csv(csv_path)
    print(f"[config] {csv_path} not found -> using default parameters ({prefer_model})")
    return build_default_config(prefer_model)


def build_config_from_csv(csv_path):
    """Select the highest-AP20(val) row from the CSV and build a StitchConfig."""
    frame = pd.read_csv(csv_path)
    ap = cfg.mcol("AP20", "validation") if cfg.mcol("AP20", "validation") in frame.columns else "AP20"
    best = frame.sort_values(ap, ascending=False, na_position="last").iloc[0]
    model_name = str(best["model_name"])
    model_path = resolve_model_path(model_name)
    print(f"[config] best: {model_name} thick={int(best['thicknesses'])} "
          f"stride={int(best['sample_strides'])} extend={int(best['extend_lens'])} "
          f"turn={float(best['turn_penalties'])} merge={int(best['merge_count'])} "
          f"{ap}={float(best[ap]):.4f}")
    return StitchConfig(
        model_name=model_name,
        model_path=model_path,
        pred_dir=cfg.pred_path(model_path, "validation"),
        thickness=int(best["thicknesses"]),
        sample_stride=int(best["sample_strides"]),
        extend_len=int(best["extend_lens"]),
        turn_penalty=float(best["turn_penalties"]),
        merge_count=max(int(best["merge_count"]), 1),
    )


def build_default_config(model_name):
    """Default combination used when the CSV is absent (same as LaneStitcher defaults)."""
    model_path = resolve_model_path(model_name)
    return StitchConfig(model_name, model_path, os.path.join(model_path, "prediction"),
                        thickness=3, sample_stride=10, extend_len=20,
                        turn_penalty=3.0, merge_count=3)


def resolve_model_path(model_name):
    """Model name -> path of the model directory where segmentation predictions are stored."""
    kind = "Internimage" if "internimage" in model_name.lower() else "mask2former"
    return os.path.join(cfg.DATA_ROOT, kind, cfg.MODEL_PREFIX + model_name)
