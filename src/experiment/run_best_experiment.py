"""Run experiments for only the single optimal combination in total_performance.csv.

Instead of the full parameter sweep in run_experiments.py, generate the prediction JSON
(coco_pred_instances_origin/merge*.json) and eval_result.csv for only the best AP20 model/parameter combination.
Reflects the current algorithm (num_merges, min_free_len, etc.) as-is.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
from stitch_config import build_config_from_csv, load_stitch_config, resolve_model_path
from run_experiments import run_single_experiment

# Reference result folder for determining the best combination (uses the highest-AP20 row in its total_performance.csv).
# If None, uses the CSV in the current RESULT_PATH.
REFERENCE_DIR = "results_260624"


def resolve_best_config():
    """If a reference folder is specified, use its CSV; otherwise use the CSV in the current RESULT_PATH to pick the best combination."""
    if REFERENCE_DIR:
        ref_csv = os.path.join(cfg.DATA_ROOT, REFERENCE_DIR, "total_performance.csv")
        if os.path.exists(ref_csv):
            print(f"[best] reference: {ref_csv}")
            return build_config_from_csv(ref_csv)
    return load_stitch_config()


def main():
    config = resolve_best_config()
    model_path = resolve_model_path(config.model_name)
    model_name = os.path.basename(model_path)
    # GT (coco AP + mIoU labels) is resolved inside run_single_experiment via cfg per split.
    run_single_experiment(
        model_path, model_name,
        config.thickness, config.sample_stride, config.extend_len,
        int(round(config.turn_penalty)),
        'validation',
        visualize=False, run_idx=1, total_runs=1)


if __name__ == "__main__":
    main()
