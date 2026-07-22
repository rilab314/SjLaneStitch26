"""Run the pipeline once, for the best model/hyperparameter combination only.

Instead of the full parameter sweep in run_experiments.py, this generates the prediction JSON
(coco_pred_{val,test}_{origin,merge1,merge2}.json) and eval_result.csv for a single combination:
the highest-F1 row of total_performance.csv when an own sweep exists in RESULT_PATH, otherwise
the published combination (config.BEST_MODEL / config.BEST_PARAMS).

Usage (from src/):
    python experiment/run_best_experiment.py                    # validation
    python experiment/run_best_experiment.py --split test
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
from stitch_config import load_stitch_config, resolve_model_path
from run_experiments import run_single_experiment


def main():
    parser = argparse.ArgumentParser(description='Run the best combination on one split')
    parser.add_argument('--split', default='validation', choices=cfg.EVAL_SPLITS,
                        help='split to run (default: validation)')
    parser.add_argument('--visualize', action='store_true',
                        help='show the processing windows and save the collage images')
    args = parser.parse_args()

    config = load_stitch_config()
    model_path = resolve_model_path(config.model_name)
    # GT (coco instances + mIoU labels) is resolved inside run_single_experiment via cfg per split.
    run_single_experiment(
        model_path, os.path.basename(model_path),
        config.thickness, config.sample_stride, config.extend_len,
        int(round(config.turn_penalty)),
        args.split,
        visualize=args.visualize, run_idx=1, total_runs=1)


if __name__ == "__main__":
    main()
