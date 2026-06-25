"""total_performance.csv의 최적 조합 1개에 대해서만 실험을 수행한다.

run_experiments.py의 전체 파라미터 스윕 대신, 최고 AP20 모델·파라미터 조합으로만
예측 JSON(coco_pred_instances_origin/merge*.json)과 eval_result.csv를 생성한다.
현재 알고리즘(num_merges, min_free_len 등)을 그대로 반영한다.
"""
import os

import config as cfg
from stitch_config import build_config_from_csv, load_stitch_config, resolve_model_path
from run_experiments import run_single_experiment

# best 조합을 결정할 기준 결과 폴더(여기 total_performance.csv의 AP20 최고 행을 사용).
# None이면 현재 RESULT_PATH의 CSV를 쓴다.
REFERENCE_DIR = "results_260624"


def resolve_best_config():
    """기준 폴더가 지정되면 그 CSV로, 아니면 현재 RESULT_PATH의 CSV로 best 조합을 정한다."""
    if REFERENCE_DIR:
        ref_csv = os.path.join(cfg.DATA_ROOT, REFERENCE_DIR, "total_performance.csv")
        if os.path.exists(ref_csv):
            print(f"[best] 기준: {ref_csv}")
            return build_config_from_csv(ref_csv)
    return load_stitch_config()


def main():
    config = resolve_best_config()
    model_path = resolve_model_path(config.model_name)
    model_name = os.path.basename(model_path)
    label_path = os.path.join(cfg.DATASET_PATH, "annotations", "validation")
    run_single_experiment(
        model_path, model_name,
        config.thickness, config.sample_stride, config.extend_len,
        int(round(config.turn_penalty)),
        cfg.COCO_MERGED_ANNO_PATH, label_path,
        visualize=False, run_idx=1, total_runs=1)


if __name__ == "__main__":
    main()
