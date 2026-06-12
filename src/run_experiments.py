import os
import glob
import pandas as pd

import config as cfg
from lane_detector import LineStringDetector
from evaluator import evaluate_all


def run_experiments():
    sample_strides = [5, 10]
    extend_lens = [20, 30, 40]
    thicknesses = 3

    model_paths = find_model_paths(cfg.DATA_ROOT)
    coco_gt_json = cfg.COCO_MERGED_ANNO_PATH
    label_path = os.path.join(cfg.DATASET_PATH, 'annotations', 'validation')
    total_runs = len(model_paths) * len(sample_strides) * len(extend_lens)
    print(f"Total Experiments: {total_runs}")

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        for s in sample_strides:
            for e in extend_lens:
                run_single_experiment(model_path, model_name, thicknesses, s, e, coco_gt_json, label_path)

    print("\nAll experiments completed.")


def run_single_experiment(model_path, model_name, t, s, e, coco_gt_json, label_path):
    param_name = f"thick={t},stride={s},extend={e}"
    result_path = os.path.join(cfg.RESULT_PATH, model_name, param_name)
    os.makedirs(result_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Running: Model = {model_name}")
    print(f"         Params = {param_name}")
    print(f"         Result Path = {result_path}")
    print('='*80)

    detector = LineStringDetector(
        data_path=cfg.DATASET_PATH,
        pred_path=model_path,
        result_path=result_path,
        thickness=t,
        sample_stride=s,
        extend_len=e
    )
    detector.detect_lines()
    evaluate_all(coco_gt_json, label_path, model_path, result_path)


def find_model_paths(data_root):
    model_paths = []
    for root, dirs, files in os.walk(data_root):
        if 'prediction' in dirs:
            model_paths.append(root)

    print(f"Found {len(model_paths)} models:")
    for m in model_paths:
        print(f" - {os.path.basename(m)}")
    return model_paths


def find_best_model_and_params():
    print("\n" + "="*80)
    print("Finding the best model and parameter combination...")
    # 경로 구조: RESULT_PATH / [model_name] / [param_name] / eval_result.csv
    csv_files = glob.glob(os.path.join(cfg.RESULT_PATH, '**', 'eval_result.csv'), recursive=True)
    if not csv_files:
        print("No eval_result.csv files found.")
        return

    df_list = [parse_single_csv(f) for f in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    merged_df = merged_df.sort_values(by=['model_name', 'sample_strides', 'extend_lens', 'thicknesses', 'merge_count'])

    float_cols = ['AP10', 'AP20', 'AP50', 'mIoU']
    merged_df[float_cols] = merged_df[float_cols].round(4)
    int_cols = ['merge_count', 'thicknesses', 'sample_strides', 'extend_lens', 'instances']
    merged_df[int_cols] = merged_df[int_cols].astype("Int64")
    save_path = os.path.join(cfg.RESULT_PATH, "total_performance.csv")
    merged_df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"Merged results saved to: {save_path}")

    top10_df = merged_df.sort_values(by='AP20', ascending=False, na_position='last').head(10)
    print("\nTop 10 combinations by AP20:")
    print(top10_df.to_string(index=False))


def parse_single_csv(file_path):
    df = pd.read_csv(file_path)
    # 경로 구조: RESULT_PATH / [model_name] / [param_name] / eval_result.csv
    param_dir = os.path.dirname(file_path)
    model_dir = os.path.dirname(param_dir)
    model_name = os.path.basename(model_dir).replace(cfg.MODEL_PREFIX, '')
    param_str = os.path.basename(param_dir)
    params = dict(item.split('=') for item in param_str.split(','))
    df['model_name'] = model_name
    df['thicknesses'] = int(params.get('thick', 0))
    df['sample_strides'] = int(params.get('stride', 0))
    df['extend_lens'] = int(params.get('extend', 0))
    column_order = ['model_name', 'merge_count', 'thicknesses', 'sample_strides', 'extend_lens', 'instances', 'AP10', 'AP20', 'AP50', 'mIoU']
    return df[column_order]


def main():
    run_experiments()
    find_best_model_and_params()


if __name__ == '__main__':
    main()
