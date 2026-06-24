import os
import glob
import argparse
import pandas as pd

import config as cfg
from lane_stitcher import LaneStitcher
from evaluator import evaluate_all, evaluate_coco_ap, _filename_to_merge_count


BEST_MODEL = 'mask2former_large'   # Swin-L 백본, AP20 최고 모델. 이 모델만 전체 파라미터 스윕


def run_experiments(visualize=True):
    sample_strides = [5, 10]
    extend_lens = [10, 20, 30]
    turn_penalties = [0, 3, 5]
    thicknesses = 3

    model_paths = find_model_paths(cfg.DATA_ROOT)
    coco_gt_json = cfg.COCO_MERGED_ANNO_PATH
    label_path = os.path.join(cfg.DATASET_PATH, 'annotations', 'validation')

    best_dir = cfg.MODEL_PREFIX + BEST_MODEL
    best_path = next((p for p in model_paths if os.path.basename(p) == best_dir), None)
    if best_path is None:
        print(f"경고: 최고 모델 '{best_dir}' 미발견 -> 첫 모델로 대체")
        best_path = model_paths[0]
    other_paths = [p for p in model_paths if p != best_path]

    combos = [(s, e, tp) for s in sample_strides for e in extend_lens for tp in turn_penalties]
    total_runs = len(combos) + len(other_paths)
    print(f"Total Experiments: {total_runs} "
          f"(최고 모델 {len(combos)} 조합 전체 스윕 + 나머지 {len(other_paths)} 모델 × 최적 1조합)")

    # 1) 최고 모델: 전체 파라미터 스윕
    run_idx = 0
    best_name = os.path.basename(best_path)
    for s, e, tp in combos:
        run_idx += 1
        run_single_experiment(best_path, best_name, thicknesses, s, e, tp,
                              coco_gt_json, label_path, visualize, run_idx, total_runs)

    # 2) 최고 모델의 최적 파라미터(AP20 최대) 결정
    bs, be, btp = _best_param_combo(best_name)
    print(f"\n최고 모델({BEST_MODEL}) 최적 파라미터: stride={bs}, extend={be}, turn={btp}")

    # 3) 나머지 모델: 모델만 교체해 최적 파라미터로 1회씩
    for p in other_paths:
        run_idx += 1
        run_single_experiment(p, os.path.basename(p), thicknesses, bs, be, btp,
                              coco_gt_json, label_path, visualize, run_idx, total_runs)

    print("\nAll experiments completed.")


def _best_param_combo(model_dir_name):
    """주어진 모델의 eval_result.csv들에서 AP20 최대 행의 (stride, extend, turn)을 반환한다."""
    csvs = glob.glob(os.path.join(cfg.RESULT_PATH, model_dir_name, '*', 'eval_result.csv'))
    rows = pd.concat([parse_single_csv(f) for f in csvs], ignore_index=True)
    best = rows.loc[rows['AP20'].idxmax()]
    return int(best['sample_strides']), int(best['extend_lens']), int(best['turn_penalties'])


def run_single_experiment(model_path, model_name, t, s, e, tp, coco_gt_json, label_path,
                          visualize=True, run_idx=1, total_runs=1):
    param_name = f"thick={t},stride={s},extend={e},turn={tp}"
    result_path = os.path.join(cfg.RESULT_PATH, model_name, param_name)
    os.makedirs(result_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Running: Model = {model_name}")
    print(f"         Params = {param_name}")
    print(f"         Result Path = {result_path}")
    print('='*80)

    detector = LaneStitcher(
        data_path=cfg.DATASET_PATH,
        pred_path=model_path,
        result_path=result_path,
        thickness=t,
        sample_stride=s,
        extend_len=e,
        visualize=visualize
    )
    detector.turn_penalty = tp  # 클래스 속성(생성자 인자 아님)을 인스턴스에서 오버라이드
    # param_name은 위 배너에 표시되므로 desc에선 생략 → 진행바 폭 확보
    desc = f"[조합 {run_idx}/{total_runs}] {model_name}"
    detector.detect_lines(desc=desc)
    evaluate_all(coco_gt_json, label_path, model_path, result_path)


def eval_only():
    """detection을 건너뛰고 기존 예측 JSON(coco_pred_instances_*.json)으로 평가만 다시 한다.

    GT(merged_annotations.json)만 바뀐 경우, mIoU는 label PNG·예측 JSON에만 의존하므로
    변하지 않고 AP만 바뀐다. 따라서 AP만 재계산하고 mIoU/instances는 기존 eval_result.csv
    에서 그대로 재사용한다(기존 CSV가 없으면 전체 평가로 폴백).
    selected_annotation.json 캐시는 GT가 더 최신이면 evaluator가 자동 무효화한다."""
    model_by_name = {os.path.basename(p): p for p in find_model_paths(cfg.DATA_ROOT)}
    coco_gt_json = cfg.COCO_MERGED_ANNO_PATH
    label_path = os.path.join(cfg.DATASET_PATH, 'annotations', 'validation')

    pred_files = glob.glob(os.path.join(cfg.RESULT_PATH, '*', '*', 'coco_pred_instances_*.json'))
    result_dirs = sorted({os.path.dirname(f) for f in pred_files})
    print(f"Re-evaluating AP for {len(result_dirs)} result dirs (detection·mIoU 생략, GT={coco_gt_json})")

    for result_path in result_dirs:
        # 경로 구조: RESULT_PATH / [model_name] / [param_name]
        model_name = os.path.basename(os.path.dirname(result_path))
        print(f"\n{'='*80}\nEvaluating AP: {model_name} / {os.path.basename(result_path)}\n{'='*80}")
        if _reeval_ap_only(coco_gt_json, result_path):
            continue
        # 기존 eval_result.csv가 없으면 mIoU 포함 전체 평가로 폴백
        model_path = model_by_name.get(model_name)
        if model_path is None:
            print(f"  skip {result_path}: 기존 CSV 없음 + 모델 '{model_name}' 디렉토리도 못 찾음")
            continue
        evaluate_all(coco_gt_json, label_path, model_path, result_path)

    print("\nEval-only(AP) completed.")


def _reeval_ap_only(coco_gt_json, result_path):
    """기존 eval_result.csv의 mIoU/instances는 두고 AP 컬럼만 새 GT로 재계산해 덮어쓴다.

    mIoU는 GT가 아니라 label PNG·예측 JSON에만 의존하므로 GT 변경 시 변하지 않는다.
    기존 CSV가 없으면 False를 반환(호출측에서 전체 평가로 폴백)."""
    csv_path = os.path.join(result_path, 'eval_result.csv')
    if not os.path.exists(csv_path):
        return False
    df = pd.read_csv(csv_path)
    for json_file in sorted(glob.glob(os.path.join(result_path, 'coco_pred_*.json'))):
        mc = _filename_to_merge_count(json_file)
        ap = evaluate_coco_ap(coco_gt_json, json_file)  # {instances, AP10, AP20, AP50}
        mask = df['merge_count'] == mc
        if not mask.any():
            continue
        for col in ('AP10', 'AP20', 'AP50'):  # AP만 갱신, mIoU/instances는 기존값 유지
            if col in df.columns and col in ap:
                df.loc[mask, col] = float(ap[col])
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(df.to_string(index=False))
    return True


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

    merged_df = merged_df.sort_values(by=['model_name', 'sample_strides', 'extend_lens', 'turn_penalties', 'thicknesses', 'merge_count'])

    float_cols = ['AP10', 'AP20', 'AP50', 'mIoU']
    merged_df[float_cols] = merged_df[float_cols].round(4)
    int_cols = ['merge_count', 'thicknesses', 'sample_strides', 'extend_lens', 'turn_penalties', 'instances']
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
    df['turn_penalties'] = int(params.get('turn', 0))
    column_order = ['model_name', 'merge_count', 'thicknesses', 'sample_strides', 'extend_lens', 'turn_penalties', 'instances', 'AP10', 'AP20', 'AP50', 'mIoU']
    return df[column_order]


def main():
    parser = argparse.ArgumentParser(description='하이퍼파라미터 탐색 및 성능평가 실행')
    parser.add_argument('--fast', action='store_true',
                        help='창 표시·시각화 콜라주를 생략하고 성능평가만 빠르게 실행')
    parser.add_argument('--eval-only', action='store_true',
                        help='detection을 건너뛰고 기존 예측 JSON으로 평가만 다시 실행 (GT 변경 시)')
    args = parser.parse_args()
    if args.eval_only:
        eval_only()
    else:
        run_experiments(visualize=not args.fast)
    find_best_model_and_params()


if __name__ == '__main__':
    main()
