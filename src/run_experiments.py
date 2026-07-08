import os
import glob
import argparse
import pandas as pd

import config as cfg
from lane_stitcher import LaneStitcher
from evaluator import evaluate_all, evaluate_coco_ap, _filename_to_merge_count


BEST_MODEL = 'mask2former_large'   # Swin-L 백본, AP20 최고 모델. 이 모델만 전체 파라미터 스윕


def run_experiments(splits, visualize=True):
    sample_strides = [5, 10]
    extend_lens = [10, 20, 30]
    turn_penalties = [0, 3, 5]
    thicknesses = 3

    model_paths = find_model_paths(cfg.DATA_ROOT)

    best_dir = cfg.MODEL_PREFIX + BEST_MODEL
    best_path = next((p for p in model_paths if os.path.basename(p) == best_dir), None)
    if best_path is None:
        print(f"경고: 최고 모델 '{best_dir}' 미발견 -> 첫 모델로 대체")
        best_path = model_paths[0]
    other_paths = [p for p in model_paths if p != best_path]
    best_name = os.path.basename(best_path)

    combos = [(s, e, tp) for s in sample_strides for e in extend_lens for tp in turn_penalties]
    # 파라미터 스윕·최적값 결정은 validation에서만 수행하고(테스트셋 튜닝 방지),
    # test는 그 최적 파라미터를 그대로 적용해 1회씩만 평가한다.
    run_val = 'validation' in splits
    run_test = 'test' in splits
    total_runs = (len(combos) + len(other_paths) if run_val else 0) + \
                 (1 + len(other_paths) if run_test else 0)
    print(f"Total Experiments: {total_runs}  (splits={list(splits)}; "
          f"val={'스윕%d+나머지%d' % (len(combos), len(other_paths)) if run_val else '생략'}, "
          f"test={'전모델%d' % (1 + len(other_paths)) if run_test else '생략'})")

    run_idx = 0
    best_short = best_name.replace(cfg.MODEL_PREFIX, '')
    # 1) [validation] 최고 모델 전체 파라미터 스윕 → total_performance 출력 → 나머지 모델 최적 1조합
    if run_val:
        for s, e, tp in combos:
            run_idx += 1
            run_single_experiment(best_path, best_name, thicknesses, s, e, tp,
                                  'validation', visualize, run_idx, total_runs)
        find_best_model_and_params()  # validation 스윕 결과로 total_performance.csv(=val 열) 출력
        bs, be, btp = _best_param_combo(best_short)
        print(f"\n최고 모델({BEST_MODEL}) 최적 파라미터(validation): stride={bs}, extend={be}, turn={btp}")
        for p in other_paths:
            run_idx += 1
            run_single_experiment(p, os.path.basename(p), thicknesses, bs, be, btp,
                                  'validation', visualize, run_idx, total_runs)
        find_best_model_and_params()  # 나머지 모델 포함해 total_performance.csv 재작성

    # 2) [test] total_performance.csv에서 validation 최적 파라미터를 읽어 전 모델 1회씩 평가.
    #    (test는 스윕 없이 val 최적값 적용 → 테스트셋 튜닝 방지)
    if run_test:
        try:
            bs, be, btp = _best_param_combo(best_short)
        except (FileNotFoundError, ValueError, KeyError):
            print("오류: test 평가에 필요한 validation total_performance.csv(AP20(val))가 없습니다. "
                  "먼저 '--split validation'(또는 split 미지정)으로 validation을 실행하세요.")
            return
        print(f"\ntest 평가 파라미터(validation 최적): stride={bs}, extend={be}, turn={btp}")
        for p in [best_path] + other_paths:
            run_idx += 1
            run_single_experiment(p, os.path.basename(p), thicknesses, bs, be, btp,
                                  'test', visualize, run_idx, total_runs)
        find_best_model_and_params()  # total_performance.csv에 test 열 추가

    print("\nAll experiments completed.")


def _best_param_combo(model_name_short):
    """total_performance.csv에서 해당 모델의 AP20(val) 최고 행의 (stride, extend, turn)."""
    tp_path = os.path.join(cfg.RESULT_PATH, "total_performance.csv")
    df = pd.read_csv(tp_path)
    df = df[df["model_name"] == model_name_short]
    ap = cfg.mcol("AP20", "validation")  # 'AP20(val)'
    best = df.loc[df[ap].idxmax()]
    return int(best["sample_strides"]), int(best["extend_lens"]), int(best["turn_penalties"])


def run_single_experiment(model_path, model_name, t, s, e, tp, split,
                          visualize=True, run_idx=1, total_runs=1):
    param_name = f"thick={t},stride={s},extend={e},turn={tp}"
    result_path = os.path.join(cfg.split_result_path(split), model_name, param_name)
    os.makedirs(result_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Running: Model = {model_name}  [split={split}]")
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
        visualize=visualize,
        split=split,
    )
    detector.turn_penalty = tp  # 클래스 속성(생성자 인자 아님)을 인스턴스에서 오버라이드
    # param_name은 위 배너에 표시되므로 desc에선 생략 → 진행바 폭 확보
    desc = f"[조합 {run_idx}/{total_runs}] {model_name}[{split}]"
    detector.detect_lines(desc=desc)
    evaluate_all(cfg.coco_anno_path(split), cfg.label_dir(split), model_path, result_path, split)


def eval_only(splits=None):
    """detection을 건너뛰고 기존 예측 JSON(coco_pred_instances_*.json)으로 평가만 다시 한다.

    GT(merged_annotations_{split}.json)만 바뀐 경우, mIoU는 label PNG·예측 JSON에만
    의존하므로 변하지 않고 AP만 바뀐다. 따라서 AP만 재계산하고 mIoU/instances는 기존
    eval_result.csv에서 그대로 재사용한다(기존 CSV가 없으면 전체 평가로 폴백). val·test를
    모두 처리한다. selected_annotation 캐시는 GT가 더 최신이면 evaluator가 자동 무효화한다."""
    model_by_name = {os.path.basename(p): p for p in find_model_paths(cfg.DATA_ROOT)}

    for split in (splits or cfg.EVAL_SPLITS):
        label = cfg.split_label(split)
        coco_gt_json = cfg.coco_anno_path(split)
        label_path = cfg.label_dir(split)
        pred_files = glob.glob(os.path.join(cfg.RESULT_PATH, '*', '*',
                                            f'coco_pred_{label}_*.json'))
        result_dirs = sorted({os.path.dirname(f) for f in pred_files})
        print(f"[{split}] Re-evaluating AP for {len(result_dirs)} result dirs "
              f"(detection·mIoU 생략, GT={coco_gt_json})")

        for result_path in result_dirs:
            # 경로 구조: RESULT_PATH / [model_name] / [param_name]
            model_name = os.path.basename(os.path.dirname(result_path))
            print(f"\n{'='*80}\nEvaluating AP[{split}]: {model_name} / {os.path.basename(result_path)}\n{'='*80}")
            if _reeval_ap_only(coco_gt_json, result_path, split):
                continue
            # 기존 eval_result.csv가 없으면 mIoU 포함 전체 평가로 폴백
            model_path = model_by_name.get(model_name)
            if model_path is None:
                print(f"  skip {result_path}: 기존 CSV 없음 + 모델 '{model_name}' 디렉토리도 못 찾음")
                continue
            evaluate_all(coco_gt_json, label_path, model_path, result_path, split)

    print("\nEval-only(AP) completed.")


def _reeval_ap_only(coco_gt_json, result_path, split):
    """기존 eval_result.csv의 mIoU/instances는 두고 해당 split의 AP 열만 새 GT로 재계산해 덮어쓴다.

    mIoU는 GT가 아니라 label PNG·예측 JSON에만 의존하므로 GT 변경 시 변하지 않는다.
    현재 split의 AP 열이 CSV에 없으면 False를 반환(호출측에서 전체 평가로 폴백)."""
    csv_path = os.path.join(result_path, 'eval_result.csv')
    if not os.path.exists(csv_path):
        return False
    label = cfg.split_label(split)
    df = pd.read_csv(csv_path)
    ap_cols = {m: cfg.mcol(m, split) for m in ('AP10', 'AP20', 'AP50')}  # {'AP10':'AP10(val)',...}
    if not any(c in df.columns for c in ap_cols.values()):
        return False  # 이 split 열이 아직 없음 -> 전체 평가로
    for json_file in sorted(glob.glob(os.path.join(result_path, f'coco_pred_{label}_*.json'))):
        mc = _filename_to_merge_count(json_file)
        ap = evaluate_coco_ap(coco_gt_json, json_file)  # {instances, AP10, AP20, AP50}
        mask = df['merge_count'] == mc
        if not mask.any():
            continue
        for m, col in ap_cols.items():  # AP만 갱신, mIoU/instances는 기존값 유지
            if col in df.columns and m in ap:
                df.loc[mask, col] = float(ap[m])
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(df.to_string(index=False))
    return True


def find_model_paths(data_root):
    """모델 예측 폴더(pred_val)를 가진 디렉토리를 모델 경로로 수집한다."""
    val_pred = cfg.pred_dirname('validation')
    model_paths = []
    for root, dirs, files in os.walk(data_root):
        if val_pred in dirs:
            model_paths.append(root)

    print(f"Found {len(model_paths)} models:")
    for m in model_paths:
        print(f" - {os.path.basename(m)}")
    return model_paths


def find_best_model_and_params():
    """모든 param 폴더의 eval_result.csv(val·test 열 포함)를 모아 total_performance.csv 재작성.
    validation 실행 후엔 val 열만, test 실행 후엔 test 열까지 채워진다."""
    print("\n" + "="*80)
    print("Finding the best model and parameter combination...")
    # 경로 구조: RESULT_PATH / [model_name] / [param_name] / eval_result.csv
    csv_files = glob.glob(os.path.join(cfg.RESULT_PATH, '*', '*', 'eval_result.csv'))
    if not csv_files:
        print("No eval_result.csv files found.")
        return

    merged_df = pd.concat([parse_single_csv(f) for f in csv_files], ignore_index=True)
    merged_df = merged_df.sort_values(
        by=['model_name', 'sample_strides', 'extend_lens', 'turn_penalties', 'thicknesses', 'merge_count'])
    merged_df = _order_total_columns(merged_df)

    save_path = os.path.join(cfg.RESULT_PATH, "total_performance.csv")
    merged_df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"Merged results saved to: {save_path}")

    ap = cfg.mcol('AP20', 'validation')  # 'AP20(val)'
    if ap in merged_df.columns:
        top10 = merged_df.sort_values(by=ap, ascending=False, na_position='last').head(10)
        print(f"\nTop 10 combinations by {ap}:")
        print(top10.to_string(index=False))


def _order_total_columns(df):
    """열 순서: 메타데이터 → 지표(split=val→test, instances/AP10/AP20/AP50/mIoU 순). 반올림·형변환."""
    meta = ['model_name', 'merge_count', 'thicknesses', 'sample_strides', 'extend_lens', 'turn_penalties']
    metrics = ['instances', 'AP10', 'AP20', 'AP50', 'mIoU']
    metric_cols = [f'{m}({l})' for l in ('val', 'test') for m in metrics if f'{m}({l})' in df.columns]
    for c in metric_cols:
        num = pd.to_numeric(df[c], errors='coerce')
        df[c] = num.astype('Int64') if c.startswith('instances') else num.round(4)
    for c in ['merge_count', 'thicknesses', 'sample_strides', 'extend_lens', 'turn_penalties']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    ordered = [c for c in meta if c in df.columns] + metric_cols
    ordered += [c for c in df.columns if c not in ordered]
    return df[ordered]


def parse_single_csv(file_path):
    """eval_result.csv(merge_count + …(val)/…(test) 열)에 모델·파라미터 메타데이터를 붙인다."""
    df = pd.read_csv(file_path)
    # 경로 구조: RESULT_PATH / [model_name] / [param_name] / eval_result.csv
    param_dir = os.path.dirname(file_path)
    model_dir = os.path.dirname(param_dir)
    model_name = os.path.basename(model_dir).replace(cfg.MODEL_PREFIX, '')
    params = dict(item.split('=') for item in os.path.basename(param_dir).split(','))
    df.insert(0, 'model_name', model_name)
    df['thicknesses'] = int(params.get('thick', 0))
    df['sample_strides'] = int(params.get('stride', 0))
    df['extend_lens'] = int(params.get('extend', 0))
    df['turn_penalties'] = int(params.get('turn', 0))
    return df


def main():
    parser = argparse.ArgumentParser(description='하이퍼파라미터 탐색 및 성능평가 실행')
    parser.add_argument('--fast', action='store_true',
                        help='창 표시·시각화 콜라주를 생략하고 성능평가만 빠르게 실행')
    parser.add_argument('--eval-only', action='store_true',
                        help='detection을 건너뛰고 기존 예측 JSON으로 평가만 다시 실행 (GT 변경 시)')
    parser.add_argument('--split', nargs='+', choices=['validation', 'test'],
                        default=list(cfg.EVAL_SPLITS),
                        help='평가할 split 선택 (기본: validation test 모두). '
                             '예: --split test (test만), --split validation test (동시)')
    args = parser.parse_args()
    splits = args.split
    if args.eval_only:
        eval_only(splits)
        find_best_model_and_params()
    else:
        # run_experiments가 split 단계별로 total_performance.csv를 갱신한다.
        run_experiments(splits, visualize=not args.fast)


if __name__ == '__main__':
    main()
