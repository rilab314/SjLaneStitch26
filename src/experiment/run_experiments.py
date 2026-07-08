import os
import sys
import glob
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
from lane_stitcher import LaneStitcher
from evaluator import evaluate_all, evaluate_coco_ap, _filename_to_merge_count


BEST_MODEL = 'mask2former_large'   # Swin-L backbone, best AP20 model. Only this model gets the full parameter sweep


def run_experiments(splits, visualize=True):
    sample_strides = [5, 10]
    extend_lens = [10, 20, 30]
    turn_penalties = [0, 3, 5]
    thicknesses = 3

    model_paths = find_model_paths(cfg.DATA_ROOT)

    best_dir = cfg.MODEL_PREFIX + BEST_MODEL
    best_path = next((p for p in model_paths if os.path.basename(p) == best_dir), None)
    if best_path is None:
        print(f"Warning: best model '{best_dir}' not found -> falling back to first model")
        best_path = model_paths[0]
    other_paths = [p for p in model_paths if p != best_path]
    best_name = os.path.basename(best_path)

    combos = [(s, e, tp) for s in sample_strides for e in extend_lens for tp in turn_penalties]
    # Parameter sweep and optimal-value selection are done on validation only (to avoid tuning on the test set);
    # test applies those optimal parameters as-is and is evaluated only once each.
    run_val = 'validation' in splits
    run_test = 'test' in splits
    total_runs = (len(combos) + len(other_paths) if run_val else 0) + \
                 (1 + len(other_paths) if run_test else 0)
    print(f"Total Experiments: {total_runs}  (splits={list(splits)}; "
          f"val={'sweep%d+rest%d' % (len(combos), len(other_paths)) if run_val else 'skip'}, "
          f"test={'allmodels%d' % (1 + len(other_paths)) if run_test else 'skip'})")

    run_idx = 0
    best_short = best_name.replace(cfg.MODEL_PREFIX, '')
    # 1) [validation] full parameter sweep of the best model -> emit total_performance -> other models with 1 optimal combo
    if run_val:
        for s, e, tp in combos:
            run_idx += 1
            run_single_experiment(best_path, best_name, thicknesses, s, e, tp,
                                  'validation', visualize, run_idx, total_runs)
        find_best_model_and_params()  # emit total_performance.csv (=val columns) from the validation sweep results
        bs, be, btp = _best_param_combo(best_short)
        print(f"\nBest model ({BEST_MODEL}) optimal parameters (validation): stride={bs}, extend={be}, turn={btp}")
        for p in other_paths:
            run_idx += 1
            run_single_experiment(p, os.path.basename(p), thicknesses, bs, be, btp,
                                  'validation', visualize, run_idx, total_runs)
        find_best_model_and_params()  # rewrite total_performance.csv including the other models

    # 2) [test] read the validation-optimal parameters from total_performance.csv and evaluate all models once each.
    #    (test applies the val-optimal values without a sweep -> avoids tuning on the test set)
    if run_test:
        try:
            bs, be, btp = _best_param_combo(best_short)
        except (FileNotFoundError, ValueError, KeyError):
            print("Error: the validation total_performance.csv (AP20(val)) required for test evaluation is missing. "
                  "Run validation first with '--split validation' (or without specifying a split).")
            return
        print(f"\ntest evaluation parameters (validation-optimal): stride={bs}, extend={be}, turn={btp}")
        for p in [best_path] + other_paths:
            run_idx += 1
            run_single_experiment(p, os.path.basename(p), thicknesses, bs, be, btp,
                                  'test', visualize, run_idx, total_runs)
        find_best_model_and_params()  # add test columns to total_performance.csv

    print("\nAll experiments completed.")


def _best_param_combo(model_name_short):
    """(stride, extend, turn) of the row with the highest AP20(val) for the given model in total_performance.csv."""
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
    detector.turn_penalty = tp  # override the class attribute (not a constructor arg) on the instance
    # param_name is shown in the banner above, so omit it from desc -> more room for the progress bar
    desc = f"[combo {run_idx}/{total_runs}] {model_name}[{split}]"
    detector.detect_lines(desc=desc)
    evaluate_all(cfg.coco_anno_path(split), cfg.label_dir(split), model_path, result_path, split)


def eval_only(splits=None):
    """Skip detection and re-run only the evaluation using existing prediction JSON (coco_pred_instances_*.json).

    When only the GT (merged_annotations_{split}.json) changed, mIoU depends solely on the label PNG
    and prediction JSON, so it does not change and only AP changes. Therefore recompute only AP and
    reuse mIoU/instances as-is from the existing eval_result.csv (fall back to full evaluation if no
    existing CSV). Both val and test are processed. The selected_annotation cache is automatically
    invalidated by the evaluator when the GT is newer."""
    model_by_name = {os.path.basename(p): p for p in find_model_paths(cfg.DATA_ROOT)}

    for split in (splits or cfg.EVAL_SPLITS):
        label = cfg.split_label(split)
        coco_gt_json = cfg.coco_anno_path(split)
        label_path = cfg.label_dir(split)
        pred_files = glob.glob(os.path.join(cfg.RESULT_PATH, '*', '*',
                                            f'coco_pred_{label}_*.json'))
        result_dirs = sorted({os.path.dirname(f) for f in pred_files})
        print(f"[{split}] Re-evaluating AP for {len(result_dirs)} result dirs "
              f"(skip detection/mIoU, GT={coco_gt_json})")

        for result_path in result_dirs:
            # path structure: RESULT_PATH / [model_name] / [param_name]
            model_name = os.path.basename(os.path.dirname(result_path))
            print(f"\n{'='*80}\nEvaluating AP[{split}]: {model_name} / {os.path.basename(result_path)}\n{'='*80}")
            if _reeval_ap_only(coco_gt_json, result_path, split):
                continue
            # fall back to full evaluation including mIoU if there is no existing eval_result.csv
            model_path = model_by_name.get(model_name)
            if model_path is None:
                print(f"  skip {result_path}: no existing CSV + model '{model_name}' directory not found either")
                continue
            evaluate_all(coco_gt_json, label_path, model_path, result_path, split)

    print("\nEval-only(AP) completed.")


def _reeval_ap_only(coco_gt_json, result_path, split):
    """Keep mIoU/instances in the existing eval_result.csv and recompute/overwrite only this split's AP columns with the new GT.

    mIoU depends on the label PNG and prediction JSON rather than the GT, so it does not change when the GT changes.
    Returns False if this split's AP columns are not in the CSV (caller falls back to full evaluation)."""
    csv_path = os.path.join(result_path, 'eval_result.csv')
    if not os.path.exists(csv_path):
        return False
    label = cfg.split_label(split)
    df = pd.read_csv(csv_path)
    ap_cols = {m: cfg.mcol(m, split) for m in ('AP10', 'AP20', 'AP50')}  # {'AP10':'AP10(val)',...}
    if not any(c in df.columns for c in ap_cols.values()):
        return False  # this split's columns don't exist yet -> full evaluation
    for json_file in sorted(glob.glob(os.path.join(result_path, f'coco_pred_{label}_*.json'))):
        mc = _filename_to_merge_count(json_file)
        ap = evaluate_coco_ap(coco_gt_json, json_file)  # {instances, AP10, AP20, AP50}
        mask = df['merge_count'] == mc
        if not mask.any():
            continue
        for m, col in ap_cols.items():  # update AP only; keep existing mIoU/instances values
            if col in df.columns and m in ap:
                df.loc[mask, col] = float(ap[m])
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(df.to_string(index=False))
    return True


def find_model_paths(data_root):
    """Collect directories that contain a model prediction folder (pred_val) as model paths."""
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
    """Gather eval_result.csv (with val/test columns) from all param folders and rewrite total_performance.csv.
    After a validation run only the val columns are filled; after a test run the test columns are filled too."""
    print("\n" + "="*80)
    print("Finding the best model and parameter combination...")
    # path structure: RESULT_PATH / [model_name] / [param_name] / eval_result.csv
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
    """Column order: metadata -> metrics (split=val->test, in instances/AP10/AP20/AP50/mIoU order). Round and cast types."""
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
    """Attach model/parameter metadata to eval_result.csv (merge_count + ...(val)/...(test) columns)."""
    df = pd.read_csv(file_path)
    # path structure: RESULT_PATH / [model_name] / [param_name] / eval_result.csv
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
    parser = argparse.ArgumentParser(description='Run hyperparameter search and performance evaluation')
    parser.add_argument('--fast', action='store_true',
                        help='Skip window display and visualization collage; run only the fast performance evaluation')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip detection and re-run only the evaluation using existing prediction JSON (when GT changed)')
    parser.add_argument('--split', nargs='+', choices=['validation', 'test'],
                        default=list(cfg.EVAL_SPLITS),
                        help='Select splits to evaluate (default: both validation and test). '
                             'e.g. --split test (test only), --split validation test (both)')
    args = parser.parse_args()
    splits = args.split
    if args.eval_only:
        eval_only(splits)
        find_best_model_and_params()
    else:
        # run_experiments updates total_performance.csv at each split stage.
        run_experiments(splits, visualize=not args.fast)


if __name__ == '__main__':
    main()
