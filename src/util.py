import os
import pandas as pd
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import config as cfg

def find_best_pred_json_path(csv_path):
    """
    Reads the given csv_path (e.g. total_performance.csv or table_1.csv) 
    and returns the model_name, merge_count, and pred_json_path of the row with the highest AP20.
    """
    if not os.path.exists(csv_path):
        fallback_path = os.path.join(cfg.RESULT_PATH, 'coco_pred_val_origin.json')
        print(f"Warning: {csv_path} not found. Fallback to {fallback_path}")
        return None, None, fallback_path

    df = pd.read_csv(csv_path)
    # 새 포맷은 지표 열에 split 접미사가 있다(AP20(val)). 구 포맷(AP20)도 폴백 지원.
    ap = cfg.mcol('AP20', 'validation') if cfg.mcol('AP20', 'validation') in df.columns else 'AP20'
    best_row = df.sort_values(ap, ascending=False, na_position='last').iloc[0]
    model_name = best_row['model_name']
    merge_count = int(best_row['merge_count'])
    t = int(best_row['thicknesses'])
    s = int(best_row['sample_strides'])
    e = int(best_row['extend_lens'])
    tp = int(best_row['turn_penalties'])

    print(f"Best target: model={model_name}, merge_count={merge_count}, {ap}={best_row[ap]:.6f}")

    model_dir = cfg.MODEL_PREFIX + model_name
    param_dir = f"thick={t},stride={s},extend={e},turn={tp}"
    filename = "origin" if merge_count == 0 else f"merge{merge_count}"
    pred_json_path = os.path.join(cfg.RESULT_PATH, model_dir, param_dir, f"coco_pred_val_{filename}.json")

    return model_name, merge_count, pred_json_path


def find_model_path(model_name):
    """모델 이름으로부터 segmentation 예측 결과가 저장된 모델 디렉토리 경로를 반환한다."""
    model_dir = cfg.MODEL_PREFIX + model_name
    model_type = "Internimage" if "internimage" in model_name.lower() else "mask2former"
    return os.path.join(cfg.DATA_ROOT, model_type, model_dir)


def load_json(path):
    """Load JSON file and return data."""
    if not os.path.exists(path):
        print(f"Warning: File not found - {path}")
        return []
    with open(path, 'r') as f:
        return json.load(f)


def group_annotations_by_image(annotations):
    """Group list of annotations into a dictionary keyed by image_id."""
    mapping = {}
    for ann in annotations:
        img_id = ann['image_id']
        mapping.setdefault(img_id, []).append(ann)
    return mapping
