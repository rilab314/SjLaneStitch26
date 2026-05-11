import os
import pandas as pd
import sys

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
        fallback_path = os.path.join(cfg.RESULT_PATH, 'coco_pred_instances_origin.json')
        print(f"Warning: {csv_path} not found. Fallback to {fallback_path}")
        return None, None, fallback_path
        
    df = pd.read_csv(csv_path)
    best_row = df.sort_values('AP20', ascending=False, na_position='last').iloc[0]
    model_name = best_row['model_name']
    merge_count = int(best_row['merge_count'])
    t = int(best_row['thicknesses'])
    s = int(best_row['sample_strides'])
    e = int(best_row['extend_lens'])
    
    print(f"Best target: model={model_name}, merge_count={merge_count}, AP20={best_row['AP20']:.6f}")
    
    model_dir = cfg.MODEL_PREFIX + model_name
    param_dir = f"thick={t},stride={s},extend={e}"
    filename = "origin" if merge_count == 0 else f"merge{merge_count}"
    pred_json_path = os.path.join(cfg.RESULT_PATH, model_dir, param_dir, f"coco_pred_instances_{filename}.json")
    
    return model_name, merge_count, pred_json_path
