"""Run and evaluate the OpenSatMap baseline (validation split).

Apply the watershed baseline on top of the best model's (Swin-L) same seg predictions to
build coco_pred_instances_baseline.json, and, identically to our Table4Builder._evaluate,
print instances, AP20, and mIoU via evaluate_coco_ap(9 classes, IoU 0.2, score 1) + evaluate_miou_json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
from lane_stitcher import LaneStitcher
from stitch_config import load_stitch_config
from baseline_opensatmap import OpenSatMapBaseline
from evaluator import evaluate_coco_ap, evaluate_miou_json


BASELINE_JSON_NAME = "coco_pred_instances_baseline.json"


def build_baseline_json():
    """Build the stitcher with the best combination to create the baseline prediction JSON and return the save path."""
    sc = load_stitch_config()
    stitcher = LaneStitcher(cfg.DATASET_PATH, sc.model_path, cfg.RESULT_PATH,
                            thickness=sc.thickness, sample_stride=sc.sample_stride,
                            extend_len=sc.extend_len, visualize=False, split='validation')
    baseline = OpenSatMapBaseline(stitcher, sample_stride=sc.sample_stride)
    save_path = os.path.join(cfg.RESULT_PATH, BASELINE_JSON_NAME)
    baseline.run_and_save(save_path)
    return save_path


def evaluate_baseline(pred_json: str):
    """Compute instances, AP20, and mIoU with the same calls as Table4Builder._evaluate."""
    ap = evaluate_coco_ap(cfg.COCO_MERGED_ANNO_PATH, pred_json)
    miou = evaluate_miou_json(pred_json, cfg.LABEL_PATH)
    return {"instances": ap["instances"], "AP20": ap["AP20"], "mIoU": miou["mIoU"]}


def main():
    pred_json = build_baseline_json()
    res = evaluate_baseline(pred_json)
    print("\n===== OpenSatMap baseline (watershed) =====")
    print(f"instances = {res['instances']}")
    print(f"AP20      = {float(res['AP20']) * 100:.2f}")
    print(f"mIoU      = {res['mIoU'] * 100:.2f}")


if __name__ == "__main__":
    main()
