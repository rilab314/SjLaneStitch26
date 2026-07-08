"""OpenSatMap baseline 실행·평가 (validation split).

best 모델(Swin-L)의 동일 seg 예측 위에 watershed baseline을 적용해
coco_pred_instances_baseline.json을 만들고, 우리 Table4Builder._evaluate와 동일하게
evaluate_coco_ap(9클래스, IoU 0.2, score 1) + evaluate_miou_json으로 instances·AP20·mIoU를 출력한다.
"""
import os

import config as cfg
from lane_stitcher import LaneStitcher
from stitch_config import load_stitch_config
from baseline_opensatmap import OpenSatMapBaseline
from evaluator import evaluate_coco_ap, evaluate_miou_json


BASELINE_JSON_NAME = "coco_pred_instances_baseline.json"


def build_baseline_json():
    """best 조합으로 stitcher를 구성해 baseline 예측 JSON을 만들고 저장 경로를 반환한다."""
    sc = load_stitch_config()
    stitcher = LaneStitcher(cfg.DATASET_PATH, sc.model_path, cfg.RESULT_PATH,
                            thickness=sc.thickness, sample_stride=sc.sample_stride,
                            extend_len=sc.extend_len, visualize=False, split='validation')
    baseline = OpenSatMapBaseline(stitcher, sample_stride=sc.sample_stride)
    save_path = os.path.join(cfg.RESULT_PATH, BASELINE_JSON_NAME)
    baseline.run_and_save(save_path)
    return save_path


def evaluate_baseline(pred_json: str):
    """Table4Builder._evaluate와 동일 호출로 instances·AP20·mIoU를 계산한다."""
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
