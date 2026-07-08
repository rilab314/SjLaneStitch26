"""Table 4 — best 모델의 단계별 성능 향상, 5줄.

열: Stage | Instances | AP20 | mIoU
누적 단계: 1차 추출 → +잔여 재추출 → +정제 → 병합×1 → 병합×2.
정제·merge1·merge2(3~5단계)는 total_performance.csv에서 그대로 읽고,
total_performance.csv에 없는 1차/잔여(1~2단계)만 stage_linestrings로 새로 평가한다.
"""
import os
import sys
import glob
import json

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from lane_stitcher import LaneStitcher
from stitch_config import load_stitch_config
from evaluator import evaluate_coco_ap, evaluate_miou_json
import table_common as tc


class Table4Builder:
    """단계별(첫추출/잔여/정제/병합1/병합2) AP20·mIoU·인스턴스 수를 5줄로 정리한다."""

    def __init__(self, total_csv_path, save_name):
        self.df = tc.with_val_aliases(pd.read_csv(total_csv_path))
        self.save_name = save_name
        self.combo = tc.best_combo(self.df)

    def build(self):
        fresh = self._evaluate_first_and_combined()
        rows = [
            self._row("First extraction", fresh["first"]),
            self._row("+ Residual re-extraction", fresh["combined"]),
            self._row("+ Refinement", self._csv_stage(0)),
            self._row("+ Merge ×1", self._csv_stage(1)),
            self._row("+ Merge ×2", self._csv_stage(2)),
        ]
        baseline = self._baseline_row()
        if baseline is not None:
            rows.append(baseline)
        tc.save_csv(pd.DataFrame(rows), self.save_name)

    def _baseline_row(self):
        """OpenSatMap baseline 예측(run_baseline.py 산출물)이 있으면 참조 행으로 맨 아래 추가한다.
        없으면 None을 반환해 건너뛴다(누적 ablation 5행과 무관한 외부 baseline 비교 행)."""
        path = os.path.join(cfg.RESULT_PATH, "coco_pred_instances_baseline.json")
        if not os.path.exists(path):
            return None
        return self._row("OpenSatMap baseline (watershed)", self._evaluate(path))

    def _row(self, stage, metrics):
        return {"Stage": stage, "Instances": int(metrics["instances"]),
                "AP20": tc.pct(metrics["AP20"]), "mIoU": tc.pct(metrics["mIoU"])}

    def _csv_stage(self, merge_count):
        """total_performance.csv에서 best 조합·merge_count 행의 지표를 읽는다."""
        c = self.combo
        mask = ((self.df["model_name"] == c["model_name"]) &
                (self.df["thicknesses"] == c["thicknesses"]) &
                (self.df["sample_strides"] == c["sample_strides"]) &
                (self.df["extend_lens"] == c["extend_lens"]) &
                (self.df["turn_penalties"] == c["turn_penalties"]) &
                (self.df["merge_count"] == merge_count))
        row = self.df[mask].iloc[0]
        return {"instances": row["instances"], "AP20": row["AP20"], "mIoU": row["mIoU"]}

    def _evaluate_first_and_combined(self):
        """stage_linestrings로 1차/잔여 단계 예측을 만들어 새로 평가한다."""
        first_json, combined_json = self._build_stage_predictions()
        return {"first": self._evaluate(first_json),
                "combined": self._evaluate(combined_json)}

    def _build_stage_predictions(self):
        det = self._build_detector()
        files = sorted(glob.glob(os.path.join(cfg.DATASET_PATH, "images", "validation", "*.png")))
        first_preds, combined_preds = [], []
        for f in tqdm(files, desc="1차/잔여 단계 추출"):
            image_id = os.path.basename(f)[:-4]
            stage = det.stage_linestrings(f, do_merge=False)
            first_preds += det.convert_to_json(det._smoothed_copies(stage["first"]), image_id)
            combined_preds += det.convert_to_json(det._smoothed_copies(stage["combined"]), image_id)
        return (self._dump(first_preds, "_stage_first.json"),
                self._dump(combined_preds, "_stage_combined.json"))

    def _build_detector(self):
        sc = load_stitch_config()
        det = LaneStitcher(cfg.DATASET_PATH, sc.model_path, cfg.RESULT_PATH,
                           thickness=sc.thickness, sample_stride=sc.sample_stride,
                           extend_len=sc.extend_len, visualize=False)
        det.turn_penalty = sc.turn_penalty
        return det

    def _dump(self, preds, name):
        os.makedirs(tc.tables_dir(), exist_ok=True)
        path = os.path.join(tc.tables_dir(), name)
        with open(path, "w") as fp:
            json.dump(preds, fp)
        return path

    def _evaluate(self, pred_json):
        ap = evaluate_coco_ap(cfg.COCO_MERGED_ANNO_PATH, pred_json)
        # total_performance의 csv 단계 mIoU와 같은 기준(기존 ade20k validation 라벨)으로 맞춘다.
        # (cfg.LABEL_PATH는 새 SEED 라벨을 가리키므로 단계 간 mIoU 기준이 섞이는 것을 방지)
        label_dir = os.path.join(cfg.DATASET_PATH, "annotations", "validation")
        miou = evaluate_miou_json(pred_json, label_dir)
        return {"instances": ap["instances"], "AP20": ap["AP20"], "mIoU": miou["mIoU"]}


def main():
    Table4Builder(tc.total_csv_path(), "table_4.csv").build()


if __name__ == "__main__":
    main()
