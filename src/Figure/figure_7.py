"""Figure 7 — 원본 vs GT vs 분할 vs 최종 결과 비교 (헤드라인 정성, 1×4 가로 콜라주).

패널: (a) 원본 | (b) 원본 + GT linestring(끝점 점) | (c) 원본 + 분할 마스크(불투명)
      | (d) 원본 + 최종 벡터 linestring(끝점 점).
프레임별 AP20·mIoU(객체가 존재하는 유효 클래스로만 계산)를 재고 아래 4그룹으로 나눠 폴더별 저장한다.
파일명 {좌표}_{AP20}_{mIoU}.png (지표는 %×10 정수, 예 AP42.3/mIoU24.5 → _423_245).
그룹은 나열 순서대로 우선 매칭한다(첫 매칭 그룹에 저장):
  HAP_HIoU: AP20>60 AND mIoU>50 | HAP_LIoU: AP20 > mIoU+10
  LAP_HIoU: AP20 < mIoU-10      | LAP_LIoU: AP20<30 AND mIoU<30
"""
import os
import sys

import cv2
import numpy as np

_CUR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_CUR, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as cfg
import figure_render as fr
import figure_metrics as fm
from figure_base import FigureGenerator
from evaluator import to_label_index_image, json_to_label_image


class SegVsMergeFigure(FigureGenerator):
    """프레임 AP20·mIoU 조합으로 4그룹(고/저 AP × 고/저 IoU)으로 나눠 저장한다."""

    name = "Figure_7"

    def __init__(self):
        super().__init__()
        self.label_dir = os.path.join(cfg.DATASET_PATH, "annotations", "validation")

    def save_if_match(self, path):
        image_id = os.path.basename(path)[:-4]
        stage = self._detector.stage_linestrings(
            path, do_merge=True, merge_iters=self._detector.num_merges)
        final = self.final_merge(stage)
        pred_anns = self._detector.convert_to_json(final, image_id)
        ap20 = fm.measure_frame_ap20(self.gt_annotations(image_id), pred_anns, image_id)
        miou = self.frame_miou(image_id, pred_anns)
        if ap20 is None or miou is None:
            return False
        group = self._group(ap20 * 100, miou * 100)
        if group is None:
            return False
        out_dir = os.path.join(self._out_dir, group)
        os.makedirs(out_dir, exist_ok=True)
        name = f"{image_id}_{round(ap20 * 1000)}_{round(miou * 1000)}.png"
        cv2.imwrite(os.path.join(out_dir, name), self.compose(stage, final, image_id))
        return True

    def _group(self, ap, miou):
        """AP20·mIoU(%) 조합으로 4그룹 판정(나열 순서 우선 매칭, 해당 없으면 None)."""
        if ap > 60 and miou > 50:
            return "HAP_HIoU"
        if ap > miou + 10:
            return "HAP_LIoU"
        if ap < miou - 10:
            return "LAP_HIoU"
        if ap < 30 and miou < 30:
            return "LAP_LIoU"
        return None

    def frame_miou(self, image_id, pred_anns):
        """프레임 mIoU(객체가 존재하는 유효 클래스만 평균). evaluate_miou_json과 동일 방식."""
        label_file = os.path.join(self.label_dir, f"{image_id}.png")
        gt_label = to_label_index_image(cv2.imread(label_file, cv2.IMREAD_UNCHANGED), True)
        if gt_label is None:
            return None
        h, w = gt_label.shape
        ann_idx = {image_id: [a for a in pred_anns
                              if int(a.get("category_id", 0)) not in cfg.EXCLUDE_IDS]}
        pred_label = json_to_label_image(ann_idx, h, w, label_file)
        ious = []
        for cid in cfg.EVAL_CLASS_IDS:
            union = int(np.sum((gt_label == cid) | (pred_label == cid)))
            if union:
                ious.append(int(np.sum((gt_label == cid) & (pred_label == cid))) / union)
        return float(np.mean(ious)) if ious else None

    def compose(self, stage, final, image_id):
        """원본 | GT | 분할(불투명) | 최종 벡터 네 패널을 검은 여백으로 가로 결합한다."""
        image = stage["image"]
        gt = fr.draw_strands(image.copy(), self._gt_strands(stage, image_id), dots=True)
        seg = fr.overlay_segmentation(image, stage["pred_img"], cfg.EXCLUDE_IDS, alpha=1.0)
        pred = fr.draw_strands(image.copy(), final, dots=True)
        return fr.concat_horizontal([image.copy(), gt, seg, pred])

    def _gt_strands(self, stage, image_id):
        """GT color 주석 이미지를 예측과 동일하게 벡터화해 GT linestring을 얻는다."""
        gt_path = os.path.join(cfg.DATASET_PATH, "color_annotations", "validation", f"{image_id}.png")
        gt_img = cv2.imread(gt_path)
        if gt_img is None:
            return []
        self._detector._img_shape = stage["img_shape"]
        self._detector._id_count = self._detector.id_offset
        strands, _ = self._detector.extract_lines(gt_img, image_id)
        return strands


if __name__ == "__main__":
    SegVsMergeFigure().run()
