"""Figure 7 — original vs GT vs segmentation vs final vs TP/FP/FN (headline qualitative, 1x5 horizontal collage).

Panels: (a) original | (b) original + GT linestring (endpoint dots) | (c) original + segmentation mask (opaque)
      | (d) original + final vector linestring (endpoint dots) | (e) original + TP (green)/FP (red)/FN (blue) matching.
Measures per-frame AP20 and mIoU (computed only over valid classes where objects exist), splits into the 4 groups
below, and saves per folder.
Filename {coord}_{AP20}_{mIoU}.png (metrics are % x 10 as integers, e.g. AP42.3/mIoU24.5 -> _423_245).
Groups are matched in listed order (saved to the first matching group):
  HAP_HIoU: AP20>60 AND mIoU>55 | HAP_LIoU: AP20 > mIoU+25
  LAP_HIoU: AP20 < mIoU-3       | LAP_LIoU: AP20<30 AND mIoU<30
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
import figure_render as fr
import figure_metrics as fm
import figure_match as fmatch
from figure_base import FigureGenerator
from evaluator import to_label_index_image, json_to_label_image


class SegVsMergeFigure(FigureGenerator):
    """Splits into 4 groups (high/low AP x high/low IoU) by frame AP20/mIoU combination and saves them."""

    name = "Figure_7"

    def __init__(self):
        super().__init__()
        self.label_dir = cfg.label_dir("validation")

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
        """Determines the group from the AP20/mIoU (%) combination (matched in listed order, None if none apply)."""
        if ap > 60 and miou > 55:
            return "HAP_HIoU"
        if ap > miou + 25:
            return "HAP_LIoU"
        if ap < miou - 3:
            return "LAP_HIoU"
        if ap < 30 and miou < 30:
            return "LAP_LIoU"
        return None

    def frame_miou(self, image_id, pred_anns):
        """Frame mIoU (averaged only over valid classes where objects exist). Same method as evaluate_miou_json."""
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
        """Combines the five panels original | GT | segmentation (opaque) | final vector | TP/FP/FN horizontally with black gaps."""
        image = stage["image"]
        gt = fr.draw_strands(image.copy(), self._gt_strands(stage, image_id), dots=True)
        seg = fr.overlay_segmentation(image, stage["pred_img"], cfg.EXCLUDE_IDS, alpha=1.0)
        pred = fr.draw_strands(image.copy(), final, dots=True)
        tpfpfn = fmatch.tpfpfn_panel(self._detector, image, final, self.gt_annotations(image_id))
        return fr.concat_horizontal([image.copy(), gt, seg, pred, tpfpfn])

    def _gt_strands(self, stage, image_id):
        """Vectorizes the GT color annotation image the same way as predictions to obtain GT linestrings."""
        gt_path = os.path.join(cfg.color_label_dir("validation"), f"{image_id}.png")
        gt_img = cv2.imread(gt_path)
        if gt_img is None:
            return []
        self._detector._img_shape = stage["img_shape"]
        self._detector._id_count = self._detector.id_offset
        strands, _ = self._detector.extract_lines(gt_img, image_id)
        return strands


if __name__ == "__main__":
    SegVsMergeFigure().run()
