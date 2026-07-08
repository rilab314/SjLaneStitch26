"""Figure 8 — per-class failure cases (per-class, original|GT|segmentation|TP/FP/FN 1x4, original resolution).

Matches the target classes' GT and predictions at IoU 0.2 to show failure signatures. 4 case types:
  edge_line (adjacent fusion + near-miss) / no_parking (under-detection) / bus_only (false detection) / bicycle (sparse, loose).
(a) original | (b) original + GT linestring (class render color, endpoint dots) | (c) original + segmentation map (opaque)
| (d) TP (green)/FP (red) = prediction lines, FN (blue) = missed GT lines.
Inspects all 4 types in each frame and saves each matching case to the class subfolder.
Sparse classes (bus_only, bicycle) have relaxed criteria. Matching/rendering shared via figure_match.
"""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
import figure_render as fr
import figure_match as fmatch
from figure_base import FigureGenerator


def cond_edge(s):        # adjacent fusion + near-miss
    return s["merge"] >= 1 and s["near"] >= 1


def cond_no_parking(s):  # long-line under-detection (missed detection)
    return s["miss"] >= 1 or (s["recall"] <= 0.5 and s["near"] >= 1)


def cond_bus_only(s):    # false detection (FP) — heavily relaxed: frames where bus_only appears in GT/prediction
    return s["nG"] >= 1 or s["nP"] >= 1


def cond_bicycle(s):     # sparse, loose — heavily relaxed: frames where bicycle appears in GT/prediction
    return s["nG"] >= 1 or s["nP"] >= 1


class FailureCaseFigure(FigureGenerator):
    """Shows the target classes' IoU-matching failure patterns across 4 panels: original|GT|segmentation|TP/FP/FN."""

    name = "Figure_8"
    MAX_PER_CLASS = 100   # per-class save cap (stop saving once exceeded)
    CASES = [
        {"class_id": 5, "name": "edge_line", "condition": cond_edge},
        {"class_id": 7, "name": "no_parking_stopping_line", "condition": cond_no_parking},
        {"class_id": 4, "name": "bus_only_lane", "condition": cond_bus_only},
        {"class_id": 11, "name": "bicycle_lane", "condition": cond_bicycle},
    ]

    def __init__(self):
        super().__init__()
        self._saved = {case["name"]: 0 for case in self.CASES}

    def save_if_match(self, path):
        if all(n >= self.MAX_PER_CLASS for n in self._saved.values()):
            return 0
        image_id = os.path.basename(path)[:-4]
        stage = self._detector.stage_linestrings(
            path, do_merge=True, merge_iters=self._detector.num_merges)
        final = self.final_merge(stage)
        image, pred_img, (h, w) = stage["image"], stage["pred_img"], stage["img_shape"]
        gt = self.gt_annotations(image_id)
        saved = 0
        for case in self.CASES:
            if self._saved[case["name"]] >= self.MAX_PER_CLASS:
                continue
            match = fmatch.match_class(self._detector, final, gt, case["class_id"], h, w)
            if match is None or not case["condition"](match["signals"]):
                continue
            self._save_case(image, pred_img, match, case, image_id)
            self._saved[case["name"]] += 1
            saved += 1
        return saved

    def _save_case(self, image, pred_img, match, case, image_id):
        out_dir = os.path.join(self._out_dir, case["name"])
        os.makedirs(out_dir, exist_ok=True)
        panel = self._build_panel(image, pred_img, match, case["class_id"])
        cv2.imwrite(os.path.join(out_dir, f"{image_id}.png"), panel)

    def _build_panel(self, image, pred_img, match, class_id):
        """Combines original | GT (render color) | segmentation (opaque) | TP green/FP red/FN blue at original resolution."""
        gt_strand_lists = [fmatch.mask_strands(self._detector, m, class_id) for m in match["gt_masks"]]
        gt_panel = image.copy()
        render = cfg.RENDER_ID2BGR.get(class_id, (255, 255, 255))
        for strands in gt_strand_lists:
            fmatch.draw(gt_panel, strands, render)
        seg = fr.overlay_segmentation(image, pred_img, cfg.EXCLUDE_IDS, alpha=1.0)
        match_panel = image.copy()
        for j, strands in enumerate(gt_strand_lists):
            if j not in match["matched_gt"]:
                fmatch.draw(match_panel, strands, fmatch.FN)
        for i, strand in enumerate(match["pred_strands"]):
            fr.draw_strand(match_panel, strand.points,
                           fmatch.TP if i in match["matched_pred"] else fmatch.FP,
                           thickness=3, draw_dots=True)
        return fr.concat_horizontal([image.copy(), gt_panel, seg, match_panel])


if __name__ == "__main__":
    FailureCaseFigure().run()
