"""Common skeleton for figure generation (FigureGenerator base class).

Configures LaneStitcher with the best combination (stitch_config), iterates over validation
frames, and saves only the frames that meet the conditions. Subclasses only define `name` and
`build_figure`.
"""
import os
import sys
import glob

import cv2
from tqdm import tqdm

import config as cfg
from lane_stitcher import LaneStitcher
from util import load_json, group_annotations_by_image
from stitch_config import load_stitch_config


class FigureGenerator:
    """Base that iterates over validation frames and saves only the figures meeting the conditions.

    build_figure returns (image, filename suffix) or None (condition not met).
    """

    name = "Figure"
    gap = 20

    def __init__(self):
        self._config = load_stitch_config()
        self._detector = self.build_detector()
        self._out_dir = os.path.join(cfg.RESULT_PATH, "Figure", self.name)
        os.makedirs(self._out_dir, exist_ok=True)
        self._val_files = sorted(glob.glob(
            os.path.join(cfg.image_dir("validation"), "*.png")))
        self._gt_map = None

    def build_detector(self):
        """Configures a LaneStitcher instance with the best parameters."""
        conf = self._config
        detector = LaneStitcher(cfg.DATASET_PATH, conf.model_path, cfg.RESULT_PATH,
                                thickness=conf.thickness, sample_stride=conf.sample_stride,
                                extend_len=conf.extend_len, visualize=False)
        detector.turn_penalty = conf.turn_penalty
        return detector

    def run(self):
        """Iterates over all frames, saves only those meeting the conditions, and reports the result."""
        kept = 0
        for path in tqdm(self.select_files(), desc=self.name):
            kept += int(self.save_if_match(path))
        self.report(kept)

    def select_files(self):
        """If the FIG_LIMIT environment variable is set, use only the first N (for smoke tests)."""
        cap = os.environ.get("FIG_LIMIT")
        return self._val_files[:int(cap)] if cap else self._val_files

    def save_if_match(self, path):
        """Saves the figure and returns True if the condition is met, otherwise False."""
        image_id = os.path.basename(path)[:-4]
        result = self.build_figure(image_id, path)
        if result is None:
            return False
        image, suffix = result
        cv2.imwrite(os.path.join(self._out_dir, f"{image_id}{suffix}.png"), image)
        return True

    def build_figure(self, image_id, path):
        """Returns (image, suffix) or None. Implemented by subclasses."""
        raise NotImplementedError

    def read_prediction(self, image_id):
        """Reads the model's segmentation prediction PNG (None if absent)."""
        return cv2.imread(os.path.join(self._config.pred_dir, f"{image_id}.png"))

    def gt_annotations(self, image_id):
        """List of GT annotations for the given image_id (loaded once on first call)."""
        if self._gt_map is None:
            data = load_json(cfg.COCO_ANNO_PATH)
            self._gt_map = group_annotations_by_image(data["annotations"]) if data else {}
        return self._gt_map.get(image_id, [])

    def final_merge(self, stage):
        """Linestrings at the best merge_count merge stage. Applies short-line removal (_filter_short) -> consistent with evaluation output.

        Unlike detect_lines output, stage_linestrings' merges do not have the short-line filter applied,
        so fragments shorter than min_lane_len (dropped in evaluation) remain in the figure. We filter them the same way here."""
        merges = stage["merges"]
        if not merges:
            return stage["refined"]
        idx = min(self._config.merge_count, len(merges)) - 1
        return self._detector._filter_short(merges[idx])

    def report(self, kept):
        """Prints a summary of the save results."""
        print(f"[{self.name}] {kept}/{len(self._val_files)} frames -> {self._out_dir}")
