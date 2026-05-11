import sys
import cv2
import json
import os
import numpy as np
import pandas as pd
from pycocotools import mask as maskUtils
from tqdm import tqdm

# ---------------------------------------------------------
# 프로젝트 경로 설정 및 config 임포트
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import src.config as cfg
from src.util import find_best_pred_json_path


class FigureGenerator:
    """Generates figures by drawing segmentation masks on validation images."""

    def __init__(self):
        """Initializes paths for images, annotations, and results."""
        self.img_dir = os.path.join(cfg.DATASET_PATH, 'images', 'validation')
        self.csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
        _, _, self.json_path = find_best_pred_json_path(self.csv_path)
        self.result_dir = os.path.join(cfg.RESULT_PATH, 'Figure', 'Figure_1_raw')
        os.makedirs(self.result_dir, exist_ok=True)


    def generate(self):
        """Main entry point to generate all figures."""
        annotations = self._load_annotations()
        if not annotations:
            return

        image_groups = self._group_by_image(annotations)
        pbar = tqdm(image_groups.items(), desc="Generating figures")
        for img_id, annotations in pbar:
            img_filename = f"{img_id}.png"
            pbar.set_description(f"Processing {img_filename}")
            img_path = os.path.join(self.img_dir, img_filename)
            result_img = cv2.imread(img_path)
            if result_img is None:
                print(f"Warning: Image not found at {img_path}")
                continue

            result_img = self._draw_annotations(result_img, annotations)
            self._save_image(img_filename, result_img)

    def _load_annotations(self):
        """Loads annotations from JSON file."""
        if not os.path.exists(self.json_path):
            print(f"Error: JSON file not found at {self.json_path}")
            return []

        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _group_by_image(self, annotations):
        """Groups annotations by image_id."""
        image_groups = {}
        for entry in annotations:
            img_id = entry['image_id']
            image_groups.setdefault(img_id, []).append(entry)
        return image_groups

    def _draw_annotations(self, img, annotations):
        """Draws a list of annotations on the image."""
        for ann in annotations:
            category_id = ann.get('category_id', 0)
            if category_id in cfg.EXCLUDE_IDS:
                continue

            seg = ann.get('segmentation')
            color = cfg.ID2BGR.get(category_id, (0, 0, 0))
            self._draw_mask(img, seg, color)
        return img

    def _draw_mask(self, img, seg, color):
        """Decodes and draws a single mask on the image."""
        if not (isinstance(seg, dict) and 'counts' in seg):
            return

        try:
            binary_mask = maskUtils.decode(seg)
            if binary_mask.ndim == 3:
                binary_mask = binary_mask[:, :, 0]

            img[binary_mask > 0] = color
        except Exception as e:
            print(f"Error decoding mask: {e}")

    def _save_image(self, filename, img):
        """Saves the resulting image to the result directory."""
        save_path = os.path.join(self.result_dir, filename)
        cv2.imwrite(save_path, img)
        # print(f"Saved: {save_path}")


def main():
    generator = FigureGenerator()
    generator.generate()
    # generate_final_figure(["126.716,37.401", "126.716,37.4857"])


if __name__ == "__main__":
    main()
