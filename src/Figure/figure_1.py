import sys
import cv2
import os
from tqdm import tqdm

# ---------------------------------------------------------
# 프로젝트 경로 설정 및 config 임포트
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import src.config as cfg
from src.util import (
    find_best_pred_json_path,
    load_json,
    group_annotations_by_image,
    draw_annotations_on_image
)


class FigureGenerator:
    """Generates figures by drawing segmentation masks on validation images."""

    def __init__(self):
        """Initializes paths for images, annotations, and results."""
        self.img_dir = os.path.join(cfg.DATASET_PATH, 'images', 'validation')
        self.csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
        _, _, self.json_path = find_best_pred_json_path(self.csv_path)
        self.result_dir = os.path.join(cfg.RESULT_PATH, 'Figure', 'Figure_1')
        os.makedirs(self.result_dir, exist_ok=True)


    def generate(self):
        """Main entry point to generate all figures."""
        annotations = load_json(self.json_path)
        if not annotations:
            return

        image_groups = group_annotations_by_image(annotations)
        pbar = tqdm(image_groups.items(), desc="Generating figures")
        for img_id, annotations in pbar:
            img_filename = f"{img_id}.png"
            pbar.set_description(f"Processing {img_filename}")
            img_path = os.path.join(self.img_dir, img_filename)
            result_img = cv2.imread(img_path)
            if result_img is None:
                print(f"Warning: Image not found at {img_path}")
                continue

            result_img = draw_annotations_on_image(result_img, annotations, cfg.EXCLUDE_IDS)
            self._save_image(img_filename, result_img)

    def _save_image(self, filename, img):
        """Saves the resulting image to the result directory."""
        save_path = os.path.join(self.result_dir, filename)
        cv2.imwrite(save_path, img)


def main():
    generator = FigureGenerator()
    generator.generate()


if __name__ == "__main__":
    main()
