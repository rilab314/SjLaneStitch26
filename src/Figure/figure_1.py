import sys
import os
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import src.config as cfg


class FinalFigureGenerator:
    """Generates the final figure comparing original images and predictions."""
    
    MARGIN = 20

    def __init__(self, img_ids):
        self.img_ids = img_ids
        self.save_path = os.path.join(cfg.RESULT_PATH, 'Figure', 'figure1.jpg')

    def generate(self):
        """Main method to generate and save the final figure."""
        rows = self._process_image_pairs()
        if not rows:
            print("No images were successfully loaded.")
            return

        final_img_width = rows[0].shape[1]
        final_img = self._stack_images_vertically(rows, final_img_width)
        
        self._save_image(final_img)

    def _process_image_pairs(self):
        """Processes each image ID to create side-by-side comparison rows."""
        rows = []
        for img_id in self.img_ids:
            row_img = self._load_and_combine_pair(img_id)
            if row_img is not None:
                rows.append(row_img)
        return rows

    def _load_and_combine_pair(self, img_id):
        """Loads and combines a single original and result image pair."""
        orig_path = os.path.join(cfg.DATASET_PATH, 'images', 'validation', f"{img_id}.png")
        res_path = os.path.join(cfg.RESULT_PATH, 'Figure', 'Figure_1_raw', f"{img_id}.png")
        
        orig_img = cv2.imread(orig_path)
        res_img = cv2.imread(res_path)
        
        if orig_img is None or res_img is None:
            print(f"Warning: Could not load images for {img_id}")
            return None
            
        return self._combine_side_by_side(orig_img, res_img)

    def _combine_side_by_side(self, orig_img, res_img):
        """Combines two images side-by-side with a margin."""
        h1, w1 = orig_img.shape[:2]
        h2, w2 = res_img.shape[:2]
        target_h = max(h1, h2)
        
        if h1 != target_h or h2 != target_h:
            orig_img = cv2.resize(orig_img, (int(w1 * target_h / h1), target_h))
            res_img = cv2.resize(res_img, (int(w2 * target_h / h2), target_h))
            
        space = np.ones((target_h, self.MARGIN, 3), dtype=np.uint8) * 255
        return np.hstack((orig_img, space, res_img))

    def _stack_images_vertically(self, rows, width):
        """Stacks multiple image rows vertically with margins."""
        stacked_rows = []
        for i, row in enumerate(rows):
            stacked_rows.append(row)
            if i < len(rows) - 1:
                space_row = np.ones((self.MARGIN, width, 3), dtype=np.uint8) * 255
                stacked_rows.append(space_row)
        return np.vstack(stacked_rows)

    def _save_image(self, img):
        """Saves the final combined image."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        cv2.imwrite(self.save_path, img)
        print(f"Saved figure 1 to {self.save_path}")


def generate_final_figure(img_ids):
    """Wrapper function to maintain original interface."""
    generator = FinalFigureGenerator(img_ids)
    generator.generate()


if __name__ == "__main__":
    generate_final_figure(["126.716,37.401", "126.716,37.4857"])
