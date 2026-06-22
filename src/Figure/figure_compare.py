import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import src.config as cfg
from src.util import (
    load_json,
    group_annotations_by_image,
    draw_annotations_on_image
)

MODEL_NAMES = ["internimage_large", "mask2former_large", "mask2former_small"]


def main():
    """Generate 2x2 collages: GT overlay + segmentation overlays of three models."""
    paths = {
        'json_gt': cfg.COCO_ANNO_PATH,
        'original_img_dir': os.path.join(cfg.DATASET_PATH, 'images', 'validation'),
        'pred_dirs': [build_pred_dir(name) for name in MODEL_NAMES],
        'output_dir': os.path.join(cfg.RESULT_PATH, 'Figure', 'Figure_compare')
    }

    print("Initializing Figure_compare generation...")
    os.makedirs(paths['output_dir'], exist_ok=True)

    gt_data = load_json(paths['json_gt'])
    gt_map = group_annotations_by_image(gt_data['annotations'])

    for img_info in tqdm(gt_data['images'], desc="Generating Figure_compare collages"):
        file_name = img_info['file_name']
        base_img = cv2.imread(os.path.join(paths['original_img_dir'], file_name))
        if base_img is None:
            continue

        img_gt = draw_annotations_on_image(
            base_img.copy(), gt_map.get(img_info['id'], []), cfg.EXCLUDE_IDS)
        seg_overlays = [generate_seg_overlay(base_img, pred_dir, file_name)
                        for pred_dir in paths['pred_dirs']]
        if any(img is None for img in seg_overlays):
            continue

        collage = create_2x2_collage([img_gt] + seg_overlays, gap=20)
        save_path = os.path.join(paths['output_dir'], f"{os.path.splitext(file_name)[0]}.png")
        cv2.imwrite(save_path, collage)

    print(f"Done! Figure_compare collages are saved in: {paths['output_dir']}")


def build_pred_dir(model_name):
    """Resolve the prediction directory for a given model name."""
    model_type = "Internimage" if "internimage" in model_name.lower() else "mask2former"
    return os.path.join(cfg.DATA_ROOT, model_type, cfg.MODEL_PREFIX + model_name, 'prediction')


def generate_seg_overlay(base_img, pred_dir, file_name):
    """Overlay non-excluded segmentation classes on the original image with render colors."""
    pred_path = os.path.join(pred_dir, os.path.splitext(file_name)[0] + ".png")
    pred_img = cv2.imread(pred_path)
    if pred_img is None:
        return None

    overlay = base_img.copy()
    for cat_id, orig_color in cfg.ID2BGR.items():
        if cat_id in cfg.EXCLUDE_IDS:
            continue
        mask = np.all(pred_img == orig_color, axis=-1)
        overlay[mask] = cfg.RENDER_ID2BGR.get(cat_id, orig_color)
    return overlay


def create_2x2_collage(imgs, gap=20):
    """Combine four images into a 2x2 grid with white gaps."""
    h, w, _ = imgs[0].shape
    collage = np.full((h * 2 + gap, w * 2 + gap, 3), 255, dtype=np.uint8)

    collage[0:h, 0:w] = imgs[0]
    collage[0:h, w+gap:] = imgs[1]
    collage[h+gap:, 0:w] = imgs[2]
    collage[h+gap:, w+gap:] = imgs[3]
    return collage


if __name__ == "__main__":
    main()
