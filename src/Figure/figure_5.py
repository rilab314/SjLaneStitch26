import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Setup project root and import config
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


def create_horizontal_collage(img_orig, img_gt, img_pred, gap=20):
    """Creates a 1x3 horizontal collage with solid white margins in between."""
    h, w, _ = img_orig.shape
    collage_w = w * 3 + gap * 2
    collage = np.full((h, collage_w, 3), 255, dtype=np.uint8)

    collage[:, 0:w] = img_orig
    collage[:, w+gap : w*2+gap] = img_gt
    collage[:, w*2+gap*2 : w*3+gap*2] = img_pred

    return collage


def main():
    """Main execution flow for generating Figure 5 side-by-side collages."""
    csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
    model_name, _, best_pred_path = find_best_pred_json_path(csv_path)

    if model_name is None:
        model_name = "internimage_large"
        result_subdir = os.path.join(cfg.RESULT_PATH, "satellite_ade20k_250925_" + model_name, "thick=3,stride=10,extend=20")
    else:
        result_subdir = os.path.dirname(best_pred_path)

    json_merge2_path = os.path.join(result_subdir, "coco_pred_instances_merge2.json")

    paths = {
        'json_gt': cfg.COCO_ANNO_PATH,
        'json_merge2': json_merge2_path,
        'original_img_dir': os.path.join(cfg.DATASET_PATH, 'images', 'validation'),
        'output_dir': os.path.join(cfg.RESULT_PATH, 'Figure', 'Figure_5')
    }

    print("Initializing Figure 5 generation...")
    os.makedirs(paths['output_dir'], exist_ok=True)

    # Load all necessary data
    print("Loading Ground Truth JSON...")
    gt_data = load_json(paths['json_gt'])
    if not gt_data or 'annotations' not in gt_data:
        print("Error: Ground Truth annotations could not be loaded or structured invalidly.")
        return

    print("Loading Merge2 Prediction JSON...")
    merge2_anns = load_json(paths['json_merge2'])

    # Group annotations by image_id
    gt_map = group_annotations_by_image(gt_data['annotations'])
    merge2_map = group_annotations_by_image(merge2_anns)

    # Process each image in the dataset
    for img_info in tqdm(gt_data['images'], desc="Generating Figure 5 collages"):
        img_id = img_info['id']
        file_name = img_info['file_name']

        img_path = os.path.join(paths['original_img_dir'], file_name)
        base_img = cv2.imread(img_path)
        if base_img is None:
            continue

        # 1. Original Image (already loaded as base_img)

        # 2. Original Image + GT overlay
        img_gt = draw_annotations_on_image(base_img.copy(), gt_map.get(img_id, []), cfg.EXCLUDE_IDS)

        # 3. Original Image + Prediction (merge2) overlay
        img_pred = draw_annotations_on_image(base_img.copy(), merge2_map.get(img_id, []), cfg.EXCLUDE_IDS)

        # Combine into a horizontal collage with a 20px white gap
        collage = create_horizontal_collage(base_img, img_gt, img_pred, gap=20)

        # Save result
        save_path = os.path.join(paths['output_dir'], file_name)
        cv2.imwrite(save_path, collage)

    print(f"\nSuccess! Figure 5 collages saved to {paths['output_dir']}")


if __name__ == "__main__":
    main()