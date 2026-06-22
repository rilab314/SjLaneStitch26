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
    find_best_pred_json_path,
    load_json,
    group_annotations_by_image,
    draw_annotations_on_image
)

def main():
    """Main execution flow for generating Figure 2 collage."""
    # 1. Configuration & Paths
    csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
    model_name, merge_count, best_pred_path = find_best_pred_json_path(csv_path)
    
    if model_name is None:
        model_name = "internimage_large"
        merge_count = 3
        result_subdir = os.path.join(cfg.RESULT_PATH, "satellite_ade20k_250925_" + model_name, "thick=3,stride=10,extend=20")
        best_pred_path = os.path.join(result_subdir, "coco_pred_instances_merge3.json")
    else:
        result_subdir = os.path.dirname(best_pred_path)

    model_dir = cfg.MODEL_PREFIX + model_name
    model_type = "Internimage" if "internimage" in model_name.lower() else "mask2former"
    pred_dir = os.path.join(cfg.DATA_ROOT, model_type, model_dir, 'prediction')

    paths = {
        'json_gt': cfg.COCO_ANNO_PATH,
        'json_origin': os.path.join(result_subdir, "coco_pred_instances_origin.json"),
        'json_merged': best_pred_path,
        'pred_dir': pred_dir,
        'original_img_dir': os.path.join(cfg.DATASET_PATH, 'images', 'validation'),
        'output_dir': os.path.join(cfg.RESULT_PATH, 'Figure', 'Figure_2')
    }

    viz_settings = {
        'gap': 20,
        'gap_color': 0, # Black
        'exclude_ids': cfg.EXCLUDE_IDS,
        'font': cv2.FONT_HERSHEY_SIMPLEX,
        'font_scale': 1.2,
        'font_thickness': 3
    }

    print("Initializing Figure 2 generation...")
    os.makedirs(paths['output_dir'], exist_ok=True)

    # 2. Load all necessary data
    gt_data = load_json(paths['json_gt'])
    origin_anns = load_json(paths['json_origin'])
    merged_anns = load_json(paths['json_merged'])

    # 3. Group annotations by image_id for efficient lookup
    gt_map = group_annotations_by_image(gt_data['annotations'])
    origin_map = group_annotations_by_image(origin_anns)
    merged_map = group_annotations_by_image(merged_anns)

    # 4. Process images and create collages
    for img_info in tqdm(gt_data['images'], desc="Generating collages"):
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_w, img_h = img_info['width'], img_info['height']

        # (a) Original image + GT annotations
        img_a = generate_a_gt_overlay(file_name, gt_map.get(img_id, []), paths['original_img_dir'], viz_settings['exclude_ids'])
        
        # (b) Pixel-wise segmentation results
        img_b = generate_b_segmentation(file_name, paths['pred_dir'], viz_settings['exclude_ids'])
        
        # (c) Initial vectorized linestrings (from skeletonization)
        img_c = generate_vectorized_mask(img_h, img_w, origin_map.get(img_id, []), viz_settings['exclude_ids'])
        
        # (d) Final merged linestrings
        img_d = generate_vectorized_mask(img_h, img_w, merged_map.get(img_id, []), viz_settings['exclude_ids'])

        if any(img is None for img in [img_a, img_b, img_c, img_d]):
            continue

        # Combine into 2x2 collage
        collage = create_2x2_collage(img_a, img_b, img_c, img_d, gap=viz_settings['gap'], gap_color=viz_settings['gap_color'])
        
        # Save results
        save_path = os.path.join(paths['output_dir'], f"{os.path.splitext(file_name)[0]}.jpg")
        cv2.imwrite(save_path, collage)

    print(f"Done! Figure 2 collages are saved in: {paths['output_dir']}")


# ================= Helper Functions =================

def generate_a_gt_overlay(file_name, anns, img_dir, exclude_ids):
    """Draw Ground Truth polygons on original image."""
    img_path = os.path.join(img_dir, file_name)
    img = cv2.imread(img_path)
    if img is None:
        return None
    return draw_annotations_on_image(img, anns, exclude_ids)

def generate_b_segmentation(file_name, pred_dir, exclude_ids):
    """Load segmentation prediction and set background/excluded classes to white."""
    pred_path = os.path.join(pred_dir, file_name)
    if not os.path.exists(pred_path):
        base = os.path.splitext(file_name)[0]
        pred_path = os.path.join(pred_dir, base + ".png")
        
    img = cv2.imread(pred_path)
    if img is None:
        return None

    # Set background (ignore/black) to white
    black_mask = np.all(img == [0, 0, 0], axis=-1)
    img[black_mask] = [255, 255, 255]

    # Paint excluded categories white to hide them
    for cat_id in exclude_ids:
        if cat_id in cfg.ID2BGR:
            color = cfg.ID2BGR[cat_id]
            mask = np.all(img == color, axis=-1)
            img[mask] = [255, 255, 255]
        if hasattr(cfg, 'RENDER_ID2BGR') and cat_id in cfg.RENDER_ID2BGR:
            color = cfg.RENDER_ID2BGR[cat_id]
            mask = np.all(img == color, axis=-1)
            img[mask] = [255, 255, 255]

    # Replace original class colors with their render colors if they differ
    for cat_id, orig_color in cfg.ID2BGR.items():
        if cat_id in exclude_ids or cat_id == 0:
            continue
        render_color = cfg.RENDER_ID2BGR.get(cat_id)
        if render_color is not None and render_color != orig_color:
            mask = np.all(img == orig_color, axis=-1)
            img[mask] = render_color

    return img

def generate_vectorized_mask(h, w, anns, exclude_ids):
    """Render vectorized linestrings from JSON on a white background."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    return draw_annotations_on_image(img, anns, exclude_ids)

def create_2x2_collage(img_a, img_b, img_c, img_d, gap=20, gap_color=0):
    """Combine four images into a 2x2 grid with gaps."""
    h, w, _ = img_a.shape
    collage_h = h * 2 + gap
    collage_w = w * 2 + gap
    collage = np.full((collage_h, collage_w, 3), gap_color, dtype=np.uint8)
    
    collage[0:h, 0:w] = img_a
    collage[0:h, w+gap : w*2+gap] = img_b
    collage[h+gap : h*2+gap, 0:w] = img_c
    collage[h+gap : h*2+gap, w+gap : w*2+gap] = img_d
    return collage

if __name__ == "__main__":
    main()