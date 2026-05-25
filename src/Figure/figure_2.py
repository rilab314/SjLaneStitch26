import os
import sys
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import src.config as cfg

def main():
    """Main execution flow for generating Figure 2 collage."""
    # 1. Configuration & Paths
    best_model = "satellite_ade20k_250925_internimage_large"
    best_param = "thick=3,stride=10,extend=20"
    result_subdir = os.path.join(cfg.RESULT_PATH, best_model, best_param)

    paths = {
        'json_gt': cfg.COCO_ANNO_PATH,
        'json_origin': os.path.join(result_subdir, "coco_pred_instances_origin.json"),
        'json_merged': os.path.join(result_subdir, "coco_pred_instances_merge3.json"),
        'pred_dir': cfg.PRED_PATH,
        'original_img_dir': os.path.join(cfg.DATASET_PATH, 'images', 'validation'),
        'output_dir': os.path.join(cfg.RESULT_PATH, 'Figure', 'Figure_2')
    }

    viz_settings = {
        'gap': 20,
        'gap_color': 0, # Black
        'exclude_ids': [0, 8, 10, 11],
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
        img_b = generate_b_segmentation(file_name, paths['pred_dir'])
        
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

def load_json(path):
    """Load JSON file and return data."""
    if not os.path.exists(path):
        print(f"Warning: File not found - {path}")
        return []
    with open(path, 'r') as f:
        return json.load(f)

def group_annotations_by_image(annotations):
    """Group list of annotations into a dictionary keyed by image_id."""
    mapping = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in mapping:
            mapping[img_id] = []
        mapping[img_id].append(ann)
    return mapping

def generate_a_gt_overlay(file_name, anns, img_dir, exclude_ids):
    """Draw Ground Truth polygons on original image."""
    img_path = os.path.join(img_dir, file_name)
    img = cv2.imread(img_path)
    if img is None:
        return None

    for ann in anns:
        cat_id = ann['category_id']
        if cat_id in exclude_ids:
            continue
        
        color = cfg.ID2BGR.get(cat_id, (255, 255, 255))
        if 'segmentation' in ann:
            segs = ann['segmentation']
            if isinstance(segs, list):
                if len(segs) > 0 and isinstance(segs[0], (int, float)):
                    segs = [segs]
                
                for seg in segs:
                    if len(seg) < 6:
                        continue
                    pts = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(img, [pts], color)
    return img

def generate_b_segmentation(file_name, pred_dir):
    """Load segmentation prediction and set background to white."""
    pred_path = os.path.join(pred_dir, file_name)
    if not os.path.exists(pred_path):
        base = os.path.splitext(file_name)[0]
        pred_path = os.path.join(pred_dir, base + ".png")
        
    img = cv2.imread(pred_path)
    if img is None:
        return None

    black_mask = np.all(img == [0, 0, 0], axis=-1)
    img[black_mask] = [255, 255, 255]
    return img

def generate_vectorized_mask(h, w, anns, exclude_ids):
    """Render vectorized linestrings from JSON on a white background."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    for ann in anns:
        cat_id = ann['category_id']
        if cat_id in exclude_ids:
            continue
        
        color = cfg.ID2BGR.get(cat_id, (255, 255, 255))
        if 'segmentation' in ann:
            mask = maskUtils.decode(ann['segmentation'])
            img[mask > 0] = color
    return img


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