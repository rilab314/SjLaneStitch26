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

def main():
    """Main execution flow for generating Figure 3."""
    # Configuration
    target_ids = [8, 10]  # 8: guiding_line, 10: safety_zone
    min_instances = 2
    gap = 20
    gap_color = (0, 0, 0)
    
    csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
    model_name, _, _ = find_best_pred_json_path(csv_path)
    
    if model_name is None:
        model_name = "internimage_large"
        
    model_dir = cfg.MODEL_PREFIX + model_name
    model_type = "Internimage" if "internimage" in model_name.lower() else "mask2former"
    pred_dir = os.path.join(cfg.DATA_ROOT, model_type, model_dir, 'prediction')
    
    paths = {
        'gt_json': cfg.COCO_ANNO_PATH,
        'pred_dir': pred_dir,
        'img_dir': os.path.join(cfg.DATASET_PATH, 'images', 'validation'),
        'output_dir': os.path.join(cfg.RESULT_PATH, 'Figure', 'Figure_3')
    }
    
    os.makedirs(paths['output_dir'], exist_ok=True)
    print("Initializing Figure 3 generation...")
    
    # Load data
    gt_data = load_json(paths['gt_json'])
    if not gt_data:
        return
        
    gt_map = group_annotations_by_image(gt_data['annotations'])
    
    # Process images
    for img_info in tqdm(gt_data['images'], desc="Generating Figure 3"):
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_anns = gt_map.get(img_id, [])
        
        # Filter for target classes
        guiding_anns = [ann for ann in img_anns if ann['category_id'] == 8]
        safety_anns = [ann for ann in img_anns if ann['category_id'] == 10]
        
        has_guiding = len(guiding_anns) >= min_instances
        has_safety = len(safety_anns) >= min_instances
        
        if not has_guiding and not has_safety:
            continue
            
        img_path = os.path.join(paths['img_dir'], file_name)
        base_img = cv2.imread(img_path)
        if base_img is None:
            continue
            
        pred_path = os.path.join(paths['pred_dir'], file_name)
        seg_img = cv2.imread(pred_path)
        
        pair_guiding = None
        if has_guiding:
            img_a = generate_gt_overlay(base_img.copy(), guiding_anns, 8)
            img_b = generate_pred_overlay(base_img.copy(), seg_img, 8)
            pair_guiding = np.hstack([img_a, img_b])
            
        pair_safety = None
        if has_safety:
            img_c = generate_gt_overlay(base_img.copy(), safety_anns, 10)
            img_d = generate_pred_overlay(base_img.copy(), seg_img, 10)
            pair_safety = np.hstack([img_c, img_d])
            
        # Combine pairs
        if pair_guiding is not None and pair_safety is not None:
            h, w, _ = pair_guiding.shape
            black_gap = np.full((gap, w, 3), gap_color, dtype=np.uint8)
            final_img = np.vstack([pair_guiding, black_gap, pair_safety])
        elif pair_guiding is not None:
            final_img = pair_guiding
        else:
            final_img = pair_safety
            
        # Save
        save_path = os.path.join(paths['output_dir'], file_name)
        cv2.imwrite(save_path, final_img)

    print(f"Done! Figure 3 images are saved in: {paths['output_dir']}")


def generate_gt_overlay(img, annotations, target_id):
    """Draw Ground Truth polygons on image for a specific category."""
    target_anns = [ann for ann in annotations if ann.get('category_id') == target_id]
    return draw_annotations_on_image(img, target_anns, [])


def generate_pred_overlay(img, seg_img, target_id):
    """Overlay Prediction masks on image for a specific category."""
    if seg_img is None:
        return img
        
    target_color_bgr = cfg.ID2BGR.get(target_id)
    render_color_bgr = cfg.RENDER_ID2BGR.get(target_id, target_color_bgr)
    
    if target_color_bgr is None:
        return img
        
    # Find pixels matching target class in segmentation image
    mask = np.all(seg_img == target_color_bgr, axis=-1)
    img[mask] = render_color_bgr
    return img


if __name__ == "__main__":
    main()