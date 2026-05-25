import os
import sys
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

# Setup project root and import config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import src.config as cfg

def main():
    """Main execution flow for generating Figure 3."""
    # Configuration
    target_ids = [8, 10]  # 8: guiding_line, 10: safety_zone
    min_instances = 2
    gap = 20
    gap_color = (0, 0, 0)
    
    paths = {
        'gt_json': cfg.COCO_ANNO_PATH,
        'pred_dir': cfg.PRED_PATH,
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

def load_json(path):
    """Load JSON file and return data."""
    if not os.path.exists(path):
        print(f"Warning: File not found - {path}")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def group_annotations_by_image(annotations):
    """Group list of annotations into a dictionary keyed by image_id."""
    mapping = {}
    for ann in annotations:
        img_id = ann['image_id']
        mapping.setdefault(img_id, []).append(ann)
    return mapping

def generate_gt_overlay(img, annotations, target_id):
    """Draw Ground Truth polygons on image for a specific category."""
    color = cfg.RENDER_ID2BGR.get(target_id, (255, 255, 255))
    
    for ann in annotations:
        if 'segmentation' in ann:
            segs = ann['segmentation']
            if isinstance(segs, list):
                # Handle COCO polygon format
                if len(segs) > 0 and isinstance(segs[0], (int, float)):
                    segs = [segs]
                for seg in segs:
                    if len(seg) < 6: continue
                    pts = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(img, [pts], color)
            elif isinstance(segs, dict):
                # Handle RLE format
                mask = maskUtils.decode(segs)
                img[mask > 0] = color
    return img

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