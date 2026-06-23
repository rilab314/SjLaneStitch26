import os
import sys
import json
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
src_dir = os.path.join(project_root, "src")
backup_dir = os.path.join(src_dir, "backup")

for path in [project_root, src_dir, backup_dir]:
    if path not in sys.path:
        sys.path.append(path)

import src.config as cfg
from src.lane_stitcher import LaneStitcher
from src.util import find_best_pred_json_path

def main():
    """Main execution flow for generating Figure 4 collages and individual panels."""
    # 1. Configuration & Paths
    csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
    model_name, merge_count, best_pred_path = find_best_pred_json_path(csv_path)
    
    if model_name is None:
        model_name = "internimage_large"
        result_subdir = os.path.join(cfg.RESULT_PATH, "satellite_ade20k_250925_" + model_name, "thick=3,stride=10,extend=20")
    else:
        result_subdir = os.path.dirname(best_pred_path)

    model_dir = cfg.MODEL_PREFIX + model_name
    model_type = "Internimage" if "internimage" in model_name.lower() else "mask2former"
    model_path = os.path.join(cfg.DATA_ROOT, model_type, model_dir)
    pred_dir = os.path.join(model_path, 'prediction')

    paths = {
        'json_origin': os.path.join(result_subdir, "coco_pred_instances_origin.json"),
        'json_merge1': os.path.join(result_subdir, "coco_pred_instances_merge1.json"),
        'pred_dir': pred_dir,
        'original_img_dir': os.path.join(cfg.DATASET_PATH, 'images', 'validation'),
        'output_dir': os.path.join(cfg.RESULT_PATH, 'Figure', 'Figure_4')
    }

    viz_settings = {
        'gap': 20,
        'gap_color': (0, 0, 0),  # Black separator line
        'class_color': cfg.RENDER_ID2BGR.get(1, cfg.ID2BGR.get(1, (255, 77, 77))),  # center_line BGR color
        'ext_color': (0, 0, 255)  # Red for extrapolation
    }

    print("Initializing Figure 4 generation...")
    os.makedirs(paths['output_dir'], exist_ok=True)

    # 2. Filter Candidate Frames
    candidates = select_candidates(paths['json_origin'], paths['json_merge1'])
    if not candidates:
        print("No candidate frames matching the 50% reduction criteria found.")
        return

    print(f"Selected {len(candidates)} candidate frames for Figure 4 generation.")

    # 3. Instantiate detector to fetch intermediate steps
    param_dir_name = os.path.basename(result_subdir)
    params = dict(item.split('=') for item in param_dir_name.split(','))
    t = int(params.get('thick', 3))
    s = int(params.get('stride', 10))
    e = int(params.get('extend', 20))

    detector = LaneStitcher(
        cfg.DATASET_PATH, 
        model_path, 
        cfg.RESULT_PATH,
        thickness=t,
        sample_stride=s,
        extend_len=e
    )

    # 4. Generate panels and collages
    total_cand = len(candidates)
    for idx, img_id in enumerate(candidates):
        print(f"[{idx+1}/{total_cand}] Processing Figure 4 for image: {img_id}")
        img_file = os.path.join(paths['original_img_dir'], f"{img_id}.png")
        if not os.path.exists(img_file):
            continue

        # Extract strands before merge
        lines, img_shape = detector.get_linestrings_for_image(img_file)
        h, w = img_shape

        # Filter only center_line class (class_id = 1)
        center_lines = [line for line in lines if line.class_id == 1]

        # Generate panels
        img_a = generate_panel_a(img_id, paths['pred_dir'], viz_settings['class_color'])
        img_b = generate_panel_b(center_lines, h, w, viz_settings['class_color'])
        img_c = generate_panel_c(center_lines, h, w, viz_settings['class_color'], viz_settings['ext_color'])
        img_d = generate_panel_d(detector, lines, h, w, viz_settings['class_color'])

        if any(img is None for img in [img_a, img_b, img_c, img_d]):
            print(f"Warning: Failed to generate all panels for {img_id}. Skipping.")
            continue

        # Create 1x4 horizontal collage with 20px black separator lines in between
        collage = create_horizontal_collage(
            img_a, img_b, img_c, img_d,
            gap=viz_settings['gap'],
            gap_color=viz_settings['gap_color']
        )

        # Save collage
        cv2.imwrite(os.path.join(paths['output_dir'], f"{img_id}.png"), collage)

    print(f"Done! Figure 4 images are saved in: {paths['output_dir']}")


def select_candidates(json_origin_path, json_merge1_path):
    """Identify image IDs where the center_line count decreased by 50% or more."""
    if not os.path.exists(json_origin_path) or not os.path.exists(json_merge1_path):
        print(f"Error: Pred JSON files not found in: {os.path.dirname(json_origin_path)}")
        return []

    with open(json_origin_path, 'r') as f:
        origin_anns = json.load(f)
    with open(json_merge1_path, 'r') as f:
        merge1_anns = json.load(f)

    # Group by image_id
    from collections import defaultdict
    origin_map = defaultdict(list)
    merge1_map = defaultdict(list)

    for ann in origin_anns:
        if ann.get('category_id') == 1:
            origin_map[ann['image_id']].append(ann)
    for ann in merge1_anns:
        if ann.get('category_id') == 1:
            merge1_map[ann['image_id']].append(ann)

    candidates = []
    all_image_ids = set(origin_map.keys()).union(set(merge1_map.keys()))

    for img_id in sorted(all_image_ids):
        o_count = len(origin_map[img_id])
        m_count = len(merge1_map[img_id])
        if o_count > 0 and m_count <= 0.5 * o_count:
            candidates.append(img_id)

    return candidates


def generate_panel_a(img_id, pred_dir, color):
    """Panel (a): Semantic segmentation result for center_line (class color on white background)."""
    pred_path = os.path.join(pred_dir, f"{img_id}.png")
    if not os.path.exists(pred_path):
        pred_path = os.path.join(pred_dir, f"{img_id}.jpg")
    img = cv2.imread(pred_path)
    if img is None:
        return None

    # Use original color to find pixels, paint with the new render color
    orig_color = cfg.ID2BGR.get(1, (255, 77, 77))
    render_color = cfg.RENDER_ID2BGR.get(1, orig_color)

    # Filter only class color
    mask = np.all(img == orig_color, axis=-1)
    out = np.full_like(img, 255)
    out[mask] = render_color
    return out


def generate_panel_b(center_lines, h, w, color):
    """Panel (b): Initial linestring representation (class color on white background)."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for line in center_lines:
        if line.points is not None and len(line.points) > 0:
            pts = line.points.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=3)
    return img


def generate_panel_c(center_lines, h, w, class_color, ext_color):
    """Panel (c): Extrapolation of endpoints (extended in red, with endpoint circles)."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for line in center_lines:
        if line.points is not None and len(line.points) > 0:
            # 1. Original linestring in class color
            pts = line.points.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=class_color, thickness=3)

            # 2. Extrapolated parts in red
            ext_pts = line.ext_points
            src_range = line.src_range
            if ext_pts is not None and src_range is not None:
                n_head = src_range[0]
                n_tail_start = src_range[1]

                # Head extension
                if n_head > 0:
                    head_line = ext_pts[0 : n_head + 1].reshape((-1, 1, 2))
                    cv2.polylines(img, [head_line], isClosed=False, color=ext_color, thickness=3)
                    # Markers
                    cv2.circle(img, tuple(ext_pts[0]), radius=3, color=ext_color, thickness=-1)
                    cv2.circle(img, tuple(ext_pts[n_head]), radius=3, color=ext_color, thickness=-1)

                # Tail extension
                if n_tail_start < len(ext_pts) - 1:
                    tail_line = ext_pts[n_tail_start :].reshape((-1, 1, 2))
                    cv2.polylines(img, [tail_line], isClosed=False, color=ext_color, thickness=3)
                    # Markers
                    cv2.circle(img, tuple(ext_pts[n_tail_start]), radius=3, color=ext_color, thickness=-1)
                    cv2.circle(img, tuple(ext_pts[-1]), radius=3, color=ext_color, thickness=-1)
    return img


def generate_panel_d(detector, lines, h, w, color):
    """Panel (d): Final integrated linestring object resulting from merging."""
    # Perform merge logic for merge1 (iter=0)
    merged_lines, _ = detector.merge_lines(lines, 0)
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for line in merged_lines:
        if line.class_id == 1 and line.points is not None and len(line.points) > 0:
            pts = line.points.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=3)
    return img


def create_horizontal_collage(img_a, img_b, img_c, img_d, gap=20, gap_color=(0, 0, 0)):
    """Create a 1x4 horizontal collage with solid gap_color separator lines in between."""
    h, w, _ = img_a.shape
    collage_w = w * 4 + gap * 3
    collage = np.full((h, collage_w, 3), gap_color, dtype=np.uint8)

    collage[:, 0:w] = img_a
    collage[:, w+gap : w*2+gap] = img_b
    collage[:, w*2+gap*2 : w*3+gap*2] = img_c
    collage[:, w*3+gap*3 : w*4+gap*3] = img_d

    return collage


if __name__ == "__main__":
    main()