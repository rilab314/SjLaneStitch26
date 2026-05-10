import os
import cv2
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

METAINFO = [
    {'id': 0, 'name': 'ignore', 'color': (0, 0, 0)},
    {'id': 1, 'name': 'center_line', 'color': (77, 77, 255)},
    {'id': 2, 'name': 'u_turn_zone_line', 'color': (77, 178, 255)},
    {'id': 3, 'name': 'lane_line', 'color': (77, 255, 77)},
    {'id': 4, 'name': 'bus_only_lane', 'color': (255, 153, 77)},
    {'id': 5, 'name': 'edge_line', 'color': (255, 77, 77)},
    {'id': 6, 'name': 'path_change_restriction_line', 'color': (178, 77, 255)},
    {'id': 7, 'name': 'no_parking_stopping_line', 'color': (77, 255, 178)},
    {'id': 8, 'name': 'guiding_line', 'color': (255, 178, 77)},
    {'id': 9, 'name': 'stop_line', 'color': (77, 102, 255)},
    {'id': 10, 'name': 'safety_zone', 'color': (255, 77, 128)},
    {'id': 11, 'name': 'bicycle_lane', 'color': (128, 255, 77)},
]
CATEGORY_COLOR = {m["id"]: m["color"] for m in METAINFO}

def ann_to_mask(ann, h, w):
    segm = ann["segmentation"]
    if isinstance(segm, dict):  # RLE
        rle = segm
        if isinstance(rle.get("counts"), bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        return maskUtils.decode(rle)
    elif isinstance(segm, list):  # polygon
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles) if isinstance(rles, list) else rles
        return maskUtils.decode(rle)
    else:
        return np.zeros((h, w), dtype=np.uint8)

def overlay_mask(image, mask, color, alpha=0.5):
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def visualize_gt(gt_json, img_root, out_dir, alpha=0.5):
    os.makedirs(out_dir, exist_ok=True)
    coco = COCO(gt_json)  # ← GT만 COCO()로
    for img_id in coco.getImgIds():
        im = coco.loadImgs([img_id])[0]
        file_name, h, w = im["file_name"], im["height"], im["width"]
        img_path = os.path.join(img_root, file_name)
        image = cv2.imread(img_path) if os.path.exists(img_path) else np.zeros((h, w, 3), np.uint8)

        vis = image.copy()
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            color = CATEGORY_COLOR.get(ann["category_id"], (255, 255, 255))
            mask = ann_to_mask(ann, h, w)
            vis = overlay_mask(vis, mask, color, alpha=alpha)

        out_path = os.path.join(out_dir, f"{os.path.splitext(file_name)[0]}.png")
        cv2.imwrite(out_path, vis, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print("[saved]", out_path)

def visualize_pred(gt_json, pred_json, img_root, out_dir, alpha=0.5, score_thr=0.0, max_dets=100):
    os.makedirs(out_dir, exist_ok=True)
    coco_gt = COCO(gt_json)              # GT 메타(이미지 크기/파일명)를 사용
    coco_dt = coco_gt.loadRes(pred_json) # ← 예측은 loadRes()로 감싸기

    for img_id in coco_gt.getImgIds():
        im = coco_gt.loadImgs([img_id])[0]
        file_name, h, w = im["file_name"], im["height"], im["width"]
        img_path = os.path.join(img_root, file_name)
        image = cv2.imread(img_path) if os.path.exists(img_path) else np.zeros((h, w, 3), np.uint8)

        # 이 이미지의 예측들 (점수/개수 제한)
        dt_ids = coco_dt.getAnnIds(imgIds=[img_id])
        dts = sorted(coco_dt.loadAnns(dt_ids), key=lambda d: -d.get("score", 0.0))
        dts = [d for d in dts if d.get("score", 0.0) >= score_thr][:max_dets]

        vis = image.copy()
        for d in dts:
            color = CATEGORY_COLOR.get(d["category_id"], (255, 255, 255))
            mask = ann_to_mask(d, h, w)
            vis = overlay_mask(vis, mask, color, alpha=alpha)

        out_path = os.path.join(out_dir, f"{os.path.splitext(file_name)[0]}.png")
        cv2.imwrite(out_path, vis, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print("[saved]", out_path)

if __name__ == "__main__":
    gt_json = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/new_coco_dataset/annotations/instances_validation2017.json'
    pred_json = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/ade20k/satellite_ade20k_250820/process/coco_pred_instances_merged.json'
    img_root = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/ade20k/satellite_ade20k_250820/images/validation'
    gt_out_dir = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/ade20k/satellite_ade20k_250820/gt_on_image'
    pred_out_dir = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/ade20k/satellite_ade20k_250820/pred_on_image'

    os.makedirs(gt_out_dir, exist_ok=True)
    os.makedirs(pred_out_dir, exist_ok=True)

    visualize_gt(gt_json, img_root, gt_out_dir, alpha=0.5)
    visualize_pred(gt_json, pred_json, img_root, pred_out_dir, alpha=0.5, score_thr=0.0, max_dets=100)
