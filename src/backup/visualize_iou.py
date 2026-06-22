import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm

# ---------------------------------------------------------
# 프로젝트 경로 설정 (../ 유지)
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

import config as cfg


def get_gradient_color(iou, low_color=(180, 255, 100), high_color=(50, 50, 255)):
    """
    IoU에 따라 민트색(Low)에서 파란색(High)으로 변하는 BGR 색상을 반환합니다.
    """
    t = max(0.0, min(1.0, float(iou)))
    color = tuple(
        int(low_color[i] * (1 - t) + high_color[i] * t)
        for i in range(3)
    )
    return color


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def visualize_and_logging_iou():
    # 1. 경로 설정
    IOU_THRESHOLD = 0.5

    coco_anno_path = cfg.COCO_MERGED_ANNO_PATH
    pred_path = cfg.MERGED_JSON_PATH
    output_dir = os.path.join(cfg.RESULT_PATH, f'iou_vis_and_report_{IOU_THRESHOLD}')
    csv_save_path = os.path.join(output_dir, "instance_statistics.csv")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. GT 데이터 로드 및 필수 필드 보정 (KeyError 방지 핵심)
    print(f"Loading GT from {coco_anno_path}...")
    with open(coco_anno_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    if 'annotations' in gt_data:
        for i, ann in enumerate(gt_data['annotations']):
            if 'id' not in ann: ann['id'] = i + 1
            if 'iscrowd' not in ann: ann['iscrowd'] = 0
            if 'area' not in ann:
                ann['area'] = float(maskUtils.area(ann['segmentation'])) if 'segmentation' in ann else 0.0

    coco = COCO()
    coco.dataset = gt_data
    coco.createIndex()

    # 클래스 정보 정렬
    cat_ids = sorted(coco.getCatIds())
    cats = coco.loadCats(cat_ids)
    cat_id_to_display_name = {cat['id']: f"{cat['id']}. {cat['name']}" for cat in cats}
    ordered_column_names = [cat_id_to_display_name[cid] for cid in cat_ids]

    print(f"Loading Preds from {pred_path}...")
    preds_data = load_json(pred_path)

    preds_by_img = {}
    for p in preds_data:
        img_id = str(p['image_id'])
        if img_id not in preds_by_img:
            preds_by_img[img_id] = []
        preds_by_img[img_id].append(p)

    img_ids = coco.getImgIds()

    # 색상 정의 (BGR)
    COLOR_FN = (0, 0, 255)  # 미검출 (Red)
    COLOR_FP = (0, 255, 255)  # 오검출 (Yellow)
    COLOR_TP_LOW = (180, 255, 100)  # IoU 낮음
    COLOR_TP_HIGH = (255, 50, 50)  # IoU 높음

    csv_rows = []
    total_stats = {name: {"p": 0, "g": 0, "m": 0} for name in ordered_column_names}
    total_overall = {"p": 0, "g": 0, "m": 0}

    for img_id_int in tqdm(img_ids):
        img_info = coco.loadImgs([img_id_int])[0]
        file_name = img_info['file_name']
        h, w = img_info['height'], img_info['width']

        # GT 데이터 로드
        ann_ids = coco.getAnnIds(imgIds=[img_id_int])
        anns = coco.loadAnns(ann_ids)
        gt_masks = [coco.annToMask(ann) for ann in anns]
        gt_classes = [ann['category_id'] for ann in anns]
        gt_matched = [False] * len(anns)

        # Pred 데이터 로드
        str_id = os.path.splitext(os.path.basename(file_name))[0]
        # ID 매칭 시 문자열/정수 모두 대응
        preds = preds_by_img.get(str_id, preds_by_img.get(str(img_id_int), []))

        pred_masks = []
        pred_classes = []
        for p in preds:
            m = maskUtils.decode(p['segmentation'])
            if m.ndim == 3: m = m[..., 0]
            pred_masks.append(m)
            pred_classes.append(p['category_id'])

        pred_matched = [False] * len(preds)
        pred_ious = [0.0] * len(preds)

        img_stats = {name: {"p": 0, "g": 0, "m": 0} for name in ordered_column_names}
        img_total = {"p": 0, "g": 0, "m": 0}

        for gc in gt_classes:
            if gc in cat_id_to_display_name:
                name = cat_id_to_display_name[gc]
                img_stats[name]["g"] += 1
                total_stats[name]["g"] += 1
                img_total["g"] += 1
                total_overall["g"] += 1

        for pc in pred_classes:
            if pc in cat_id_to_display_name:
                name = cat_id_to_display_name[pc]
                img_stats[name]["p"] += 1
                total_stats[name]["p"] += 1
                img_total["p"] += 1
                total_overall["p"] += 1

        # 3. 인스턴스 매칭
        iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))
        for i in range(len(gt_masks)):
            for j in range(len(pred_masks)):
                if gt_classes[i] == pred_classes[j]:
                    iou_matrix[i, j] = compute_iou(gt_masks[i], pred_masks[j])

        if iou_matrix.size > 0:
            flat_indices = np.argsort(-iou_matrix, axis=None)
            gt_indices, pred_indices = np.unravel_index(flat_indices, iou_matrix.shape)

            for gi, pi in zip(gt_indices, pred_indices):
                if gt_matched[gi] or pred_matched[pi]: continue
                iou = iou_matrix[gi, pi]
                if iou < IOU_THRESHOLD: break

                gt_matched[gi] = True
                pred_matched[pi] = True
                pred_ious[pi] = iou

                name = cat_id_to_display_name[gt_classes[gi]]
                img_stats[name]["m"] += 1
                total_stats[name]["m"] += 1
                img_total["m"] += 1
                total_overall["m"] += 1

        # 4. 시각화
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for i, matched in enumerate(gt_matched):
            if not matched: canvas[gt_masks[i] > 0] = COLOR_FN
        for i, matched in enumerate(pred_matched):
            if not matched: canvas[pred_masks[i] > 0] = COLOR_FP
        for i, matched in enumerate(pred_matched):
            if matched:
                color = get_gradient_color(pred_ious[i], COLOR_TP_LOW, COLOR_TP_HIGH)
                canvas[pred_masks[i] > 0] = color

        cv2.imwrite(os.path.join(output_dir, os.path.basename(file_name)), canvas)

        # 5. CSV 행 구성
        row_dict = {"Image Name": file_name}
        for name in ordered_column_names:
            s = img_stats[name]
            row_dict[name] = f"{s['p']} / {s['g']} (M: {s['m']})"
        row_dict["TOTAL_IMG"] = f"{img_total['p']} / {img_total['g']} (M: {img_total['m']})"
        csv_rows.append(row_dict)

    # 6. TOTAL 행 추가
    total_row = {"Image Name": "TOTAL_ALL"}
    for name in ordered_column_names:
        ts = total_stats[name]
        total_row[name] = f"{ts['p']} / {ts['g']} (M: {ts['m']})"
    total_row["TOTAL_IMG"] = f"{total_overall['p']} / {total_overall['g']} (M: {total_overall['m']})"
    csv_rows.append(total_row)

    # 7. 최종 CSV 저장
    df = pd.DataFrame(csv_rows)
    cols = ["Image Name"] + ordered_column_names + ["TOTAL_IMG"]
    df = df[cols]
    df.set_index("Image Name", inplace=True)
    df.to_csv(csv_save_path, encoding='utf-8-sig')

    print(f"\n[Success] Images and CSV report saved in: {output_dir}")


if __name__ == '__main__':
    visualize_and_logging_iou()