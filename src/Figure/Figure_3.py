import os
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils

import config as cfg

# ================= 설정 섹션 =================
# 입력 경로
ORIGINAL_IMG_DIR = os.path.join(cfg.DATA_PATH, 'images', 'validation')
GT_JSON_PATH = cfg.COCO_ANNO_PATH
PRED_JSON_PATH = os.path.join(cfg.DATA_PATH, 'result', 'thickness=3', 'coco_pred_instances_merged.json')

# 출력 경로 (하위 폴더 포함)
BASE_OUT = os.path.join(cfg.DATA_PATH, 'result', 'Figure', 'Figure_3')

# 타겟 클래스 및 폴더명 설정
TARGET_CLASSES = {
    8: 'guiding_line',
    10: 'safety_zone'
}

# BGR 색상표 (METAINFO 기준)
PALETTE_BGR = {
    8: (77, 178, 255),  # guiding_line
    10: (128, 77, 255)  # safety_zone
}


# 폴더 생성 함수
def make_dirs(base, sub_name):
    for class_name in TARGET_CLASSES.values():
        path = os.path.join(base, sub_name, class_name)
        os.makedirs(path, exist_ok=True)
    return os.path.join(base, sub_name)


OUT_A_BASE = make_dirs(BASE_OUT, 'Figure_3_a_gt')
OUT_B_BASE = make_dirs(BASE_OUT, 'Figure_3_b_pred')

# 1. [작업 a] GT 시각화 (클래스별 별도 저장)
print("Processing Figure 3 (a) - GT by class...")
with open(GT_JSON_PATH, 'r') as f:
    gt_data = json.load(f)

img_to_gt = {}
annotations = gt_data.get('annotations', gt_data)
for ann in annotations:
    cat_id = ann['category_id']
    if cat_id in TARGET_CLASSES:
        img_id = ann['image_id']
        if img_id not in img_to_gt:
            img_to_gt[img_id] = {}
        if cat_id not in img_to_gt[img_id]:
            img_to_gt[img_id][cat_id] = []
        img_to_gt[img_id][cat_id].append(ann)

for img_id, cat_dict in img_to_gt.items():
    img_name = f"{img_id}.png"
    src_path = os.path.join(ORIGINAL_IMG_DIR, img_name)

    if os.path.exists(src_path):
        # 각 클래스별로 별도의 결과 이미지를 생성
        for cat_id, anns in cat_dict.items():
            img = cv2.imread(src_path)
            class_name = TARGET_CLASSES[cat_id]
            color = PALETTE_BGR[cat_id]

            for ann in anns:
                for seg in ann['segmentation']:
                    pts = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(img, [pts], color)

            # Figure_3_a_gt/[class_name]/이미지명.png 로 저장
            save_path = os.path.join(OUT_A_BASE, class_name, img_name)
            cv2.imwrite(save_path, img)

# 2. [작업 b] Pred 시각화 (클래스별 별도 저장)
print("Processing Figure 3 (b) - Pred by class...")
with open(PRED_JSON_PATH, 'r') as f:
    pred_data = json.load(f)

img_to_pred = {}
for ann in pred_data:
    cat_id = ann['category_id']
    if cat_id in TARGET_CLASSES:
        img_id = ann['image_id']
        if img_id not in img_to_pred:
            img_to_pred[img_id] = {}
        if cat_id not in img_to_pred[img_id]:
            img_to_pred[img_id][cat_id] = []
        img_to_pred[img_id][cat_id].append(ann)

for img_id, cat_dict in img_to_pred.items():
    img_name = f"{img_id}.png"
    src_path = os.path.join(ORIGINAL_IMG_DIR, img_name)

    if os.path.exists(src_path):
        for cat_id, anns in cat_dict.items():
            img = cv2.imread(src_path)
            class_name = TARGET_CLASSES[cat_id]
            color = PALETTE_BGR[cat_id]

            for ann in anns:
                mask = maskUtils.decode(ann['segmentation'])
                img[mask > 0] = color

            save_path = os.path.join(OUT_B_BASE, class_name, img_name)
            cv2.imwrite(save_path, img)

print(f"Done! All Figure 3 images are organized by class in: {BASE_OUT}")