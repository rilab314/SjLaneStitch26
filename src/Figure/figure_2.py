import os
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils

import config as cfg

# ================= 설정 섹션 (경로 하드코딩) =================
# 입력 경로
ORIGINAL_IMG_DIR = os.path.join(cfg.DATA_PATH, 'images', 'validation')  # 원본 이미지 폴더
JSON_A_PATH = cfg.COCO_ANNO_PATH  # 데이터 (a) 경로
IMG_B_DIR = os.path.join(cfg.DATA_PATH, 'prediction')  # 이미지 (b) 폴더
IMG_C_DIR = os.path.join(cfg.DATA_PATH, 'result', 'Figure', 'Figure_2', 'Figure_2_c')  # 이미지 (c) 폴더
JSON_D_PATH = os.path.join(cfg.DATA_PATH, 'result', 'coco_pred_instances_merged.json')  # 데이터 (d) 경로

# 출력 경로
BASE_OUT = os.path.join(cfg.DATA_PATH, 'result', 'Figure', 'Figure_2')
OUT_A = os.path.join(BASE_OUT, 'Figure_2_a')
OUT_B = os.path.join(BASE_OUT, 'Figure_2_b')
OUT_C = os.path.join(BASE_OUT, 'Figure_2_c_white')
OUT_D = os.path.join(BASE_OUT, 'Figure_2_d')

# 제외할 카테고리 ID 리스트
EXCLUDE_IDS = [8, 10]

# METAINFO 기반 색상표 구성
METAINFO = [
    {'id': 0, 'name': 'ignore', 'color': (0, 0, 0)},
    {'id': 1, 'name': 'center_line', 'color': (255, 77, 77)},
    {'id': 2, 'name': 'u_turn_zone_line', 'color': (255, 178, 77)},
    {'id': 3, 'name': 'lane_line', 'color': (77, 255, 77)},
    {'id': 4, 'name': 'bus_only_lane', 'color': (77, 153, 255)},
    {'id': 5, 'name': 'edge_line', 'color': (77, 77, 255)},
    {'id': 6, 'name': 'path_change_restriction_line', 'color': (255, 77, 178)},
    {'id': 7, 'name': 'no_parking_stopping_line', 'color': (178, 255, 77)},
    {'id': 8, 'name': 'guiding_line', 'color': (77, 178, 255)},
    {'id': 9, 'name': 'stop_line', 'color': (255, 102, 77)},
    {'id': 10, 'name': 'safety_zone', 'color': (128, 77, 255)},
    {'id': 11, 'name': 'bicycle_lane', 'color': (77, 255, 128)},
]

# ID를 키로 하는 팔레트 딕셔너리 생성
PALETTE_BGR = {}
for item in METAINFO:
    # 8, 10번은 검은색으로 처리하여 나중에 배경(흰색)으로 바뀌게 함
    if item['id'] in EXCLUDE_IDS:
        PALETTE_BGR[item['id']] = (0, 0, 0)
    else:
        PALETTE_BGR[item['id']] = item['color']

for p in [OUT_A, OUT_B, OUT_C, OUT_D]:
    os.makedirs(p, exist_ok=True)


def process_background_white_with_exclusion(img_path, save_path):
    """배경(0,0,0)과 제외된 클래스 색상을 찾아 흰색(255,255,255)으로 변경"""
    img = cv2.imread(img_path)
    if img is None: return

    # 1. 원본 팔레트에서 8, 10번의 실제 색상을 찾아와서 배경과 함께 흰색으로 처리
    # (이미 생성된 이미지 B, C를 처리하기 위한 로직)
    target_colors = [(0, 0, 0)] # 기본 배경색
    for item in METAINFO:
        if item['id'] in EXCLUDE_IDS:
            # 원본 색상 (BGR 순서로 뒤집어서 추가)
            target_colors.append(item['color'])

    for color in target_colors:
        mask = np.all(img == color, axis=-1)
        img[mask] = [255, 255, 255]

    cv2.imwrite(save_path, img)


# 1. [작업 a] 원본 이미지 위에 GT 그리기 (8, 10 제외)
print("Processing (a)...")
with open(JSON_A_PATH, 'r') as f:
    data_a = json.load(f)

img_to_anns = {}
annotations = data_a.get('annotations', data_a)
if not isinstance(annotations, list):
    annotations = [annotations]

for ann in annotations:
    img_id = ann['image_id']
    if img_id not in img_to_anns:
        img_to_anns[img_id] = []
    img_to_anns[img_id].append(ann)

for img_id, anns in img_to_anns.items():
    img_name = f"{img_id}.png"
    src_path = os.path.join(ORIGINAL_IMG_DIR, img_name)

    if os.path.exists(src_path):
        img = cv2.imread(src_path)

        for ann in anns:
            category_id = ann['category_id']
            if category_id in EXCLUDE_IDS:
                continue # 8, 10번은 그리지 않음

            color = PALETTE_BGR.get(category_id, (255, 255, 255))
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(img, [pts], color)

        cv2.imwrite(os.path.join(OUT_A, img_name), img)

# 2. [작업 b & c] 기존 이미지 처리 (배경 + 8, 10번을 흰색으로)
print("Processing (b) and (c)...")
for folder_in, folder_out in [(IMG_B_DIR, OUT_B), (IMG_C_DIR, OUT_C)]:
    if not os.path.exists(folder_in):
        continue
    for f_name in os.listdir(folder_in):
        if f_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_background_white_with_exclusion(os.path.join(folder_in, f_name), os.path.join(folder_out, f_name))

# 3. [작업 d] JSON(d) 기반 (8, 10 제외하고 흰 배경에 그리기)
print("Processing (d)...")
with open(JSON_D_PATH, 'r') as f:
    data_d = json.load(f)

combined_masks = {}

for ann in data_d:
    category_id = ann['category_id']
    if category_id in EXCLUDE_IDS:
        continue # 8, 10번 제외

    img_id = ann['image_id']
    if img_id not in combined_masks:
        h, w = ann['segmentation']['size']
        combined_masks[img_id] = np.zeros((h, w, 3), dtype=np.uint8)

    mask = maskUtils.decode(ann['segmentation'])
    color = PALETTE_BGR.get(category_id, (255, 255, 255))
    combined_masks[img_id][mask > 0] = color

# 결과 저장 (검은 배경 -> 흰색 반전)
for img_id, mask_img in combined_masks.items():
    black_pixels = np.all(mask_img == [0, 0, 0], axis=-1)
    mask_img[black_pixels] = [255, 255, 255]
    cv2.imwrite(os.path.join(OUT_D, f"{img_id}.png"), mask_img)

print(f"Done! All figures (excluding ID 8, 10) are saved in: {BASE_OUT}")