import os
import cv2
import numpy as np
import json
from pycocotools import mask as maskUtils
from tqdm import tqdm

# =================================================================
# [하드코딩 영역] 경로를 본인의 환경에 맞게 수정하세요.
# =================================================================
DIR_A = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector(copy)/ade20k/result/Figure/Figure_2/Figure_2_a'
DIR_B = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector(copy)/ade20k/result/Figure/Figure_2/Figure_2_b'
JSON_C = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector(copy)/ade20k/result/coco_pred_instances_origin.json'
JSON_D = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector(copy)/ade20k/result/coco_pred_instances_merged.json'

SAVE_ROOT = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector(copy)/ade20k/result/Figure/Figure_5'

EXCLUDE_IDS = [0, 8, 10, 11]  # 제외 및 무시 클래스 (이 영역은 흰색 배경으로 남음)
# =================================================================

# 색상 정보
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

ID2BGR = {c['id']: (c['color'][2], c['color'][1], c['color'][0]) for c in METAINFO}
EXCLUDE_BGR = [ID2BGR[cid] for cid in EXCLUDE_IDS if cid != 0]


def index_annotations(anns):
    """JSON 어노테이션을 image_id 기준으로 딕셔너리 인덱싱"""
    indexed = {}
    for ann in anns:
        img_id = str(ann.get('image_id'))
        if img_id not in indexed:
            indexed[img_id] = []
        indexed[img_id].append(ann)
    return indexed


def decode_to_img(indexed_anns, img_name, h, w):
    """특정 이미지에 대한 어노테이션을 흰색 배경에 그림"""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    stem = os.path.splitext(img_name)[0]

    target_anns = indexed_anns.get(img_name, indexed_anns.get(stem, []))

    if not target_anns:
        return canvas

    for ann in target_anns:
        cid = ann.get('category_id')
        if cid in EXCLUDE_IDS:
            continue

        seg = ann.get('segmentation')
        if seg:
            mask = maskUtils.decode(seg)
            if mask.ndim == 3: mask = mask[:, :, 0]
            canvas[mask > 0] = ID2BGR[cid]
    return canvas


def process_img_b(img_path):
    """이미지 B에서 제외된 클래스 색상을 검정색으로 필터링"""
    img = cv2.imread(img_path)
    if img is None: return None
    for bgr in EXCLUDE_BGR:
        mask = np.all(img == np.array(bgr, dtype=np.uint8), axis=-1)
        img[mask] = (0, 0, 0)
    return img


def make_figure_5():
    # 폴더 생성 (combined_diff 폴더 추가)
    for sub in ['a', 'b', 'c', 'd', 'combined', 'combined_diff']:
        os.makedirs(os.path.join(SAVE_ROOT, f'Figure_5_{sub}'), exist_ok=True)

    print("Loading and indexing JSON C...")
    with open(JSON_C, 'r') as f:
        data_c = json.load(f)
    anns_c = data_c.get('annotations', data_c) if isinstance(data_c, dict) else data_c
    indexed_c = index_annotations(anns_c)

    print("Loading and indexing JSON D...")
    with open(JSON_D, 'r') as f:
        data_d = json.load(f)
    anns_d = data_d.get('annotations', data_d) if isinstance(data_d, dict) else data_d
    indexed_d = index_annotations(anns_d)

    img_list = sorted([f for f in os.listdir(DIR_A) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    diff_count = 0  # 차이가 발견된 이미지 개수 카운트

    for img_name in tqdm(img_list, desc="Processing Figure 5"):
        # 1. Image A
        img_a = cv2.imread(os.path.join(DIR_A, img_name))
        if img_a is None: continue
        h, w = img_a.shape[:2]

        # 2. Image B
        img_b = process_img_b(os.path.join(DIR_B, img_name))
        if img_b is None: img_b = np.zeros_like(img_a)

        # 3. Image C (Origin)
        img_c = decode_to_img(indexed_c, img_name, h, w)

        # 4. Image D (Merged)
        img_d = decode_to_img(indexed_d, img_name, h, w)

        # 개별 저장
        cv2.imwrite(os.path.join(SAVE_ROOT, 'Figure_5_a', img_name), img_a)
        cv2.imwrite(os.path.join(SAVE_ROOT, 'Figure_5_b', img_name), img_b)
        cv2.imwrite(os.path.join(SAVE_ROOT, 'Figure_5_c', img_name), img_c)
        cv2.imwrite(os.path.join(SAVE_ROOT, 'Figure_5_d', img_name), img_d)

        # 5. Combined 1x4 저장
        sep_w = 10
        sep = np.full((h, sep_w, 3), 255, dtype=np.uint8)
        combined = np.hstack([img_a, sep, img_b, sep, img_c, sep, img_d])
        cv2.imwrite(os.path.join(SAVE_ROOT, 'Figure_5_combined', img_name), combined)

        # ---------------------------------------------------------
        # 6. C와 D가 다른 경우에만 'combined_diff' 폴더에 따로 저장
        # ---------------------------------------------------------
        difference = cv2.absdiff(img_c, img_d)
        non_zero_pixels = np.count_nonzero(difference)

        if non_zero_pixels > 0:
            cv2.imwrite(os.path.join(SAVE_ROOT, 'Figure_5_combined_diff', img_name), combined)
            diff_count += 1
            # 터미널에 차이가 발생한 파일과 픽셀 차이 개수를 출력
            tqdm.write(f"🔍 [Diff Found] {img_name} - 차이 픽셀 수: {non_zero_pixels}")

    print(f"\nSuccess! Results saved in {SAVE_ROOT}")
    print(f"총 {len(img_list)}개 이미지 중 Origin과 Merged가 다른 이미지는 {diff_count}개 입니다.")


if __name__ == "__main__":
    make_figure_5()