import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import random
import config as cfg


def visualize_gt_instances():
    # 1. 경로 설정 (config.py 파일 기준)
    coco_anno_path = cfg.COCO_ANNO_PATH
    # 결과물을 저장할 폴더 생성
    output_dir = os.path.join(cfg.RESULT_PATH, 'gt_instance_random_colors')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. COCO 데이터 로드
    print(f"Loading GT from {coco_anno_path}...")
    coco = COCO(coco_anno_path)

    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids):
        # 이미지 정보 로드
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info['file_name']
        h, w = img_info['height'], img_info['width']

        # 검은색 캔버스 생성
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # 해당 이미지의 모든 어노테이션(객체) 가져오기
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        # 3. 각 객체마다 랜덤 색상으로 그리기
        for ann in anns:
            # 개별 객체를 마스크(0과 1)로 변환
            mask = coco.annToMask(ann)

            # 랜덤 BGR 색상 생성 (너무 어둡지 않게 100~255 범위 권장)
            random_color = (
                random.randint(50, 255),  # Blue
                random.randint(50, 255),  # Green
                random.randint(50, 255)  # Red
            )

            # 마스크 영역에 색상 입히기
            canvas[mask > 0] = random_color

        # 4. 결과 저장
        save_path = os.path.join(output_dir, os.path.basename(file_name))
        cv2.imwrite(save_path, canvas)


if __name__ == '__main__':
    visualize_gt_instances()