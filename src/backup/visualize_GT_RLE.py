import os
import cv2
import numpy as np
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import random


def visualize_merged_gt():
    # 1. 경로 설정
    test_path = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector(copy)/test'
    merged_json_path = os.path.join(test_path, 'result', 'merged_annotations.json')
    output_dir = os.path.join(test_path, 'result', 'merged_gt_random_visualized')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 2. 데이터 보정 및 COCO 로드
    # pycocotools의 KeyError: 'id' 문제를 해결하기 위해 메모리에서 ID를 강제 부여합니다.
    print(f"Loading and validating JSON from {merged_json_path}...")

    if not os.path.exists(merged_json_path):
        print(f"Error: 파일을 찾을 수 없습니다 -> {merged_json_path}")
        return

    try:
        with open(merged_json_path, 'r') as f:
            data = json.load(f)

        # 'images' 리스트 내의 id 누락 보정
        image_filename_map = {}
        for i, img in enumerate(data.get('images', [])):
            if 'id' not in img:
                img['id'] = i
            image_filename_map[img['file_name']] = img['id']

        # 'annotations' 리스트 내의 id 및 image_id 누락 보정
        for i, ann in enumerate(data.get('annotations', [])):
            if 'id' not in ann:
                ann['id'] = i
            # image_id가 없는 경우 file_name 등을 통해 매칭하거나 순서대로 부여 (여기선 인덱스 활용)
            if 'image_id' not in ann:
                # 만약 lane_merger에서 image_id를 누락했다면 기본적으로 0번 혹은 매칭 로직 필요
                # 여기서는 안전하게 0번 혹은 기존 인덱스에 맞춤
                ann['image_id'] = ann.get('image_id', 0)

        # 보정된 임시 JSON 저장
        fixed_json_path = merged_json_path.replace('.json', '_fixed_temp.json')
        with open(fixed_json_path, 'w') as f:
            json.dump(data, f)

        # 보정된 파일로 COCO 객체 생성
        coco = COCO(fixed_json_path)
    except Exception as e:
        print(f"JSON 로드 및 보정 중 오류 발생: {e}")
        return

    img_ids = coco.getImgIds()
    print(f"Processing {len(img_ids)} images...")

    # 3. 시각화 루프
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info['file_name']

        h = img_info.get('height', 1080)
        w = img_info.get('width', 1920)

        # 검은색 캔버스 생성
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            try:
                # RLE 디코딩하여 마스크 생성
                mask = coco.annToMask(ann)

                # 랜덤 BGR 색상 (배경과 구분되도록 밝은 색 위주)
                random_color = (
                    random.randint(80, 255),
                    random.randint(80, 255),
                    random.randint(80, 255)
                )

                canvas[mask > 0] = random_color
            except Exception as e:
                # 특정 어노테이션에서 에러 발생 시 건너뜀
                continue

        # 4. 결과 저장
        base_name = os.path.basename(file_name)
        # 확장자가 없을 경우를 대비해 .png 강제 지정 혹은 유지
        if not os.path.splitext(base_name)[1]:
            base_name += ".png"

        save_path = os.path.join(output_dir, base_name)
        cv2.imwrite(save_path, canvas)

    # 작업 완료 후 임시 파일 삭제 (선택 사항)
    if os.path.exists(fixed_json_path):
        os.remove(fixed_json_path)

    print(f"\nDone! Visualized images are saved in: {output_dir}")


if __name__ == '__main__':
    visualize_merged_gt()