import os
import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm
import config as cfg


class VisualComparator:
    """
    COCO 형식의 두 instance segmentation annotation 파일을 비교하여
    원본 이미지 위에 TP, FP, FN을 시각화하는 클래스입니다.
    """
    # 색상 상수를 클래스 변수로 정의하여 가독성 및 유지보수성 향상 (BGR 순서)
    COLORS = {
        "TP": [0, 255, 255],  # 노란색 (Yellow)
        "FP": [0, 0, 255],  # 빨간색 (Red)
        "FN": [255, 0, 0],  # 파란색 (Blue)
    }

    def compare(self, gt_anno_file: str, pred_anno_file: str, image_dir: str, save_path: str):
        """
        GT와 예측 annotation을 비교하여 원본 이미지 위에 결과를 오버레이하여 저장합니다.

        Args:
            gt_anno_file (str): GT annotation이 저장된 json 파일 경로
            pred_anno_file (str): 모델의 예측 결과가 저장된 json 파일 경로
            image_dir (str): 원본 이미지가 저장된 디렉토리 경로
            save_path (str): 결과 이미지를 저장할 디렉토리 경로
        """
        os.makedirs(save_path, exist_ok=True)
        try:
            gt_coco = COCO(gt_anno_file)
            pred_coco = gt_coco.loadRes(pred_anno_file)
        except Exception as e:
            print(f"Error: COCO annotation 파일을 로드하는 중 오류 발생: {e}")
            return

        image_ids = gt_coco.getImgIds()
        print(f"총 {len(image_ids)}개의 이미지에 대한 비교를 시작합니다.")

        for img_id in tqdm(image_ids, desc="Comparing masks"):
            img_info = gt_coco.loadImgs([img_id])[0]
            height, width, file_name = img_info['height'], img_info['width'], img_info['file_name']

            image_path = os.path.join(image_dir, file_name)
            background_img = cv2.imread(image_path)
            if background_img is None:
                print(f"Warning: 원본 이미지 '{image_path}'를 찾을 수 없습니다. 검은 배경으로 대체합니다.")
                background_img = np.zeros((height, width, 3), dtype=np.uint8)

            gt_mask = self._create_combined_mask(gt_coco, img_id, height, width)
            pred_mask = self._create_combined_mask(pred_coco, img_id, height, width)
            overlay_image = self._create_comparison_image(gt_mask, pred_mask)

            alpha = 0.6  # 원본 이미지의 가중치
            beta = 0.4  # 마스크 오버레이의 가중치
            gamma = 0
            blended_image = cv2.addWeighted(background_img, alpha, overlay_image, beta, gamma)

            output_file_path = os.path.join(save_path, file_name)
            cv2.imwrite(output_file_path, blended_image)
            # cv2.imshow('blended_image', blended_image)
            # cv2.waitKey(0)

        print("모든 이미지 비교 및 저장이 완료되었습니다.")

    def _create_combined_mask(self, coco_api: COCO, img_id: int, height: int, width: int) -> np.ndarray:
        """주어진 이미지 ID에 대한 모든 어노테이션을 합쳐 하나의 이진 마스크를 생성합니다."""
        ann_ids = coco_api.getAnnIds(imgIds=[img_id])
        anns = coco_api.loadAnns(ann_ids)
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            combined_mask = np.maximum(combined_mask, coco_api.annToMask(ann))
        return combined_mask

    def _create_comparison_image(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
        """두 마스크를 비교하여 TP, FP, FN을 시각화한 컬러 오버레이 이미지를 생성합니다."""
        height, width = gt_mask.shape
        overlay_image = np.zeros((height, width, 3), dtype=np.uint8)
        overlay_image[(gt_mask == 1) & (pred_mask == 1)] = self.COLORS["TP"]
        overlay_image[(gt_mask == 0) & (pred_mask == 1)] = self.COLORS["FP"]
        overlay_image[(gt_mask == 1) & (pred_mask == 0)] = self.COLORS["FN"]
        return overlay_image



def main():
    comparator = VisualComparator()
    comparator.compare(
        gt_anno_file=cfg.COCO_ANNO_PATH,
        pred_anno_file=cfg.MERGED_JSON_PATH,
        image_dir=cfg.ORIGIN_PATH,  # 원본 이미지 디렉토리 경로 전달
        save_path=os.path.join(cfg.RESULT_PATH, 'compare'),
    )

if __name__ == '__main__':
    main()
