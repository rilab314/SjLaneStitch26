"""OpenSatMap baseline(watershed) 후처리 재구현 — 리뷰 지적 M1(외부 baseline 정량 비교 부재) 방어용.

우리 best 모델(Mask2Former Swin-L)의 동일 segmentation 예측 위에서 watershed 인스턴스 분리 +
PCA 주축 정렬 벡터화만 적용한다. 우리 파이프라인의 강점(곡률 추적·단편 merge·이중선 trim·residual)은
의도적으로 배제해 순수 후처리 알고리즘 차이만 드러낸다.

I/O·래스터화(3px)·RLE 인코딩은 LaneStitcher를 컴포지션으로 재사용해 우리와 완전히 동일하게 맞춘다.
설계 근거: baseline_opensatmap_design.md (§2 의사코드, §6 권장 상수).
"""
import os
import json

import cv2
import numpy as np
from tqdm import tqdm

import config as cfg
from lane_stitcher import LaneStitcher, Strand, resample_polyline


class OpenSatMapBaseline:
    """seg 예측 → (watershed 인스턴스 분리 → PCA 정렬 벡터화) → COCO 예측 JSON.

    LaneStitcher 인스턴스 하나를 들고 _split_image_files·_read_image·_palette·convert_to_json을
    그대로 호출해 입력 이미지 목록·예측 로드·래스터화·인코딩을 우리 파이프라인과 동일하게 유지한다.
    baseline은 논문 상수만 쓰고 validation에서 재튜닝하지 않는다(공정성)."""

    watershed_alpha = 0.5   # sure_fg 임계 = alpha * dist.max() (OpenCV 튜토리얼 레시피, 논문 §4.1 인용)
    min_area = 100          # 인스턴스 최소 픽셀 수(<100px 제거, 논문 상수)

    def __init__(self, stitcher: LaneStitcher, sample_stride: int):
        self._stitcher = stitcher          # I/O·래스터화·인코딩 재사용(컴포지션)
        self._sample_stride = sample_stride  # 재샘플 간격(우리 best와 동일, §6-3)

    def run(self):
        """현재 split(validation) 전량을 처리해 baseline 예측 리스트를 반환한다."""
        files = self._stitcher._split_image_files()
        preds = []
        for file_name in tqdm(files, desc="OpenSatMap baseline"):
            image, pred_img, _ = self._stitcher._read_image(file_name)
            self._stitcher._img_shape = image.shape[:2]  # convert_to_json 래스터 캔버스 크기
            image_id = os.path.basename(file_name)[:-4]
            strands = self._extract(pred_img)
            preds += self._stitcher.convert_to_json(strands, image_id)
        return preds

    def run_and_save(self, save_path: str):
        """run() 결과를 JSON으로 저장하고 예측 리스트를 반환한다."""
        preds = self.run()
        with open(save_path, "w") as fp:
            json.dump(preds, fp)
        print(f"saved: {save_path} (instances={len(preds)})")
        return preds

    def _extract(self, pred_img: np.ndarray):
        """예측 이미지에서 평가 9클래스별 baseline strand 리스트를 만든다."""
        strands = []
        next_id = LaneStitcher.id_offset
        for class_id in cfg.EVAL_CLASS_IDS:
            color = self._stitcher._palette[class_id]  # 우리와 동일한 클래스-색 추출
            mask = np.all(pred_img == color, axis=-1).astype(np.uint8)
            for poly in self._vectorize_class(mask):
                strands.append(Strand(id=next_id, peak=(0, 0), class_id=class_id, points=poly))
                next_id += 1
        return strands

    def _vectorize_class(self, mask: np.ndarray):
        """한 클래스 이진 마스크 → watershed 인스턴스별 PCA 정렬·재샘플 폴리라인 리스트."""
        labels = self._watershed_labels(mask)
        polys = []
        for label in np.unique(labels):
            if label <= 1:  # 1=배경, -1=경계선
                continue
            instance = labels == label
            if int(instance.sum()) < self.min_area:  # 논문 상수: <100px 제거
                continue
            poly = self._order_and_resample(instance)
            if poly is not None:
                polys.append(poly)
        return polys

    def _watershed_labels(self, mask: np.ndarray) -> np.ndarray:
        """OpenCV 튜토리얼 레시피(논문 인용)로 인스턴스 라벨맵을 만든다.

        opening 노이즈 제거 → sure_bg(dilate) / sure_fg(distanceTransform>alpha·max) →
        unknown 영역 마커 0 → watershed. 얇은 선은 sure_fg가 중앙 리지만 남아 연결 블롭당
        한 인스턴스로 분리된다(이중선 등 붙은 블롭은 한 인스턴스로 남음 = baseline 한계 그대로)."""
        binary = (mask > 0).astype(np.uint8) * 255
        if binary.max() == 0:
            return np.ones(binary.shape, dtype=np.int32)  # 배경만
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        sure_bg = cv2.dilate(opening, kernel)
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, self.watershed_alpha * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown > 0] = 0
        return cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)

    def _order_and_resample(self, instance_mask: np.ndarray):
        """인스턴스 픽셀을 PCA 주축 투영으로 정렬하고 sample_stride로 균일 재샘플한다."""
        rows, cols = np.nonzero(instance_mask)
        if len(cols) < 2:
            return None
        pts = np.stack([cols, rows], axis=1).astype(np.float64)  # (P,2) = (x,y), 우리 점 규약과 동일
        axis = self._principal_axis(pts)
        order = np.argsort(pts @ axis)  # 주축 투영으로 점 정렬(곡률 처리 없음, baseline 순진 유지)
        poly = resample_polyline(pts[order], self._sample_stride)
        if len(poly) < 2:
            return None
        return np.rint(poly).astype(np.int32)

    @staticmethod
    def _principal_axis(pts: np.ndarray) -> np.ndarray:
        """PCA 첫 주성분(주축) 단위벡터."""
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt[0]
