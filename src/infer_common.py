"""
세그멘테이션 추론 공용 유틸 (infer_common)

InternImage(env: internimage, mmseg 0.x)와 Mask2Former(env: mmseg, mmseg 1.x)
추론 스크립트가 공유하는 부분:
  - split별 원본 이미지 목록(dataset.json + SRC_IMAGE_DIR)
  - 모델 클래스 인덱스맵 -> 순수 클래스-색상 마스크(BGR) 변환 (lane_stitcher 입력 형식)
  - split 순회하며 <model>/pred_val, <model>/pred_test 에 저장
  - 기존 prediction/ 폴더(원래 val 추론 결과)와 대조해 색상 매핑을 검증

색상 매핑 주의: 모델이 내는 클래스 인덱스 i 에 대해 METAINFO id = i + class_offset.
기존 예측(prediction/*.png)은 배경이 검정(id 0)이므로 두 모델 모두 offset=0 이 기본값이다.
새 모델/설정에서 확실치 않으면 먼저 validate_against_existing 로 기존 val 예측을 재현하는지
확인한 뒤 test 예측을 신뢰한다.
"""

import os
import json

import cv2
import numpy as np
from tqdm import tqdm

import config as cfg


def split_basenames(split):
    """dataset.json에서 split(train/validation/test)의 basename 목록(정렬)을 읽는다."""
    with open(cfg.DATASET_SPLIT_JSON, 'r') as f:
        return sorted(json.load(f)[split])


def colorize(seg, class_offset=0):
    """모델 클래스 인덱스맵(HxW) -> 순수 클래스-색상 BGR 마스크.

    METAINFO id = seg + class_offset. id 0(ignore/배경)은 검정으로 남긴다."""
    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ids = seg.astype(np.int64) + class_offset
    for cid, bgr in cfg.ID2BGR.items():
        if cid == 0:
            continue
        out[ids == cid] = bgr
    return out


def run_inference(infer_fn, model_out_dir, splits=None, class_offset=0, overwrite=False):
    """split별로 원본 이미지를 추론해 순수 색상 마스크를 pred_val/pred_test에 저장한다.

    infer_fn(img_path) -> HxW 클래스 인덱스맵(np.ndarray)."""
    splits = splits or cfg.EVAL_SPLITS
    for split in splits:
        out_dir = cfg.pred_path(model_out_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        bases = split_basenames(split)
        made = skipped = missing = 0
        for b in tqdm(bases, desc=f'infer[{split}]'):
            dst = os.path.join(out_dir, b + '.png')
            if os.path.exists(dst) and not overwrite:
                skipped += 1
                continue
            img_path = os.path.join(cfg.SRC_IMAGE_DIR, b + '.png')
            if not os.path.exists(img_path):
                missing += 1
                continue
            seg = infer_fn(img_path)
            cv2.imwrite(dst, colorize(seg, class_offset))
            made += 1
        print(f'[infer] split={split}: 저장 {made}, 건너뜀(기존) {skipped}, 이미지없음 {missing} -> {out_dir}')


def validate_against_existing(infer_fn, model_out_dir, class_offset=0, split='validation',
                              n=20, existing_dirname='prediction'):
    """추론 결과가 기존 val 예측(<model>/prediction/*.png)을 재현하는지 픽셀 일치율로 검증한다.

    색상 인덱스 매핑(class_offset)·전처리 파이프라인이 원래와 같은지 확인하는 용도.
    일치율이 1.0에 가까우면 매핑이 올바른 것이다."""
    existing_dir = os.path.join(model_out_dir, existing_dirname)
    if not os.path.isdir(existing_dir):
        print(f'[validate] 기존 예측 폴더 없음: {existing_dir} -> 검증 생략')
        return
    bases = split_basenames(split)[:n]
    agree_sum = 0.0
    cnt = 0
    for b in bases:
        ref_path = os.path.join(existing_dir, b + '.png')
        img_path = os.path.join(cfg.SRC_IMAGE_DIR, b + '.png')
        if not (os.path.exists(ref_path) and os.path.exists(img_path)):
            continue
        ref = cv2.imread(ref_path)
        gen = colorize(infer_fn(img_path), class_offset)
        if ref.shape != gen.shape:
            gen = cv2.resize(gen, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
        agree_sum += float(np.mean(np.all(ref == gen, axis=-1)))
        cnt += 1
    if cnt:
        print(f'[validate] {split} {cnt}장 기존예측 대비 평균 픽셀 일치율: {agree_sum / cnt:.4f} '
              f'(1.0에 가까울수록 매핑 정확)')
    else:
        print('[validate] 비교할 이미지가 없음')
