"""
InternImage 세그멘테이션 추론 → 순수 클래스-색상 마스크 생성 (infer_internimage)

실행 환경(conda): internimage  (torch 1.11+cu113, mmcv-full 1.5.0, mmseg 0.27.0, DCNv3)
사전 준비:
  - InternImage/segmentation/ops_dcnv3 폴더가 필요하다. 이 저장소엔 빠져 있으므로
    빌드된 rilab 저장소에서 복사한다(1회):
      cp -r /media/.../2025_LaneDetector_rilab/InternImage/segmentation/ops_dcnv3 \
            <project>/InternImage/segmentation/
  - config(.py)와 best 체크포인트는 아래 CKPT_ROOT 아래에 있다.

사용:
  cd src
  conda run -n internimage python infer_internimage.py                 # val+test 추론
  conda run -n internimage python infer_internimage.py --validate      # 기존 val 예측 재현 검증
  conda run -n internimage python infer_internimage.py --splits test   # test만

출력: <DATA_ROOT>/Internimage/satellite_ade20k_250925_internimage_large/{pred_val,pred_test}/*.png

InternImage 설정은 num_classes=12, reduce_zero_label=False + ADE20K 라벨 id+1 규칙이라
모델 클래스 = METAINFO id + 1 이다. 따라서 색상화 시 offset=-1 을 적용한다
(CLASS_OFFSET=-1; 기존 prediction/ 예측을 정확히 재현함을 확인).
"""

import os
import sys
import argparse

# InternImage 세그멘테이션 툴킷(커스텀 backbone/dataset, ops_dcnv3)을 import 경로에 추가
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEG_DIR = os.path.join(_PROJ, 'InternImage', 'segmentation')
sys.path.insert(0, _SEG_DIR)

import numpy as np
import mmcv_custom  # noqa: F401  (backbone/op 등록)
import mmseg_custom  # noqa: F401  (SatelliteDataset·InternImage 등록)
from mmseg.apis import init_segmentor, inference_segmentor

import config as cfg
import infer_common

CKPT_ROOT = os.path.join(cfg.DATA_ROOT, 'Internimage', 'checkpoint')
# ADE20K 라벨이 id+1(값1=배경, reduce_zero_label=False)로 학습돼 모델 클래스 = id+1 이다.
# 따라서 METAINFO id = 모델클래스 - 1 (offset=-1). 기존 prediction/ 예측을 정확히 재현함(검증됨).
CLASS_OFFSET = -1

# model_name -> (config/체크포인트 디렉토리, 체크포인트 파일)
MODELS = {
    'internimage_large': ('upernet_internimage_l_512_160k_satellite', 'best_mIoU_iter_160000.pth'),
}


def model_out_dir(model_name):
    return os.path.join(cfg.DATA_ROOT, 'Internimage', cfg.MODEL_PREFIX + model_name)


def build_infer_fn(model_name):
    cfg_dir, ckpt_file = MODELS[model_name]
    config_py = os.path.join(CKPT_ROOT, cfg_dir, cfg_dir + '.py')
    checkpoint = os.path.join(CKPT_ROOT, cfg_dir, ckpt_file)
    assert os.path.exists(config_py), f'config 없음: {config_py}'
    assert os.path.exists(checkpoint), f'checkpoint 없음: {checkpoint}'
    model = init_segmentor(config_py, checkpoint, device='cuda:0')

    def infer_fn(img_path):
        result = inference_segmentor(model, img_path)  # [HxW ndarray]
        return np.asarray(result[0]).astype(np.int32)

    return infer_fn


def main():
    ap = argparse.ArgumentParser(description='InternImage 추론 → pred_val/pred_test 마스크 생성')
    ap.add_argument('--model', default='internimage_large', choices=list(MODELS))
    ap.add_argument('--splits', nargs='+', default=None, help='기본: config.EVAL_SPLITS')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--validate', action='store_true',
                    help='추론만 하지 않고 기존 prediction/ 대비 픽셀 일치율 검증')
    args = ap.parse_args()

    infer_fn = build_infer_fn(args.model)
    out_dir = model_out_dir(args.model)
    if args.validate:
        infer_common.validate_against_existing(infer_fn, out_dir, CLASS_OFFSET)
        return
    infer_common.run_inference(infer_fn, out_dir, args.splits, CLASS_OFFSET, args.overwrite)


if __name__ == '__main__':
    main()
