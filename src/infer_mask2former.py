"""
Mask2Former 세그멘테이션 추론 → 순수 클래스-색상 마스크 생성 (infer_mask2former)

실행 환경(conda): mmseg  (torch 2.0+cu118, mmcv 2.0.0, mmdet 3.0.0, mmsegmentation 1.2.2)
config(.py, self-contained)와 best 체크포인트는 CKPT_ROOT 아래에 있다.

사용:
  cd src
  conda run -n mmseg python infer_mask2former.py --model mask2former_large
  conda run -n mmseg python infer_mask2former.py --model mask2former_large --validate
  conda run -n mmseg python infer_mask2former.py --model mask2former_small --splits test

출력: <DATA_ROOT>/mask2former/satellite_ade20k_250925_<model>/{pred_val,pred_test}/*.png

주의 — 클래스 인덱스 매핑:
  체크포인트 head는 150-class(스톡 ADE20K)이고 config에 reduce_zero_label=True가 있으나,
  기존 예측(prediction/*.png)은 배경이 검정(id 0)이라 모델 인덱스가 METAINFO id와 그대로
  일치하는 것으로 보인다(CLASS_OFFSET=0). test 예측을 신뢰하기 전에 반드시 --validate 로
  기존 val 예측을 재현하는지(픽셀 일치율≈1.0) 확인할 것. 어긋나면 --class-offset 1 로 시도.
"""

import os
import argparse

import numpy as np
from mmseg.apis import init_model, inference_model

import config as cfg
import infer_common

CKPT_ROOT = os.path.join(cfg.DATA_ROOT, 'mask2former', 'checkpoint')

# model_name -> (config/체크포인트 디렉토리, 체크포인트 파일)
MODELS = {
    'mask2former_large': ('mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640',
                          'best_mIoU_iter_160000.pth'),
    'mask2former_small': ('mask2former_swin-s_8xb2-160k_ade20k-512x512',
                          'best_mIoU_iter_160000.pth'),
}


def model_out_dir(model_name):
    return os.path.join(cfg.DATA_ROOT, 'mask2former', cfg.MODEL_PREFIX + model_name)


def build_infer_fn(model_name):
    cfg_dir, ckpt_file = MODELS[model_name]
    config_py = os.path.join(CKPT_ROOT, cfg_dir, cfg_dir + '.py')
    checkpoint = os.path.join(CKPT_ROOT, cfg_dir, ckpt_file)
    assert os.path.exists(config_py), f'config 없음: {config_py}'
    assert os.path.exists(checkpoint), f'checkpoint 없음: {checkpoint}'
    model = init_model(config_py, checkpoint, device='cuda:0')

    def infer_fn(img_path):
        result = inference_model(model, img_path)  # SegDataSample
        seg = result.pred_sem_seg.data.squeeze().cpu().numpy()
        return np.asarray(seg).astype(np.int32)

    return infer_fn


def main():
    ap = argparse.ArgumentParser(description='Mask2Former 추론 → pred_val/pred_test 마스크 생성')
    ap.add_argument('--model', default='mask2former_large', choices=list(MODELS))
    ap.add_argument('--splits', nargs='+', default=None, help='기본: config.EVAL_SPLITS')
    ap.add_argument('--class-offset', type=int, default=0,
                    help='METAINFO id = 모델인덱스 + offset. 기본 0(검증으로 확인)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--validate', action='store_true',
                    help='추론만 하지 않고 기존 prediction/ 대비 픽셀 일치율 검증')
    args = ap.parse_args()

    infer_fn = build_infer_fn(args.model)
    out_dir = model_out_dir(args.model)
    if args.validate:
        infer_common.validate_against_existing(infer_fn, out_dir, args.class_offset)
        return
    infer_common.run_inference(infer_fn, out_dir, args.splits, args.class_offset, args.overwrite)


if __name__ == '__main__':
    main()
