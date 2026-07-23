"""
InternImage segmentation inference -> generate pure class-color masks (infer_internimage)

Run environment: the InternImage inference venv (see README §3.2 / requirements-internimage.txt)
  — torch 1.11+cu113, mmcv-full 1.5.0, mmseg 0.27.0, plus the compiled DCNv3 CUDA op.
Prerequisites:
  - Build the DCNv3 operator once (sources are in this repo):
      cd InternImage/segmentation/ops_dcnv3 && sh make.sh
  - The config (.py) and best checkpoint are under CKPT_ROOT below.

Usage (with .venv-internimage activated):
  cd src
  python inference/infer_internimage.py                 # val+test inference
  python inference/infer_internimage.py --validate      # validate reproduction of existing val predictions
  python inference/infer_internimage.py --splits test   # test only

Output: <DATA_ROOT>/Internimage/satellite_ade20k_250925_internimage_large/{pred_val,pred_test}/*.png

The InternImage config uses num_classes=12, reduce_zero_label=False + the ADE20K label id+1 rule, so
the model class = METAINFO id + 1. Therefore apply offset=-1 when colorizing
(CLASS_OFFSET=-1; confirmed to exactly reproduce the existing pred_val/ predictions).
"""

import os
import sys
import argparse

# add the InternImage segmentation toolkit (custom backbone/dataset, ops_dcnv3) to the import path
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEG_DIR = os.path.join(_PROJ, 'InternImage', 'segmentation')
sys.path.insert(0, _SEG_DIR)

import numpy as np
import mmcv_custom  # noqa: F401  (register backbone/op)
import mmseg_custom  # noqa: F401  (register SatelliteDataset/InternImage)
from mmseg.apis import init_segmentor, inference_segmentor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
import infer_common

CKPT_ROOT = os.path.join(cfg.DATA_ROOT, 'Internimage', 'checkpoint')
# The ADE20K labels are trained with id+1 (value 1 = background, reduce_zero_label=False), so the model class = id+1.
# Therefore METAINFO id = model class - 1 (offset=-1). Confirmed to exactly reproduce the existing prediction/ predictions.
CLASS_OFFSET = -1

# model_name -> (config/checkpoint directory, checkpoint file)
MODELS = {
    'internimage_large': ('upernet_internimage_l_512_160k_satellite', 'best_mIoU_iter_160000.pth'),
}


def model_out_dir(model_name):
    return os.path.join(cfg.DATA_ROOT, 'Internimage', cfg.MODEL_PREFIX + model_name)


def build_infer_fn(model_name):
    cfg_dir, ckpt_file = MODELS[model_name]
    config_py = os.path.join(CKPT_ROOT, cfg_dir, cfg_dir + '.py')
    checkpoint = os.path.join(CKPT_ROOT, cfg_dir, ckpt_file)
    assert os.path.exists(config_py), f'config not found: {config_py}'
    assert os.path.exists(checkpoint), f'checkpoint not found: {checkpoint}'
    model = init_segmentor(config_py, checkpoint, device='cuda:0')

    def infer_fn(img_path):
        result = inference_segmentor(model, img_path)  # [HxW ndarray]
        return np.asarray(result[0]).astype(np.int32)

    return infer_fn


def main():
    ap = argparse.ArgumentParser(description='InternImage inference -> generate pred_val/pred_test masks')
    ap.add_argument('--model', default='internimage_large', choices=list(MODELS))
    ap.add_argument('--splits', nargs='+', default=None, help='default: config.EVAL_SPLITS')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--validate', action='store_true',
                    help='Instead of only inferring, validate pixel agreement against the existing prediction/')
    args = ap.parse_args()

    infer_fn = build_infer_fn(args.model)
    out_dir = model_out_dir(args.model)
    if args.validate:
        infer_common.validate_against_existing(infer_fn, out_dir, CLASS_OFFSET)
        return
    infer_common.run_inference(infer_fn, out_dir, args.splits, CLASS_OFFSET, args.overwrite)


if __name__ == '__main__':
    main()
