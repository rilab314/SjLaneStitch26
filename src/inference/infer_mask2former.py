"""
Mask2Former segmentation inference -> generate pure class-color masks (infer_mask2former)

Run environment (conda): mmseg  (torch 2.0+cu118, mmcv 2.0.0, mmdet 3.0.0, mmsegmentation 1.2.2)
The config (.py, self-contained) and best checkpoint are under CKPT_ROOT.

Usage:
  cd src
  conda run -n mmseg python infer_mask2former.py --model mask2former_large
  conda run -n mmseg python infer_mask2former.py --model mask2former_large --validate
  conda run -n mmseg python infer_mask2former.py --model mask2former_small --splits test

Output: <DATA_ROOT>/mask2former/satellite_ade20k_250925_<model>/{pred_val,pred_test}/*.png

Note - class index mapping:
  The checkpoint head is 150-class (stock ADE20K) and the config has reduce_zero_label=True, but
  in the existing predictions (prediction/*.png) the background is black (id 0), so the model index
  appears to match the METAINFO id directly (CLASS_OFFSET=0). Before trusting the test predictions,
  always confirm with --validate that it reproduces the existing val predictions (pixel agreement ~1.0).
  If it does not match, try --class-offset 1.
"""

import os
import sys
import argparse

import numpy as np
from mmseg.apis import init_model, inference_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
import infer_common

CKPT_ROOT = os.path.join(cfg.DATA_ROOT, 'mask2former', 'checkpoint')

# model_name -> (config/checkpoint directory, checkpoint file)
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
    assert os.path.exists(config_py), f'config not found: {config_py}'
    assert os.path.exists(checkpoint), f'checkpoint not found: {checkpoint}'
    model = init_model(config_py, checkpoint, device='cuda:0')

    def infer_fn(img_path):
        result = inference_model(model, img_path)  # SegDataSample
        seg = result.pred_sem_seg.data.squeeze().cpu().numpy()
        return np.asarray(seg).astype(np.int32)

    return infer_fn


def main():
    ap = argparse.ArgumentParser(description='Mask2Former inference -> generate pred_val/pred_test masks')
    ap.add_argument('--model', default='mask2former_large', choices=list(MODELS))
    ap.add_argument('--splits', nargs='+', default=None, help='default: config.EVAL_SPLITS')
    ap.add_argument('--class-offset', type=int, default=0,
                    help='METAINFO id = model index + offset. default 0 (confirmed by validation)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--validate', action='store_true',
                    help='Instead of only inferring, validate pixel agreement against the existing prediction/')
    args = ap.parse_args()

    infer_fn = build_infer_fn(args.model)
    out_dir = model_out_dir(args.model)
    if args.validate:
        infer_common.validate_against_existing(infer_fn, out_dir, args.class_offset)
        return
    infer_common.run_inference(infer_fn, out_dir, args.splits, args.class_offset, args.overwrite)


if __name__ == '__main__':
    main()
