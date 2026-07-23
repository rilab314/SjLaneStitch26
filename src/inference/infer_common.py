"""
Common utilities for segmentation inference (infer_common)

Parts shared by the InternImage (.venv-internimage, mmseg 0.x) and Mask2Former (.venv-mask2former,
mmseg 1.x) inference scripts — see README §3.2 / §3.3:
  - per-split list of original images (dataset.json split lists + built ade20k images)
  - model class index map -> pure class-color mask (BGR) conversion (lane_stitcher input format)
  - iterate over splits and save to <model>/pred_val, <model>/pred_test
  - validate the color mapping against the existing pred_val/ folder (the original val inference results)

Color mapping note: for the class index i the model outputs, METAINFO id = i + class_offset.
In the existing predictions (pred_val/*.png) the background is black (id 0), so offset=0 is the default for both models.
If unsure for a new model/config, first confirm with validate_against_existing that it reproduces the existing val
predictions, then trust the test predictions.
"""

import os
import json

import cv2
import numpy as np
from tqdm import tqdm

import config as cfg


def split_basenames(split):
    """Read the sorted list of basenames for the split (train/validation/test) from dataset.json."""
    with open(cfg.DATASET_SPLIT_JSON, 'r') as f:
        return sorted(json.load(f)[split])


def colorize(seg, class_offset=0):
    """Model class index map (HxW) -> pure class-color BGR mask.

    METAINFO id = seg + class_offset. id 0 (ignore/background) is left black."""
    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ids = seg.astype(np.int64) + class_offset
    for cid, bgr in cfg.ID2BGR.items():
        if cid == 0:
            continue
        out[ids == cid] = bgr
    return out


def run_inference(infer_fn, model_out_dir, splits=None, class_offset=0, overwrite=False):
    """Infer original images per split and save the pure color masks to pred_val/pred_test.

    infer_fn(img_path) -> HxW class index map (np.ndarray)."""
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
            img_path = os.path.join(cfg.image_dir(split), b + '.png')
            if not os.path.exists(img_path):
                missing += 1
                continue
            seg = infer_fn(img_path)
            cv2.imwrite(dst, colorize(seg, class_offset))
            made += 1
        print(f'[infer] split={split}: saved {made}, skipped(existing) {skipped}, missing images {missing} -> {out_dir}')


def validate_against_existing(infer_fn, model_out_dir, class_offset=0, split='validation',
                              n=20, existing_dirname='pred_val'):
    """Validate via pixel agreement whether the inference results reproduce the existing val predictions (<model>/pred_val/*.png).

    Used to confirm that the color index mapping (class_offset) and preprocessing pipeline match the original.
    An agreement close to 1.0 means the mapping is correct."""
    existing_dir = os.path.join(model_out_dir, existing_dirname)
    if not os.path.isdir(existing_dir):
        print(f'[validate] no existing prediction folder: {existing_dir} -> skipping validation')
        return
    bases = split_basenames(split)[:n]
    agree_sum = 0.0
    cnt = 0
    for b in bases:
        ref_path = os.path.join(existing_dir, b + '.png')
        img_path = os.path.join(cfg.image_dir(split), b + '.png')
        if not (os.path.exists(ref_path) and os.path.exists(img_path)):
            continue
        ref = cv2.imread(ref_path)
        gen = colorize(infer_fn(img_path), class_offset)
        if ref.shape != gen.shape:
            gen = cv2.resize(gen, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
        agree_sum += float(np.mean(np.all(ref == gen, axis=-1)))
        cnt += 1
    if cnt:
        print(f'[validate] {split} {cnt} images: mean pixel agreement vs existing predictions: {agree_sum / cnt:.4f} '
              f'(closer to 1.0 = more accurate mapping)')
    else:
        print('[validate] no images to compare')
