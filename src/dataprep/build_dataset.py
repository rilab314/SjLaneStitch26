"""
Build the complete ade20k + coco datasets from the raw SEED source (build_dataset)

Single entry point that turns the raw SEED source (satellite_good_matching_250206:
image/ + label/ + dataset.json) into two self-contained datasets that every
downstream script reads from:

  ade20k/  (ADE20K semantic segmentation)
    images/{training,validation,test}/*.png             copied satellite images
    annotations/{training,validation,test}/*.png        index labels (mIoU GT, pixel = class_id + 1)
    color_annotations/{training,validation,test}/*.png  color visualization labels (ID2BGR of the index)
  coco/    (COCO instance segmentation)
    annotations/instances_{train,validation,test}2017.json  merged lane GT (COCO AP)
    {train2017,val2017,test2017}/*.png                       copied satellite images

It reuses the existing building blocks instead of duplicating logic:
  - dataprep/make_seg_labels.SegLabelRasterizer  -> ade20k index labels
  - dataprep/merge_annotation.MergeAnnotator     -> coco merged instance GT
and adds two thin steps here: image copying and index->color colorization.

Run order (each step is idempotent; existing image copies are skipped):
    python dataprep/build_dataset.py                     # all splits, all steps
    python dataprep/build_dataset.py --split validation test
    python dataprep/build_dataset.py --skip images       # regenerate labels/color/coco only

The ade20k annotation split folder for 'train' is 'training' (ADE20K convention);
see config.ADE_SPLIT_DIR / COCO_IMG_DIR for the naming maps.
"""

import os
import sys
import glob
import json
import shutil
import argparse

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg
from make_seg_labels import SegLabelRasterizer
from merge_annotation import MergeAnnotator, count_class_instances

STEPS = ('images', 'labels', 'color', 'coco')


def copy_images(bases, dst_dirs, overwrite=False):
    """Copy SRC images for the given basenames into every destination directory."""
    for d in dst_dirs:
        os.makedirs(d, exist_ok=True)
    copied = skipped = missing = 0
    for b in tqdm(bases, desc='copy images'):
        src = os.path.join(cfg.SRC_IMAGE_DIR, b + '.png')
        if not os.path.exists(src):
            missing += 1
            continue
        for d in dst_dirs:
            dst = os.path.join(d, b + '.png')
            if os.path.exists(dst) and not overwrite:
                skipped += 1
                continue
            shutil.copyfile(src, dst)
            copied += 1
    print(f'[images] copied {copied}, skipped(existing) {skipped}, missing {missing} '
          f'-> {", ".join(dst_dirs)}')


def colorize_labels(split):
    """Build ade20k color_annotations from the index annotations via ID2BGR.

    The index label stores pixel value = class_id + 1, so the color of index v is
    ID2BGR[v - 1] (v=1 background -> (0,0,0)). Reproduces the original color labels."""
    src_dir = cfg.label_dir(split)
    dst_dir = cfg.color_label_dir(split)
    os.makedirs(dst_dir, exist_ok=True)
    lut = np.zeros((256, 3), dtype=np.uint8)  # index value -> BGR
    for c in cfg.METAINFO:
        lut[c['id'] + 1] = cfg.ID2BGR[c['id']]
    files = sorted(glob.glob(os.path.join(src_dir, '*.png')))
    for f in tqdm(files, desc=f'colorize[{split}]'):
        idx = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(dst_dir, os.path.basename(f)), lut[idx])
    print(f'[color] {len(files)} -> {dst_dir}')


def build_split(split, bases, steps):
    print(f"\n{'='*70}\nBuilding split={split}  ({len(bases)} images)  steps={sorted(steps)}\n{'='*70}")
    if 'images' in steps:
        copy_images(bases, [cfg.image_dir(split), cfg.coco_image_dir(split)])
    if 'labels' in steps:
        SegLabelRasterizer(split=split, image_ids=bases, out_dir=cfg.label_dir(split)).run()
    if 'color' in steps:
        colorize_labels(split)
    if 'coco' in steps:
        MergeAnnotator(
            split=split, image_ids=bases,
            label_path=cfg.SEED_LABEL_PATH, image_path=cfg.SRC_IMAGE_DIR,
            compare_path=cfg.merge_compare_dir(split), coco_path=cfg.coco_anno_path(split),
            write_compare=False,          # skip the 12k before/after overlay PNGs for a bulk build
        ).run()


def main():
    parser = argparse.ArgumentParser(description='Build ade20k + coco datasets from the SEED source')
    parser.add_argument('--split', nargs='+', default=list(cfg.ALL_SPLITS), choices=cfg.ALL_SPLITS,
                        help='splits to build (default: all -> train validation test)')
    parser.add_argument('--skip', nargs='*', default=[], choices=list(STEPS),
                        help='steps to skip (images/labels/color/coco)')
    parser.add_argument('--overwrite-images', action='store_true',
                        help='re-copy images even if the destination already exists')
    args = parser.parse_args()

    steps = set(STEPS) - set(args.skip)
    with open(cfg.DATASET_SPLIT_JSON, 'r') as f:
        dataset = json.load(f)

    for split in args.split:
        bases = sorted(dataset[split])
        if args.overwrite_images and 'images' in steps:
            copy_images(bases, [cfg.image_dir(split), cfg.coco_image_dir(split)], overwrite=True)
            build_split(split, bases, steps - {'images'})
        else:
            build_split(split, bases, steps)

    if 'coco' in steps:
        csv_path = os.path.join(cfg.COCO_PATH, 'class_counts.csv')
        count_class_instances(args.split, csv_path)
    print('\nDataset build completed.')


if __name__ == '__main__':
    main()
