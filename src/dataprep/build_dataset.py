"""
Build the complete ade20k + coco datasets from the SEED source (build_dataset)

Single entry point that turns the SEED source configured in config.SRC_DATASET_PATH
(SEED_MAP_v1.1: label/ + dataset.json, plus image/ when the release ships it) into the two
datasets every downstream script reads from:

  ade20k/  (ADE20K semantic segmentation)
    images/{training,validation,test}/*.png             copied satellite images
    annotations/{training,validation,test}/*.png        index labels (mIoU GT, pixel = class_id + 1)
    color_annotations/{training,validation,test}/*.png  color visualization labels (ID2BGR of the index)
  coco/    (COCO instance segmentation)
    annotations/instances_{train,validation,test}2017.json  lane instance GT (object F1)
    {train2017,val2017,test2017}/*.png                       satellite images (only with --coco-images)

This is **format conversion only** — no geometry is modified. Fragmented lanes are merged one
stage earlier, when the SEED revision itself is built (dataprep/merge_annotation.py), so both
outputs describe exactly the polylines stored in the source.

It reuses the existing building blocks instead of duplicating logic:
  - dataprep/make_seg_labels.SegLabelRasterizer  -> ade20k index labels
  - dataprep/seed_to_coco.CocoInstanceBuilder    -> coco lane instance GT
and adds two thin steps here: image copying and index->color colorization.

Run order (each step is idempotent; existing image copies are skipped):
    python dataprep/build_dataset.py                     # all splits, all steps
    python dataprep/build_dataset.py --split validation test
    python dataprep/build_dataset.py --skip images       # regenerate labels/color/coco only
    python dataprep/build_dataset.py --coco-images       # also fill coco/{split}2017 with images

When the SEED source ships no image/ folder, the images step is skipped automatically and the
canvas size is taken from the images already present in ade20k/images.

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
from seed_to_coco import CocoInstanceBuilder, count_class_instances

STEPS = ('images', 'labels', 'color', 'coco')


def copy_images(bases, dst_dirs, overwrite=False):
    """Copy SRC images for the given basenames into every destination directory.

    A destination that is a symlink (a build sharing the images of another build) is left alone."""
    dst_dirs = [d for d in dst_dirs if not _is_linked(d)]
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
          f'-> {", ".join(dst_dirs) if dst_dirs else "(all destinations are symlinks)"}')


def _is_linked(image_dir):
    """True when the directory (or a parent inside DATA_ROOT) is a symlink to another build."""
    path = os.path.abspath(image_dir)
    root = os.path.abspath(cfg.DATA_ROOT)
    while path.startswith(root) and path != root:
        if os.path.islink(path):
            return True
        path = os.path.dirname(path)
    return False


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


def source_image_dir(split):
    """Directory holding the satellite images of a split.

    The SEED release ships them in one flat image/ folder; when it does not (the published
    dataset carries the images only once, inside ade20k/), fall back to the built split folder."""
    return cfg.SRC_IMAGE_DIR if os.path.isdir(cfg.SRC_IMAGE_DIR) else cfg.image_dir(split)


def build_split(split, bases, steps, coco_images=False):
    print(f"\n{'='*70}\nBuilding split={split}  ({len(bases)} images)  steps={sorted(steps)}\n{'='*70}")
    image_dir = source_image_dir(split)
    if 'images' in steps:
        copy_images(bases, image_destinations(split, coco_images))
    if 'labels' in steps:
        SegLabelRasterizer(split=split, image_ids=bases, out_dir=cfg.label_dir(split),
                           label_dir=cfg.SRC_LABEL_DIR, image_dir=image_dir).run()
    if 'color' in steps:
        colorize_labels(split)
    if 'coco' in steps:
        CocoInstanceBuilder(
            split=split, image_ids=bases,
            label_dir=cfg.SRC_LABEL_DIR, image_dir=image_dir,
            coco_path=cfg.coco_anno_path(split),
        ).run()


def image_destinations(split, coco_images):
    """Where the satellite images are copied to. The COCO tree only needs them for a stand-alone
    COCO dataset (--coco-images); the pipeline itself reads images from ade20k/images."""
    dst_dirs = [cfg.image_dir(split)]
    if coco_images:
        dst_dirs.append(cfg.coco_image_dir(split))
    return dst_dirs


def main():
    parser = argparse.ArgumentParser(description='Build ade20k + coco datasets from the SEED source')
    parser.add_argument('--split', nargs='+', default=list(cfg.ALL_SPLITS), choices=cfg.ALL_SPLITS,
                        help='splits to build (default: all -> train validation test)')
    parser.add_argument('--skip', nargs='*', default=[], choices=list(STEPS),
                        help='steps to skip (images/labels/color/coco)')
    parser.add_argument('--overwrite-images', action='store_true',
                        help='re-copy images even if the destination already exists')
    parser.add_argument('--coco-images', action='store_true',
                        help='also copy the images into coco/{split}2017 (stand-alone COCO dataset)')
    args = parser.parse_args()

    steps = set(STEPS) - set(args.skip)
    if not os.path.isdir(cfg.SRC_IMAGE_DIR):
        steps -= {'images'}   # the SEED source ships no images: they are already in ade20k/images
    with open(cfg.DATASET_SPLIT_JSON, 'r') as f:
        dataset = json.load(f)

    for split in args.split:
        bases = sorted(dataset[split])
        if args.overwrite_images and 'images' in steps:
            copy_images(bases, image_destinations(split, args.coco_images), overwrite=True)
            build_split(split, bases, steps - {'images'}, args.coco_images)
        else:
            build_split(split, bases, steps, args.coco_images)

    if 'coco' in steps:
        csv_path = os.path.join(cfg.COCO_PATH, 'class_counts.csv')
        count_class_instances(args.split, csv_path)
    print('\nDataset build completed.')


if __name__ == '__main__':
    main()
