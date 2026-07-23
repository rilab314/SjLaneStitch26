"""
Build COCO instance-segmentation GT from SEED lane polylines (seed_to_coco)

Pure format conversion: every lane polyline of the SEED source is drawn with thickness
mask_thickness and stored as one RLE instance. No geometry is changed here.

Fragmented lanes are already merged in the SEED source (SEED_MAP_v1.1, produced by
dataprep/merge_annotation.py), so this step never touches the geometry.

    coco/annotations/instances_{train,validation,test}2017.json   lane instances (object F1 GT)
"""

import os
import sys
import json
from typing import List

import cv2
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg
import seed_label


class CocoInstanceBuilder:
    """Convert the SEED lane polylines of one split into a COCO instance-segmentation json."""
    mask_thickness = 3          # polyline rendering thickness of an instance mask (px)
    default_shape = (768, 768)  # canvas used when the satellite image cannot be read

    def __init__(self, split: str, image_ids: List[str], label_dir: str,
                 image_dir: str, coco_path: str):
        self._split = split
        self._image_ids = image_ids
        self._label_dir = label_dir
        self._image_dir = image_dir
        self._coco_path = coco_path
        os.makedirs(os.path.dirname(coco_path), exist_ok=True)

    def run(self):
        images, annotations = [], []
        for base in tqdm(self._image_ids, desc=f'coco[{self._split}]'):
            json_file = os.path.join(self._label_dir, base + '.json')
            if not os.path.exists(json_file):
                continue
            lanes = seed_label.load_lane_objects(json_file)
            if not lanes:
                continue
            shape = self._image_shape(base)
            images.append(make_image_entry(base, shape))
            annotations += self._annotations_of(lanes, base, shape)
        save_coco(self._coco_path, images, annotations)
        print(f'[coco] split={self._split}: images={len(images)}, annotations={len(annotations)}')

    def _image_shape(self, base: str):
        """(height, width) of the satellite image; the default canvas if it cannot be read."""
        image = cv2.imread(os.path.join(self._image_dir, base + '.png'))
        return self.default_shape if image is None else image.shape[:2]

    def _annotations_of(self, lanes: List[seed_label.SeedLane], image_id: str, shape):
        anns = [make_annotation(lane.points, lane.category_id, image_id, shape,
                                self.mask_thickness) for lane in lanes]
        return [ann for ann in anns if ann is not None]


def make_image_entry(base: str, shape) -> dict:
    """COCO images[] entry. The image id is the basename string, as in the original GT."""
    height, width = shape
    return {'license': 1, 'file_name': base + '.png', 'coco_url': '',
            'height': int(height), 'width': int(width), 'date_captured': '',
            'flickr_url': '', 'id': base}


def make_annotation(points: np.ndarray, category_id: int, image_id: str, shape,
                    thickness: int) -> dict:
    """COCO annotations[] entry holding the polyline band as an RLE mask (None if empty)."""
    rle = encode_polyline(points, shape, thickness)
    if rle is None:
        return None
    return {'image_id': image_id, 'category_id': category_id,
            'segmentation': rle, 'score': 1.0}


def encode_polyline(points: np.ndarray, shape, thickness: int) -> dict:
    """Draw the polyline with the given thickness and RLE-encode it (None if nothing is drawn)."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    polyline = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask, [polyline], isClosed=False, color=1, thickness=thickness)
    if mask.sum() == 0:
        return None
    rle = maskUtils.encode(np.asfortranarray(mask))
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def save_coco(coco_path: str, images: list, annotations: list):
    """Write the COCO json (same header/categories as the original merged_annotations.json)."""
    coco = {
        'info': {'contributor': '', 'date_created': '2024/12/13', 'description': '',
                 'url': '', 'version': '1.0', 'year': 2024},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': [{'id': c['id'], 'name': c['name'], 'supercategory': 'segmentation'}
                       for c in cfg.METAINFO],
    }
    with open(coco_path, 'w') as f:
        json.dump(coco, f)
    print(f'[save] coco json saved: {coco_path} (images={len(images)}, '
          f'annotations={len(annotations)}, categories={len(coco["categories"])})')


def count_class_instances(splits: List[str], csv_path: str):
    """Read the per-split COCO json, compute per-class instance counts and save them as csv.

    Rows: class (name), columns: split. A total row/column is appended at the end."""
    id2name = {c['id']: c['name'] for c in cfg.METAINFO}
    class_ids = [c['id'] for c in cfg.METAINFO if c['id'] != 0]  # skip ignore

    counts = {sp: {cid: 0 for cid in class_ids} for sp in splits}
    for split in splits:
        path = cfg.coco_anno_path(split)
        if not os.path.exists(path):
            print(f'[count] {path} not found -> skipping {split}')
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        for ann in data['annotations']:
            if ann['category_id'] in counts[split]:
                counts[split][ann['category_id']] += 1

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write('class_id,class_name,' + ','.join(splits) + ',total\n')
        for cid in class_ids:
            row = [counts[sp][cid] for sp in splits]
            f.write(f'{cid},{id2name[cid]},' + ','.join(map(str, row)) + f',{sum(row)}\n')
        totals = [sum(counts[sp].values()) for sp in splits]
        f.write('-,total,' + ','.join(map(str, totals)) + f',{sum(totals)}\n')
    print(f'[count] per-class instance counts csv saved: {csv_path}')


def main():
    """Regenerate the COCO instance GT of the evaluation splits from the configured SEED source."""
    with open(cfg.DATASET_SPLIT_JSON, 'r') as f:
        dataset = json.load(f)

    for split in cfg.EVAL_SPLITS:
        CocoInstanceBuilder(split=split, image_ids=sorted(dataset[split]),
                            label_dir=cfg.SRC_LABEL_DIR, image_dir=cfg.SRC_IMAGE_DIR,
                            coco_path=cfg.coco_anno_path(split)).run()
    count_class_instances(cfg.EVAL_SPLITS, os.path.join(cfg.COCO_PATH, 'class_counts.csv'))


if __name__ == '__main__':
    main()
