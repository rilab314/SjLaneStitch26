"""
Generate ADE20K index segmentation labels (PNG) from SEED vector labels (make_seg_labels)

Pixel-wise mIoU evaluation uses ADE20K-format index label PNGs as GT. Originally only the validation
labels (ade20k/annotations/validation) exist and test labels do not, so both val and test are freshly
rasterized from SEED json with the same rules into the result folder (cfg.label_dir(split)).

Encoding/rendering is matched **exactly to the original ADE20K generator** (reproducing the existing validation labels):
  - pixel value = METAINFO class id + 1  (background/road = 1, center_line = 2, ...)
    The evaluator shifts by -1 when reading the GT, so follow this +1 rule as-is.
  - lane width: not a cv2 pen thickness but expand the center line into a polygon via
    **shapely buffer (radius 1.5px, round cap)** and fill it with cv2.fillPoly. It is a "constant-width band"
    with uniform width even on curved segments, so it cannot be reproduced by polylines pen thickness that
    bulges the area at vertices (GT-GT IoU vs the original is capped at ~97%). The buffer method is
    effectively a 100% match with the original.
  - overlap resolution: the draw order is set by class priority (config_converter.ADE20K_LANE_CATEGORIES).
    A smaller priority number is drawn on top (e.g. stop_line topmost, center_line bottommost).

With these rules, the val labels reproduce the existing ade20k labels and test is rendered with exactly the same rules.
"""

import os
import sys
import json

import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import LineString

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

import config as cfg
import seed_label


class SegLabelRasterizer:
    buffer_size = 1.5    # center-line left/right expansion radius (px). Same as the original generator (shapely buffer)
    cap_style = 'round'  # line-end handling (same as the original)
    default_size = 768   # default canvas size used when the image cannot be read
    # class draw priority (priority of the original config_converter.ADE20K_LANE_CATEGORIES, id->priority).
    # A smaller priority number is drawn on top (wins on overlap). ignore(0) is not drawn.
    CLASS_PRIORITY = {1: 10, 2: 6, 3: 7, 4: 3, 5: 8, 6: 4, 7: 5, 8: 9, 9: 0, 10: 1, 11: 2}

    def __init__(self, split: str, image_ids, out_dir: str, label_dir: str, image_dir: str):
        self._split = split
        self._image_ids = image_ids
        self._out_dir = out_dir
        self._label_dir = label_dir
        self._image_dir = image_dir
        os.makedirs(out_dir, exist_ok=True)

    def run(self):
        made = 0
        skipped = 0
        for base in tqdm(self._image_ids, desc=f'seg-label[{self._split}]'):
            seed = os.path.join(self._label_dir, base + '.json')
            if not os.path.exists(seed):
                skipped += 1
                continue
            label = self._rasterize(seed, self._image_size(base))
            cv2.imwrite(os.path.join(self._out_dir, base + '.png'), label)
            made += 1
        print(f'[seg-label] split={self._split}: created {made}, skipped {skipped} -> {self._out_dir}')

    def _image_size(self, base: str):
        img_file = os.path.join(self._image_dir, base + '.png')
        img = cv2.imread(img_file)
        if img is None:
            return self.default_size, self.default_size
        return img.shape[0], img.shape[1]

    def _rasterize(self, seed_json: str, hw) -> np.ndarray:
        h, w = hw
        label = np.ones((h, w), dtype=np.uint8)  # background/road = 1
        lanes = seed_label.load_lane_objects(seed_json)
        # Draw classes with larger priority first so that classes with smaller (=higher) priority remain on top
        # (same overlap-resolution rule as the original ADE20K generator).
        lanes.sort(key=lambda lane: self.CLASS_PRIORITY.get(lane.category_id, 0), reverse=True)
        for lane in lanes:
            for polygon in self._line_to_polygons(lane.points):
                cv2.fillPoly(label, [polygon], lane.category_id + 1)
        return label

    def _line_to_polygons(self, pts: np.ndarray):
        """Convert the center line into a list of polygons (integer exterior coordinates) expanded by buffer_size radius.
        Identical to the original generator's expand_line_to_polygon (shapely buffer + fillPoly)."""
        buffered = LineString(pts).buffer(self.buffer_size, cap_style=self.cap_style)
        if buffered.is_empty:
            return []
        geoms = [buffered] if buffered.geom_type == 'Polygon' else list(buffered.geoms)
        polygons = []
        for geom in geoms:
            ext = [[int(round(x)), int(round(y))] for x, y in geom.exterior.coords]
            polygons.append(np.array(ext, dtype=np.int32))
        return polygons


def main():
    with open(cfg.DATASET_SPLIT_JSON, 'r') as f:
        dataset = json.load(f)

    for split in cfg.EVAL_SPLITS:
        rasterizer = SegLabelRasterizer(
            split=split,
            image_ids=sorted(dataset[split]),
            out_dir=cfg.label_dir(split),
            label_dir=cfg.SRC_LABEL_DIR,
            image_dir=cfg.SRC_IMAGE_DIR,
        )
        rasterizer.run()


if __name__ == '__main__':
    main()
