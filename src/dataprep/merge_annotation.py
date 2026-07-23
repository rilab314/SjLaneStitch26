"""
Merge the fragmented SEED lane annotations: SEED_MAP_v1.0 -> SEED_MAP_v1.1 (merge_annotation)

WHAT THIS SCRIPT IS FOR
-----------------------
The raw SEED release (SEED_MAP_v1.0) stores what is visually one lane as several fragmented
polyline objects. This script repairs that fragmentation once, at the source, and writes the
result as a new SEED revision with the identical file format:

    SEED_MAP_v1.0/label/*.json   raw lanes (fragmented)
            |  lane_merger.LaneMerger  (dedup -> trim -> endpoint merge)
    SEED_MAP_v1.1/label/*.json   one polyline per lane, same SEED schema

**SEED_MAP_v1.1 is the source dataprep/build_dataset.py converts**, so the dataset build itself
is a pure format conversion and the semantic (ADE20K) and instance (COCO) GT describe the same
geometry. This script therefore only has to run when the raw SEED release changes — the released
data already ships the merged revision.

Everything that is not an evaluated lane polyline (MetaData, POLYGON markings such as crosswalks
and arrows, categories outside METAINFO) is copied over verbatim. `image/` is symlinked rather
than duplicated, and `dataset.json` is copied unchanged.

Format notes
  - image_points hold the merged geometry as floats (trim resamples along the polyline); the raw
    revision stored integers. Consumers round when rasterizing, so the built datasets are unchanged.
  - global_points are re-derived from the tile's own image<->global pairs (affine fit), because
    merged points exist in no source object. They are kept for format compatibility only;
    the pipeline reads image_points.

Usage (from src/):
    python dataprep/merge_annotation.py                          # all splits, parallel
    python dataprep/merge_annotation.py --split validation       # one split
    python dataprep/merge_annotation.py --jobs 1 --overwrite     # single process, rewrite existing
    python dataprep/merge_annotation.py --count-only             # report raw vs merged lane counts
    python dataprep/merge_annotation.py --src <dir> --dst <dir>  # non-default SEED locations
"""

import os
import sys
import json
import shutil
import argparse
import multiprocessing as mp

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg
import seed_label
from lane_merger import LaneMerger

DEFAULT_JOBS = max(1, (os.cpu_count() or 2) // 2)
_WRITER = {}   # per-worker SeedLabelWriter (multiprocessing initializer)


class MergedSeedBuilder:
    """Create the merged SEED revision (label/ merged, image/ symlinked, dataset.json copied)."""

    def __init__(self, src_path: str, dst_path: str, splits, jobs: int = 1,
                 overwrite: bool = False):
        self._src_path = src_path
        self._dst_path = dst_path
        self._splits = splits
        self._jobs = max(1, jobs)
        self._overwrite = overwrite
        os.makedirs(os.path.join(dst_path, 'label'), exist_ok=True)

    def run(self):
        self._link_images()
        self._copy_split_json()
        with open(os.path.join(self._src_path, 'dataset.json'), 'r') as f:
            dataset = json.load(f)
        for split in self._splits:
            self._build_split(split, sorted(dataset[split]))

    def _link_images(self):
        """Point the merged revision at the raw images (identical content, not worth copying)."""
        link = os.path.join(self._dst_path, 'image')
        if os.path.lexists(link):
            return
        os.symlink(os.path.relpath(os.path.join(self._src_path, 'image'), self._dst_path), link)
        print(f'[link] {link} -> {os.readlink(link)}')

    def _copy_split_json(self):
        """dataset.json (per-split basename lists) is unchanged between the two revisions."""
        dst = os.path.join(self._dst_path, 'dataset.json')
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(self._src_path, 'dataset.json'), dst)
            print(f'[copy] dataset.json -> {dst}')

    def _build_split(self, split: str, bases):
        print(f"\n{'='*70}\nmerged SEED label build  split={split}  ({len(bases)} images)  "
              f"jobs={self._jobs}\n{'='*70}")
        counts = self._convert_all(split, bases)
        raw = sum(c[0] for c in counts)
        merged = sum(c[1] for c in counts)
        reduced = (1 - merged / raw) * 100 if raw else 0.0
        print(f'[merge] split={split}: lanes {raw} -> {merged} ({reduced:.1f}% reduced)')

    def _convert_all(self, split: str, bases):
        args = (self._src_path, self._dst_path, self._overwrite)
        desc = f'merge[{split}]'
        if self._jobs == 1:
            _init_worker(*args)
            return [_convert_one(base) for base in tqdm(bases, desc=desc)]
        with mp.Pool(self._jobs, initializer=_init_worker, initargs=args) as pool:
            return list(tqdm(pool.imap(_convert_one, bases, chunksize=8),
                             total=len(bases), desc=desc))


def _init_worker(src_path: str, dst_path: str, overwrite: bool):
    """Give each worker process its own writer and keep OpenCV single-threaded."""
    cv2.setNumThreads(1)
    _WRITER['writer'] = SeedLabelWriter(src_path, dst_path, overwrite)


def _convert_one(base: str):
    return _WRITER['writer'].write(base)


class SeedLabelWriter:
    """Convert one SEED label json from the raw revision to the merged revision."""

    def __init__(self, src_path: str, dst_path: str, overwrite: bool = False):
        self._src_label_dir = os.path.join(src_path, 'label')
        self._dst_label_dir = os.path.join(dst_path, 'label')
        self._overwrite = overwrite
        self._merger = LaneMerger(
            split='merged-seed', image_ids=[], label_path=self._src_label_dir,
            image_path=os.path.join(src_path, 'image'), write_compare=False)

    def write(self, base: str):
        """Write the merged label json of one image and return (raw lanes, merged lanes)."""
        src_file = os.path.join(self._src_label_dir, base + '.json')
        dst_file = os.path.join(self._dst_label_dir, base + '.json')
        if not os.path.exists(src_file):
            return 0, 0
        if os.path.exists(dst_file) and not self._overwrite:
            return 0, 0

        raw_lanes, merged_lanes = self._merger.merge_image(base)
        objects = seed_label.load_objects(src_file)
        mapper = GlobalPointMapper(objects)
        used_ids = set()
        kept = [obj for obj in objects if not seed_label.is_lane_object(obj)]
        lanes = [self._lane_object(lane, mapper, used_ids) for lane in merged_lanes]
        with open(dst_file, 'w') as f:
            json.dump(kept + lanes, f)
        return len(raw_lanes), len(merged_lanes)

    def _lane_object(self, lane, mapper, used_ids: set) -> dict:
        """SEED object for a merged lane: source attributes with the new geometry."""
        points = np.asarray(lane.points, dtype=np.float64)
        obj = dict(lane.source)   # keeps the original key order and attributes (category/type/...)
        obj['id'] = unique_object_id(lane.source.get('id'), used_ids)
        obj['image_points'] = points.tolist()
        obj['global_points'] = mapper(points)
        return obj


def unique_object_id(object_id, used_ids: set) -> str:
    """Reuse the SEED object id, suffixing duplicates (trim can split one source line in pieces)."""
    root = object_id if object_id else 'MERGED'
    candidate, serial = root, 1
    while candidate in used_ids:
        serial += 1
        candidate = f'{root}-{serial}'
    used_ids.add(candidate)
    return candidate


class GlobalPointMapper:
    """Map image pixel coordinates to lon/lat (EPSG:4326) for one tile.

    Merged polylines contain points that exist in no source object, so their global coordinates
    are re-derived with an affine fitted on every (image_points, global_points) pair of the same
    tile (residual ~1 px, i.e. the quantization of the integer image_points themselves)."""
    min_pairs = 3   # an affine needs three non-collinear correspondences

    def __init__(self, objects):
        self._matrix = self._fit(objects)

    def __call__(self, points: np.ndarray):
        """Global (lon, lat) list for the given image points ([] when no fit was possible)."""
        if self._matrix is None:
            return []
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        return (homogeneous @ self._matrix).tolist()

    @classmethod
    def _fit(cls, objects):
        image_points, global_points = [], []
        for obj in objects:
            image, glob = obj.get('image_points'), obj.get('global_points')
            if not image or not glob or len(image) != len(glob):
                continue
            image_points.append(np.asarray(image, dtype=np.float64))
            global_points.append(np.asarray(glob, dtype=np.float64))
        if not image_points:
            return None
        source, target = np.vstack(image_points), np.vstack(global_points)
        if len(source) < cls.min_pairs:
            return None
        homogeneous = np.hstack([source, np.ones((len(source), 1))])
        return np.linalg.lstsq(homogeneous, target, rcond=None)[0]


def write_compare_images(src_path: str, splits, compare_dir: str):
    """Render before|after overlays of the merge, to inspect the result visually."""
    with open(os.path.join(src_path, 'dataset.json'), 'r') as f:
        dataset = json.load(f)
    for split in splits:
        LaneMerger(split=split, image_ids=sorted(dataset[split]),
                   label_path=os.path.join(src_path, 'label'),
                   image_path=os.path.join(src_path, 'image'),
                   compare_path=os.path.join(compare_dir, split)).run()


def count_before_after(splits, src_path: str, dst_path: str):
    """Print the lane-object counts of each split before and after merging."""
    with open(os.path.join(dst_path, 'dataset.json'), 'r') as f:
        dataset = json.load(f)

    label_dirs = (os.path.join(src_path, 'label'), os.path.join(dst_path, 'label'))
    print(f"\n{'split':12}{'raw':>10}{'merged':>10}{'reduced':>10}{'reduce%':>9}")
    totals = [0, 0]
    for split in splits:
        counts = [0, 0]
        for base in sorted(dataset[split]):
            for i, label_dir in enumerate(label_dirs):
                json_file = os.path.join(label_dir, base + '.json')
                if os.path.exists(json_file):
                    counts[i] += len(seed_label.load_lane_objects(json_file))
        totals = [t + c for t, c in zip(totals, counts)]
        print(_count_row(split, counts))
    print(_count_row('TOTAL', totals))


def _count_row(name: str, counts) -> str:
    raw, merged = counts
    percent = (1 - merged / raw) * 100 if raw else 0.0
    return f'{name:12}{raw:>10}{merged:>10}{raw - merged:>10}{percent:>8.1f}%'


def main():
    parser = argparse.ArgumentParser(
        description='Merge the fragmented SEED lane annotations into a new SEED revision')
    parser.add_argument('--src', default=cfg.RAW_SEED_PATH,
                        help=f'raw SEED release to read (default: {cfg.RAW_SEED_PATH})')
    parser.add_argument('--dst', default=cfg.SEED_SOURCE_PATH,
                        help=f'merged SEED revision to write (default: {cfg.SEED_SOURCE_PATH})')
    parser.add_argument('--split', nargs='+', default=list(cfg.ALL_SPLITS), choices=cfg.ALL_SPLITS,
                        help='splits to convert (default: all -> train validation test)')
    parser.add_argument('--jobs', type=int, default=DEFAULT_JOBS,
                        help=f'worker processes (default: {DEFAULT_JOBS})')
    parser.add_argument('--overwrite', action='store_true',
                        help='rewrite label json files that already exist')
    parser.add_argument('--count-only', action='store_true',
                        help='only report the raw vs merged lane counts of the two revisions')
    parser.add_argument('--compare-dir',
                        help='write before/after overlay images of the merge into this folder '
                             'instead of converting (visual inspection)')
    args = parser.parse_args()

    if args.compare_dir:
        write_compare_images(args.src, args.split, args.compare_dir)
        return

    if not args.count_only:
        if not os.path.isdir(os.path.join(args.src, 'label')):
            parser.error(f'raw SEED labels not found: {os.path.join(args.src, "label")}\n'
                         'The released dataset ships the merged revision only; point --src at '
                         'the raw SEED release to run this conversion.')
        MergedSeedBuilder(src_path=args.src, dst_path=args.dst, splits=args.split,
                          jobs=args.jobs, overwrite=args.overwrite).run()
    count_before_after(args.split, args.src, args.dst)


if __name__ == '__main__':
    main()
