"""
Shared reader for the SEED vector label json (seed_label)

A SEED label json is a flat list of objects. The lane annotations this project evaluates are the
RoadObject / LINE_STRING entries whose category exists in config.METAINFO; everything else
(MetaData, POLYGON markings such as crosswalks and arrows, categories outside METAINFO) is
ignored by the pipeline and carried over verbatim when a new SEED revision is written.

The ADE20K rasterizer (make_seg_labels), the COCO builder (seed_to_coco) and the annotation
merger (merge_annotation) all load lanes through this module so that the three apply exactly
the same filtering rule.
"""

import os
import sys
import json
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg

NAME2ID = {c['name']: c['id'] for c in cfg.METAINFO}
MIN_POINTS = 2   # a polyline needs at least two points to be drawable


@dataclass
class SeedLane:
    """One lane polyline read from a SEED label json."""
    category_id: int
    category: str
    points: np.ndarray   # ordered image points (N, 2), float64
    source: dict         # the original SEED object (its attributes are carried over on rewrite)


def load_lane_objects(json_file: str):
    """Load the lane polylines of a SEED label json (RoadObject / LINE_STRING / METAINFO category)."""
    return [SeedLane(NAME2ID[obj['category']], obj['category'],
                     np.array(obj['image_points'], dtype=np.float64), obj)
            for obj in load_objects(json_file) if is_lane_object(obj)]


def load_objects(json_file: str):
    """Load every object of a SEED label json as-is."""
    with open(json_file, 'r') as f:
        return json.load(f)


def is_lane_object(obj: dict) -> bool:
    """True for the lane polylines the pipeline evaluates; all other objects are pass-through."""
    if obj.get('class') != 'RoadObject' or obj.get('geometry_type') != 'LINE_STRING':
        return False
    points = obj.get('image_points')
    if not points or len(points) < MIN_POINTS:
        return False
    return obj.get('category') in NAME2ID
