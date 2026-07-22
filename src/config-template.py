import os

DATA_ROOT = "/path/to/2026_LaneStitch_deploy"

# ────────────────────────────────────────────────────────────────────── #
# Datasets (the single authoritative GT for every downstream script).
#   ade20k/ : ADE20K semantic-segmentation format
#       images/{training,validation,test}/*.png            satellite images
#       annotations/{training,validation,test}/*.png       index labels (mIoU GT, pixel = class_id + 1)
#       color_annotations/{training,validation,test}/*.png color visualization labels
#   coco/   : COCO instance-segmentation format
#       annotations/instances_{train,validation,test}2017.json  lane instance GT (object F1)
# ────────────────────────────────────────────────────────────────────── #
DATASET_PATH = DATA_ROOT + "/ade20k"   # ADE20K semantic-seg dataset
COCO_PATH = DATA_ROOT + "/coco"        # COCO instance-seg dataset

# Folder name for outputs (prediction JSON, CSV, Figure, Table). Change it per run to keep results separate.
RESULT_DIR = "results"
RESULT_PATH = os.path.join(DATA_ROOT, RESULT_DIR)

# ────────────────────────────────────────────────────────────────────── #
# SEED vector source. **Only dataprep/build_dataset.py and dataprep/merge_annotation.py read
# this**; no other script touches it.
#   label/        : SEED vector labels {basename}.json across all splits
#   dataset.json  : basename list per split (train/validation/test)
#   image/        : satellite images {basename}.png across all splits, when the release ships
#                   them; otherwise the build falls back to the images in ade20k/images.
#
# SEED_MAP_v1.1 holds one polyline per lane and is the revision build_dataset.py converts, so
# the build is a pure format conversion. It is produced from the raw SEED release
# (SEED_MAP_v1.0, fragmented polylines) by dataprep/merge_annotation.py — see RAW_SEED_PATH.
# ────────────────────────────────────────────────────────────────────── #
SEED_SOURCE_PATH = os.path.join(DATA_ROOT, "SEED_MAP_v1.1")  # merged SEED revision (build input)
RAW_SEED_PATH = os.path.join(DATA_ROOT, "SEED_MAP_v1.0")     # raw release (merge_annotation input)

SRC_DATASET_PATH = SEED_SOURCE_PATH
SRC_IMAGE_DIR = os.path.join(SRC_DATASET_PATH, "image")
SRC_LABEL_DIR = os.path.join(SRC_DATASET_PATH, "label")
DATASET_SPLIT_JSON = os.path.join(SRC_DATASET_PATH, "dataset.json")

# All splits the dataset builder produces vs the splits actually evaluated.
ALL_SPLITS = ['train', 'validation', 'test']
EVAL_SPLITS = ['validation', 'test']

# split (dataset.json key) -> folder/file naming in each dataset format.
#   ADE_SPLIT_DIR : ade20k images/annotations/color_annotations subfolder (train -> 'training')
#   COCO_IMG_DIR  : coco image subfolder (COCO 2017 convention)
# The coco annotation file is instances_{split}2017.json (split = dataset.json key).
ADE_SPLIT_DIR = {'train': 'training', 'validation': 'validation', 'test': 'test'}
COCO_IMG_DIR = {'train': 'train2017', 'validation': 'val2017', 'test': 'test2017'}

# Splits to evaluate and the split->model prediction folder name mapping.
# Model inference saves the pure class-color masks to <model>/pred_val, <model>/pred_test
# (the former single 'prediction' folder is split per split).
SPLIT_PRED_DIR = {'validation': 'pred_val', 'test': 'pred_test'}
# Short split label to append to output filenames and CSV columns (coco_pred_val_*, F1@0.5(val), etc.)
SPLIT_LABEL = {'validation': 'val', 'test': 'test'}


def pred_dirname(split):
    """split -> model prediction mask subfolder name (pred_val / pred_test)."""
    return SPLIT_PRED_DIR[split]


def split_label(split):
    """split -> short label used in output filenames/columns (validation->val, test->test)."""
    return SPLIT_LABEL[split]


def mcol(metric, split):
    """Per-split metric column name. e.g. mcol('F1@0.5','test') -> 'F1@0.5(test)'."""
    return f"{metric}({SPLIT_LABEL[split]})"


# ────────────────────────────────────────────────────────────────────── #
# Object metric (F1). The evaluator reports the object-level metric as
# macro-averaged F1 over EVAL_CLASS_IDS, one column per IoU threshold in
# F1_IOUS (replaces the old COCO AP10/AP20/AP50 columns).
# ────────────────────────────────────────────────────────────────────── #
F1_IOUS = [0.50]  # standard COCO-style matching threshold


def f1_metric(iou):
    """F1 column base name for an IoU threshold. e.g. 0.5 -> 'F1@0.5'."""
    return f"F1@{iou:g}"


F1_METRICS = [f1_metric(iou) for iou in F1_IOUS]  # all F1 column base names
F1_PRIMARY = F1_METRICS[0]  # operating point used to rank models/params ('F1@0.5')


def primary_metric_col(columns):
    """Column that ranks models/params: 'F1@0.5(val)', with fallbacks for suffix-less or legacy AP CSVs."""
    for col in (mcol(F1_PRIMARY, 'validation'), F1_PRIMARY, mcol('AP20', 'validation'), 'AP20'):
        if col in columns:
            return col
    raise KeyError(f"no object-metric column found among: {list(columns)}")


def pred_path(model_path, split):
    """Per-split prediction mask folder path under the model directory."""
    return os.path.join(model_path, SPLIT_PRED_DIR[split])


def image_dir(split):
    """Per-split satellite image directory in the built ade20k dataset."""
    return os.path.join(DATASET_PATH, "images", ADE_SPLIT_DIR[split])


def label_dir(split):
    """Per-split ADE20K index label (PNG) directory (mIoU GT). pixel = class_id + 1."""
    return os.path.join(DATASET_PATH, "annotations", ADE_SPLIT_DIR[split])


def color_label_dir(split):
    """Per-split ADE20K color visualization label directory (figures / imshow overlay)."""
    return os.path.join(DATASET_PATH, "color_annotations", ADE_SPLIT_DIR[split])


def coco_anno_path(split):
    """Per-split COCO instance GT (merged lanes) path in the built coco dataset."""
    return os.path.join(COCO_PATH, "annotations", f"instances_{split}2017.json")


def coco_image_dir(split):
    """Per-split satellite image directory in the built coco dataset (COCO 2017 layout)."""
    return os.path.join(COCO_PATH, COCO_IMG_DIR[split])


def split_result_path(split):
    """Algorithm result root. val/test are not split by path; they use the same RESULT_PATH.
    The two splits are distinguished only by filename (coco_pred_val_*/coco_pred_test_*) and CSV
    column suffix (…(val)/…(test)). (The argument is kept for caller compatibility, but the return
    value is identical regardless of split.)"""
    return RESULT_PATH


# ── validation default aliases for compatibility with existing scripts (Figure/Table, etc.) ── #
ANNO_DIR = RESULT_DIR
DATA_PATH = DATASET_PATH
LABEL_PATH = label_dir('validation')
COCO_MERGED_ANNO_PATH = coco_anno_path('validation')
COCO_ANNO_PATH = COCO_MERGED_ANNO_PATH

METAINFO = [
    {'id': 0, 'name': 'ignore', 'color': (0, 0, 0)},
    {'id': 1, 'name': 'center_line', 'color': (77, 77, 255)},
    {'id': 2, 'name': 'u_turn_zone_line', 'color': (77, 178, 255)},
    {'id': 3, 'name': 'lane_line', 'color': (77, 255, 77)},
    {'id': 4, 'name': 'bus_only_lane', 'color': (255, 153, 77)},
    {'id': 5, 'name': 'edge_line', 'color': (255, 77, 77)},
    {'id': 6, 'name': 'path_change_restriction_line', 'color': (178, 77, 255)},
    {'id': 7, 'name': 'no_parking_stopping_line', 'color': (77, 255, 178)},
    {'id': 8, 'name': 'guiding_line', 'color': (255, 178, 77)},
    {'id': 9, 'name': 'stop_line', 'color': (77, 102, 255)},
    {'id': 10, 'name': 'safety_zone', 'color': (255, 77, 128)},
    {'id': 11, 'name': 'bicycle_lane', 'color': (128, 255, 77)},
]
EXCLUDE_IDS = [0, 8, 10]   # bicycle_lane(11) included in evaluation
ID2BGR = {c['id']: (c['color'][2], c['color'][1], c['color'][0]) for c in METAINFO}
EVAL_CLASS_IDS = [c['id'] for c in METAINFO if c['id'] not in EXCLUDE_IDS]
ID2NAME = {c['id']: c['name'] for c in METAINFO}

RENDER_METAINFO = [
    {'id': 0, 'name': 'ignore', 'color': (0, 0, 0)},
    {'id': 1, 'name': 'center_line', 'color': (77, 77, 255)}, # original
    {'id': 2, 'name': 'u_turn_zone_line', 'color': (77, 178, 255)}, # original
    {'id': 3, 'name': 'lane_line', 'color': (77, 255, 77)}, # original
    {'id': 4, 'name': 'bus_only_lane', 'color': (255, 153, 77)}, # original
    {'id': 5, 'name': 'edge_line', 'color': (255, 77, 77)}, # original
    {'id': 6, 'name': 'path_change_restriction_line', 'color': (178, 77, 255)}, # original
    {'id': 7, 'name': 'no_parking_stopping_line', 'color': (77, 255, 178)}, # original
    {'id': 8, 'name': 'guiding_line', 'color': (255, 178, 77)}, # original
    {'id': 9, 'name': 'stop_line', 'color': (255, 215, 0)}, # Gold/Yellow color for high visual distinction against white background
    {'id': 10, 'name': 'safety_zone', 'color': (255, 77, 128)}, # original
    {'id': 11, 'name': 'bicycle_lane', 'color': (0, 139, 139)}, # Dark Cyan/Teal for high visual distinction against green and purple lines
]
RENDER_ID2BGR = {c['id']: (c['color'][2], c['color'][1], c['color'][0]) for c in RENDER_METAINFO}
MODEL_PREFIX = "satellite_ade20k_250925_"

# ────────────────────────────────────────────────────────────────────── #
# Published combination: the model and stitching hyperparameters selected on the validation
# split (highest F1@0.5) and reported in the paper. Used whenever no total_performance.csv of
# an own sweep is available, so a fresh checkout reproduces the published run directly.
# ────────────────────────────────────────────────────────────────────── #
BEST_MODEL = "mask2former_large"
BEST_PARAMS = {'thickness': 3, 'sample_stride': 5, 'extend_len': 20,
               'turn_penalty': 5.0, 'merge_count': 3}
