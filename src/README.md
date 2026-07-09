# LaneStitch — source layout & reproduction guide

Pipeline that vectorizes satellite-image segmentation into lane polylines, stitches
fragmented lines into whole lanes, and evaluates them (COCO AP + pixel mIoU) for the
paper's tables and figures.

The whole experiment is reproducible from a raw SEED source in five steps
(config → build dataset → inference → experiments → tables/figures). Every downstream
script reads from two built datasets — `ade20k/` (semantic seg) and `coco/` (instance
seg) — so there is a single, non-duplicated ground truth.

## Directory layout

```
src/
  config.py            # all paths & class metadata — EDIT BEFORE RUNNING
  config-template.py   # template to copy from when writing a new config
  _bootstrap.py        # puts core/, tables/, figures/ on sys.path (imported by run scripts)

  core/                # shared libraries (imported, not run directly)
    lane_stitcher.py       # LaneStitcher: segmentation -> polyline vectorization & merge
    evaluator.py           # COCO AP + mIoU evaluation
    stitch_config.py       # best-config loader from total_performance.csv
    util.py  show_imgs.py

  dataprep/            # build the datasets / ground truth
    build_dataset.py       # *** main builder: SEED source -> ade20k/ + coco/ (all splits) ***
    make_seg_labels.py     # (library + partial regen) SEED -> ade20k index labels
    merge_annotation.py    # (library + partial regen) SEED -> coco merged instance GT

  inference/           # run segmentation models -> per-model pred_val/ pred_test/ PNGs
    infer_internimage.py  infer_mask2former.py  infer_common.py (lib)

  experiment/          # run the stitching pipeline & evaluate
    run_experiments.py     # hyper-parameter sweep (val) + fixed-param eval (test)
    run_parallel_sweep.py  # same sweep, fanned out across processes (deterministic, much faster)
    run_best_experiment.py # single run with the best config

  tables/              # paper tables (Table 1..5) + shared helper
    num_params.py table_1.py .. table_5.py  table_common.py (lib)

  figures/             # paper figures (Figure 1..8) + shared render/metrics libs
    figure_1.py .. figure_8.py  figure_base.py figure_render.py figure_metrics.py figure_match.py (lib)
```

## Data model (no duplication)

```
DATA_ROOT/
  satellite_good_matching_250206/   RAW SEED SOURCE (only build_dataset.py reads it)
    image/*.png  label/*.json  dataset.json          # 12828 images, per-split basename lists

  ade20k/                           ADE20K semantic-seg dataset  (built)
    images/{training,validation,test}/*.png
    annotations/{training,validation,test}/*.png       # index labels, pixel = class_id + 1  (mIoU GT)
    color_annotations/{training,validation,test}/*.png # color visualization labels

  coco/                             COCO instance-seg dataset    (built)
    annotations/instances_{train,validation,test}2017.json   # merged lane GT  (COCO AP)
    {train2017,val2017,test2017}/*.png                       # images

  Internimage/  mask2former/        model predictions <model>/{pred_val,pred_test}/*.png
  results_<date>/                   one run's outputs (prediction JSON, CSV, Table, Figure)
```

`config.image_dir / label_dir / color_label_dir / coco_anno_path / coco_image_dir` map a
split to the paths above; every script uses those helpers, so there is exactly one copy
of each label/GT and moving `DATA_ROOT` only touches `config.py`.

## 1. Setup — edit `config.py` (see `config-template.py`)

- `DATA_ROOT`       — root that holds the datasets and model outputs
- `SRC_DATASET_PATH`— raw SEED source (`image/`, `label/`, `dataset.json`)
- `DATASET_PATH`    — built ADE20K dataset (`DATA_ROOT/ade20k`)
- `COCO_PATH`       — built COCO dataset (`DATA_ROOT/coco`)
- `RESULT_DIR`      — output folder name, **changed per run** (e.g. `results_260709`)
- `EVAL_SPLITS`     — splits to evaluate (`validation`, `test`)

Class metadata (`METAINFO`, `EXCLUDE_IDS`, …) and the split-name maps
(`ADE_SPLIT_DIR`, `COCO_IMG_DIR`) are also in `config.py`.

## 2. Build the datasets (once per SEED source)

```bash
cd src
python dataprep/build_dataset.py               # all splits -> ade20k/ + coco/
# options:
#   --split validation test     # only some splits
#   --skip images               # regenerate labels/color/coco only (e.g. after a rule change)
```

This copies images, rasterizes the ADE20K index + color labels
(`make_seg_labels.SegLabelRasterizer`), and builds the merged COCO instance GT
(`merge_annotation.MergeAnnotator`) — writing `class_counts.csv` at the end.

## 3. Segmentation inference → `<model>/pred_val`, `<model>/pred_test`

```bash
python inference/infer_internimage.py     # env: internimage (mmseg 0.x)
python inference/infer_mask2former.py      # env: mmseg (mmseg 1.x)
```

Predictions read images from `ade20k/images/<split>` and are deterministic, so an
existing set can be reused as-is (no need to re-run for evaluation reproduction).

## 4. Stitching pipeline + evaluation → `RESULT_PATH`

The sweep searches params on validation, then evaluates the best params on test.

```bash
# straightforward, single process (~48 min/combo, ~20 h for the full 18+2+3 sweep):
python experiment/run_experiments.py --split validation test

# same result, fanned out across processes (recommended for a full re-run):
MAXJOBS=14 python experiment/run_parallel_sweep.py

# other entry points:
python experiment/run_experiments.py --fast        # skip windows/collages
python experiment/run_experiments.py --eval-only    # re-evaluate existing predictions (GT changed)
python experiment/run_best_experiment.py            # single run with the best config
```

Both drivers write `total_performance.csv` (and per-combo `eval_result.csv`) into
`RESULT_PATH`. The evaluator caches the EXCLUDE_IDS-filtered COCO GT next to the coco
GT (`..._selected.json`) and the per-model segmentation mIoU in `<model>/metrics_{split}.json`
(delete these to force a clean recompute).

## 5. Tables → `RESULT_PATH/Tables/*.csv`

```bash
python tables/num_params.py    # model parameter counts (Table 1 Params column)
python tables/table_1.py       # model comparison (segmentation vs merge x1), val/test
python tables/table_2.py       # best model, per-class performance
python tables/table_3.py       # best model, per-class diagnostic breakdown
python tables/table_4.py       # stage-wise gains (first->residual->refinement->merge1->merge2)
python tables/table_5.py       # parameter ablation (stride/extend/turn)
```

## 6. Figures → `RESULT_PATH/Figure/*`

```bash
python figures/figure_1.py     # ... through figure_8.py
```

---

Naming convention: directly launched scripts whose base name is a plain noun are prefixed
`run_` / `build_` (`run_experiments`, `run_parallel_sweep`, `build_dataset`); the others
already read as actions or numbered artifacts (`infer_*`, `merge_*`, `make_*`, `table_N`,
`figure_N`). `core/evaluator.py` and `core/lane_stitcher.py` have a `main()` too, but those
are smoke tests only.
