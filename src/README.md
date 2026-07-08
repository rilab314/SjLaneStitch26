# LaneStitch — source layout & run order

Pipeline that vectorizes satellite-image segmentation into lane polylines, stitches
fragmented lines into whole lanes, and evaluates them (COCO AP + pixel mIoU) for the
paper's tables and figures.

## Directory layout

```
src/
  config.py            # all paths & class metadata — EDIT BEFORE RUNNING
  config-template.py   # template to copy from when writing a new config
  _bootstrap.py        # puts core/, tables/, figures/ on sys.path (imported by run scripts)

  core/                # shared libraries (imported, not run directly)
    lane_stitcher.py       # LaneStitcher: segmentation -> polyline vectorization & merge
    evaluator.py           # COCO AP + mIoU evaluation
    baseline_opensatmap.py # OpenSatMap watershed baseline (reference comparison)
    stitch_config.py       # best-config loader from total_performance.csv
    util.py  show_imgs.py

  inference/           # run segmentation models -> per-model pred_val/ pred_test/ PNGs
    infer_internimage.py  infer_mask2former.py  infer_common.py (lib)

  dataprep/            # build ground truth for evaluation
    merge_annotation.py    # -> merged_annotations_{split}.json (COCO GT, for AP)
    make_seg_labels.py     # -> labels/{split}/*.png (index labels, for mIoU)

  experiment/          # run the stitching pipeline & evaluate
    run_experiments.py     # hyper-parameter sweep (val) + fixed-param eval (test)
    run_best_experiment.py # single run with the best config
    run_baseline.py        # OpenSatMap baseline on the best model's predictions

  tables/              # paper tables (Table 1..5) + shared helper
    num_params.py table_1.py .. table_5.py  table_common.py (lib)

  figures/             # paper figures (Figure 1..8) + shared render/metrics libs
    figure_1.py .. figure_8.py  figure_base.py figure_render.py figure_metrics.py figure_match.py (lib)
```

Naming convention: files whose base name is a plain noun and are meant to be launched
directly are prefixed `run_` (`run_experiments`, `run_best_experiment`, `run_baseline`).
The other launchers already read as actions or as numbered artifacts
(`infer_*`, `merge_*`, `make_*`, `table_N`, `figure_N`), so they keep those names.
`core/evaluator.py` and `core/lane_stitcher.py` contain a `main()` too, but those are
smoke tests only — the real runs go through the scripts above.

## Setup

Edit `config.py` first (see `config-template.py`): `DATA_ROOT`, `RESULT_DIR`,
`SRC_DATASET_PATH` (images / SEED labels / `dataset.json` split lists), and the eval
`EVAL_SPLITS`. All commands below are run from the `src/` directory.

## Run order

```bash
cd src

# 1. Segmentation inference -> <model>/pred_val, <model>/pred_test (per model)
python inference/infer_internimage.py
python inference/infer_mask2former.py

# 2. Ground truth (both derived from the same SEED vector labels)
python dataprep/merge_annotation.py     # COCO GT: merged_annotations_{val,test}.json  (AP)
python dataprep/make_seg_labels.py      # index label PNGs: labels/{val,test}/          (mIoU)

# 3. Stitching pipeline + evaluation
#    - sweep params on validation, then evaluate best params on test
python experiment/run_experiments.py --split validation test
python experiment/run_experiments.py --fast          # skip windows/collages (fast eval)
python experiment/run_best_experiment.py             # single run with the best config
python experiment/run_baseline.py                    # OpenSatMap watershed baseline (comparison)

# 4. Tables -> RESULT_PATH/Tables/*.csv
python tables/num_params.py             # model parameter counts (Table 1 Params column)
python tables/table_1.py                # model comparison (segmentation vs merge x1), val/test
python tables/table_2.py                # best model, per-class performance
python tables/table_3.py                # best model, per-class diagnostic breakdown
python tables/table_4.py                # stage-wise gains (+ OpenSatMap baseline row if present)
python tables/table_5.py               # parameter ablation (stride/extend/turn)

# 5. Figures -> RESULT_PATH/Figure/*
python figures/figure_1.py             # ... through figure_8.py
```

Re-run only evaluation (predictions unchanged, e.g. after the GT changed):
`python experiment/run_experiments.py --eval-only`.
