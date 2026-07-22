# LaneStitch — Vectorized Lane Marking Detection and Refinement From Satellite Imagery

Reference implementation for the paper:

> **Vectorized Lane Marking Detection and Refinement From Satellite Imagery for HD Maps**
> Seongjun Youn, Hae Min Cho, Hyukdoo Choi
> *Submitted to IEEE Access — currently under review.*

> ⚠️ **Status:** The manuscript is under peer review. Details (numbers, class set,
> parameters) may still change with revisions. This repository is tagged **v1**,
> matching the submitted version.

---

## 1. What this project does

High-Definition (HD) maps are essential for autonomous driving but expensive to build
with Mobile Mapping Systems. Satellite imagery is a scalable, inherently georeferenced
alternative, but a plain semantic segmentation model alone yields **fragmented masks,
collapses double markings into single strokes, and produces no instance-level objects**.

**LaneStitch** is a post-processing pipeline that turns a segmentation mask into
**vectorized, continuous, instance-level lane marking polylines** suitable for HD-map
generation. It only *trims* and *chains* existing points — it never averages or reorders
them — so it corrects segmentation defects without introducing the zigzag that blending
parallel/overlapping lines would create.

On the SEED-MAP validation split, the post-processing raises the object-level
**F1@0.5 from 29.4 to 40.2** while keeping pixel-level **mIoU around 36–37** —
i.e. it restores object continuity without sacrificing coverage.

### Pipeline (`core/lane_stitcher.py`)

1. **Semantic Segmentation** — a standard model (Mask2Former or InternImage) labels each
   pixel with one of the lane marking classes. Trained externally; this repo consumes its
   color-coded prediction PNGs.
2. **Blob-to-Linestring Vectorization** — each connected blob is thinned to a 1-pixel
   skeleton (Zhang–Suen) and traced into an ordered polyline with a curvature-aware tracer;
   a residual pass recovers lines missed in the first pass (e.g. the opposite rail of a
   double center line).
3. **Linestring Refinement** — overlapping parallel center lines (from double markings) are
   trimmed so one representative line remains, while genuinely diverging branches are kept
   as separate objects.
4. **Linestring Merging (stitch)** — fragmented linestrings are chained end to end into
   longer, continuous lanes, with a parallel-rejection rule so the two rails of a double
   marking are not fused.

### Evaluation

- **Object-level F1@0.5** — each lane marking is an instance; a prediction is a true positive
  when it overlaps a GT instance with IoU ≥ 0.5 (greedy 1:1 matching). Per-class precision/recall
  are aggregated over the whole split into F1; the overall score is the macro average over the
  nine evaluated classes (`core/evaluator.py`).
- **Pixel-level mIoU** — mean IoU over class pixels, measuring how much of the marked area is
  covered (independent of how pixels group into objects).

### Dataset — SEED-MAP

12,828 satellite images at 768×768 (8,979 train / 1,282 validation / 2,567 test) with
eleven lane marking classes. Nine are evaluated (`guiding_line` and `safety_zone` are
excluded, as they don't admit a clean linestring representation). Annotations are
consolidated (duplicates collapsed, fragments chained) before evaluation.

The dataset is released separately:
<https://github.com/rilab314/SatelliteLaneDataset2024>.

---

## 2. Repository layout

The whole experiment is reproducible from a raw SEED source in five steps
(config → build dataset → inference → experiments → tables/figures). Every downstream
script reads from two built datasets — `ade20k/` (semantic seg) and `coco/` (instance
seg) — so there is a single, non-duplicated ground truth. **All commands run from `src/`.**

```
src/
  config.py            # all paths & class metadata — EDIT BEFORE RUNNING (not committed)
  config-template.py   # template to copy from when writing config.py
  _bootstrap.py        # puts core/, tables/, figures/ on sys.path (imported by run scripts)

  core/                # shared libraries (imported, not run directly)
    lane_stitcher.py       # LaneStitcher: segmentation -> polyline vectorization & merge
    evaluator.py           # object F1 (greedy IoU matching) + mIoU evaluation
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
    run_parallel_sweep.py  # same sweep, fanned out across processes (deterministic, faster)
    run_best_experiment.py # single run with the best config

  tables/              # paper tables (Table 1..5) + shared helper
    num_params.py  table_1.py .. table_5.py  table_common.py (lib)

  figures/             # paper figures (Figure 1..8) + shared render/metrics libs
    figure_1.py .. figure_8.py  figure_base.py figure_render.py figure_metrics.py figure_match.py

InternImage/           # InternImage segmentation model tree (incl. ops_dcnv3 CUDA op)
```

### Data model (no duplication)

```
DATA_ROOT/
  satellite_good_matching_250206/   RAW SEED SOURCE (only build_dataset.py reads it)
    image/*.png  label/*.json  dataset.json            # 12828 images, per-split basename lists

  ade20k/                           ADE20K semantic-seg dataset  (built)
    images/{training,validation,test}/*.png
    annotations/{training,validation,test}/*.png       # index labels, pixel = class_id + 1  (mIoU GT)
    color_annotations/{training,validation,test}/*.png # color visualization labels

  coco/                             COCO instance-seg dataset    (built)
    annotations/instances_{train,validation,test}2017.json   # merged lane GT  (object F1)
    {train2017,val2017,test2017}/*.png                       # images

  Internimage/  mask2former/        model predictions <model>/{pred_val,pred_test}/*.png
  results_<date>/                   one run's outputs (prediction JSON, CSV, Table, Figure)
```

`config.image_dir / label_dir / color_label_dir / coco_anno_path / coco_image_dir` map a
split to the paths above; every script uses those helpers, so there is exactly one copy of
each label/GT and relocating `DATA_ROOT` only touches `config.py`.

---

## 3. Requirements & installation

**Python 3.9** is recommended.

### Post-processing pipeline (this repo)

```bash
pip install -r requirements.txt        # scikit-image, opencv, scipy, numpy, ...
pip install pycocotools pandas shapely # additional pipeline/evaluation deps
```

`requirements.txt` pins the vectorization/thinning stack (scikit-image, opencv, scipy,
networkx, …). The evaluator and table/figure scripts additionally use `pycocotools`,
`pandas`, and `shapely`.

### Segmentation inference (optional, separate environment)

Running the segmentation models (`inference/infer_*.py`) needs a GPU and an
**mmsegmentation** environment with `torch`, `mmcv`, and `mmseg`, plus the compiled
**DCNv3** CUDA operator for InternImage. Build it once:

```bash
cd InternImage/segmentation/ops_dcnv3
sh make.sh                             # python setup.py build install
```

Inference is **not required to reproduce the tables/figures** — the pipeline consumes the
model prediction PNGs, which are deterministic and can be reused as-is. Only re-run
inference if you retrain a segmentation model or add a new one.

---

## 4. Setup — write `config.py`

`config.py` is **not committed** (it holds machine-specific absolute paths). Copy the
template and edit it:

```bash
cd src
cp config-template.py config.py
```

Key fields:

| field | meaning |
|-------|---------|
| `DATA_ROOT`        | root that holds the datasets and model outputs |
| `SRC_DATASET_PATH` | raw SEED source (`image/`, `label/`, `dataset.json`) |
| `DATASET_PATH`     | built ADE20K dataset (`DATA_ROOT/ade20k`) |
| `COCO_PATH`        | built COCO dataset (`DATA_ROOT/coco`) |
| `RESULT_DIR`       | output folder name — **change per run** (e.g. `results_260709`) |
| `EVAL_SPLITS`      | splits to evaluate (`validation`, `test`) |

Class metadata (`METAINFO`, `EXCLUDE_IDS`, `EVAL_CLASS_IDS`, …), the object-metric config
(`F1_IOUS`, `F1_PRIMARY`), and the split-name maps (`ADE_SPLIT_DIR`, `COCO_IMG_DIR`) are
also in `config.py`.

---

## 5. Running the pipeline

All commands run from `src/`.

### 5.1 Build the datasets (once per SEED source)

```bash
python dataprep/build_dataset.py               # all splits -> ade20k/ + coco/
#   --split validation test     # only some splits
#   --skip images               # regenerate labels/color/coco only (e.g. after a rule change)
```

This copies images, rasterizes the ADE20K index + color labels, and builds the merged COCO
instance GT (writing `class_counts.csv` at the end).

### 5.2 Segmentation inference → `<model>/pred_val`, `<model>/pred_test`

```bash
python inference/infer_internimage.py     # env: internimage (mmseg 0.x)
python inference/infer_mask2former.py      # env: mmseg (mmseg 1.x)
```

Reads images from `ade20k/images/<split>`. Deterministic — reuse existing predictions to
reproduce results without re-running.

### 5.3 Stitching pipeline + evaluation → `RESULT_PATH`

The sweep searches hyper-parameters on validation, then evaluates the best params on test.

```bash
# single process (~48 min/combo, ~20 h for the full sweep):
python experiment/run_experiments.py --split validation test

# same result, fanned out across processes (recommended for a full re-run):
MAXJOBS=14 python experiment/run_parallel_sweep.py

# other entry points:
python experiment/run_experiments.py --fast        # skip windows/collages
python experiment/run_experiments.py --eval-only    # recompute F1 only from existing predictions
python experiment/run_best_experiment.py            # single run with the best config
```

Both drivers write `total_performance.csv` (and per-combo `eval_result.csv`) into
`RESULT_PATH`. The evaluator caches the EXCLUDE_IDS-filtered COCO GT next to the coco GT
(`..._selected.json`) and the per-model segmentation mIoU in
`<model>/metrics_{split}.json` — delete these to force a clean recompute.

### 5.4 Tables → `RESULT_PATH/Tables/*.csv`

```bash
python tables/num_params.py    # model parameter counts (Table 1 Params column)
python tables/table_1.py       # model comparison (segmentation vs merge x1), val/test
python tables/table_2.py       # best model, per-class performance
python tables/table_3.py       # best model, per-class diagnostic breakdown
python tables/table_4.py       # stage-wise gains (baseline->residual->refinement->merge1->merge2)
python tables/table_5.py       # parameter ablation (stride/extend/turn)
```

### 5.5 Figures → `RESULT_PATH/Figure/*`

```bash
python figures/figure_1.py     # ... through figure_8.py
```

---

## 6. Notes

- Directly launched scripts whose base name is a plain noun are prefixed `run_` / `build_`
  (`run_experiments`, `run_parallel_sweep`, `build_dataset`); others already read as actions
  or numbered artifacts (`infer_*`, `merge_*`, `make_*`, `table_N`, `figure_N`).
- `core/evaluator.py` and `core/lane_stitcher.py` have a `main()`, but those are smoke tests
  only — the real entry point is `experiment/run_experiments.py`.
- `config.py`, results/CSV/JSON outputs, and DCNv3 build artifacts are git-ignored.
