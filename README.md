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

Reproducing the paper takes two steps — write `config.py` (§4), then run the experiments
(§5.3) and the tables/figures (§5.4–5.5). Building the datasets (§5.1) and running the
segmentation models (§5.2) is only needed if you change the data or retrain a model. Every
script reads its ground truth from the two datasets `ade20k/` (semantic seg) and `coco/`
(instance seg), so there is a single, non-duplicated GT. **All commands run from `src/`.**

```
src/
  config.py            # all paths & class metadata — EDIT BEFORE RUNNING (not committed)
  config-template.py   # template to copy from when writing config.py
  _bootstrap.py        # puts core/, tables/, figures/ on sys.path (imported by run scripts)

  core/                # shared libraries (imported, not run directly)
    lane_stitcher.py       # LaneStitcher: segmentation -> polyline vectorization & merge
    evaluator.py           # object F1 (greedy IoU matching) + mIoU evaluation
    stitch_config.py       # best-config loader (own sweep CSV, else the published combination)
    util.py  show_imgs.py

  dataprep/            # build the datasets / ground truth
    build_dataset.py       # *** main builder: SEED source -> ade20k/ + coco/ (all splits) ***
    merge_annotation.py    # raw SEED release -> merged SEED source (fragmented lanes chained)
    make_seg_labels.py     # (lib) SEED -> ade20k index labels
    seed_to_coco.py        # (lib) SEED -> coco instance GT
    seed_label.py  lane_merger.py   # (lib) SEED json reader / lane merging geometry

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

### Data bundle

Everything the pipeline reads and writes lives under one folder, published as
**`2026_LaneStitch_deploy.zip`** (link: _to be added_). Unpack it and point
`config.DATA_ROOT` at it:

```
DATA_ROOT/
  ade20k/                           ADE20K semantic-seg dataset — images + mIoU GT
    images/{training,validation,test}/*.png
    annotations/{training,validation,test}/*.png       # index labels, pixel = class_id + 1
    color_annotations/{training,validation,test}/*.png # color visualization labels

  coco/                             COCO instance-seg dataset — object F1 GT
    annotations/instances_{train,validation,test}2017.json
    class_counts.csv                                   # instances per split and class

  SEED_MAP_v1.1/                    SEED vector source (one polyline per lane)
    image/*.png  label/*.json  dataset.json            # 12828 images, per-split basename lists

  Internimage/  mask2former/        <model>/{pred_val,pred_test}/*.png + checkpoint/
  results/                          RESULT_DIR — prediction JSON, CSV, Tables, Figures
```

Both datasets are shipped ready to use, so **nothing has to be built to reproduce the
results**. They are derived from the SEED vector source: `dataprep/build_dataset.py`
converts `SEED_MAP_v1.1` into the ADE20K and COCO formats above (§5.1). The COCO part ships
annotations only — the pipeline reads its images from `ade20k/images`.

`config.image_dir / label_dir / color_label_dir / coco_anno_path / coco_image_dir` map a
split to the paths above; every script uses those helpers, so there is exactly one copy of
each label/GT and relocating `DATA_ROOT` only touches `config.py`.
See the bundle's own `README.md` for the details of the SEED revisions.

### Quick start — test the stitching algorithm only

The pipeline's input is the segmentation prediction masks, which ship with the bundle in
`<model>/pred_val` and `<model>/pred_test`. So after §3.1 and §4 you can run the algorithm
end to end without building datasets or running a segmentation model:

```bash
cd src
python experiment/run_best_experiment.py --split validation   # or --split test
```

---

## 3. Requirements & installation

Three independent environments, one requirements file each. **You only need the first one**
for everyday use — the two inference environments exist solely to regenerate prediction PNGs
and cannot be merged (see the note below). Plain `pip` + `venv` is enough; conda is not required.

| Environment | File | Python | Purpose |
|---|---|---|---|
| Post-processing pipeline | `requirements.txt` | 3.10 | stitching, evaluation, tables, figures (this repo) |
| InternImage inference | `requirements-internimage.txt` | 3.9 | (re)generate InternImage prediction PNGs — GPU, CUDA 11.3 |
| Mask2Former inference | `requirements-mask2former.txt` | 3.8–3.10 | (re)generate Mask2Former prediction PNGs — GPU, CUDA 11.8 |

### 3.1 Post-processing pipeline (this repo) — the only one you normally need

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists only the top-level packages the pipeline imports directly
(numpy, opencv-contrib-python, pandas, pycocotools, shapely, tqdm); their transitive
sub-dependencies install automatically. Zhang–Suen thinning uses
`cv2.ximgproc.thinning`, so **opencv-contrib**-python (not plain opencv-python) is required.

Inference is **not required to reproduce the tables/figures** — the pipeline consumes the
model prediction PNGs, which are deterministic and can be reused as-is. Only set up §3.2/§3.3
if you retrain a segmentation model or add a new one.

### 3.2 InternImage inference (optional) — GPU, CUDA 11.3

```bash
python3.9 -m venv .venv-internimage && source .venv-internimage/bin/activate
pip install -r requirements-internimage.txt
# build the DCNv3 CUDA operator once (needs nvcc matching the torch CUDA):
cd InternImage/segmentation/ops_dcnv3 && sh make.sh   # python setup.py build install
```

### 3.3 Mask2Former inference (optional) — GPU, CUDA 11.8

```bash
python3.9 -m venv .venv-mask2former && source .venv-mask2former/bin/activate
pip install -r requirements-mask2former.txt
```

Each inference requirements file carries the `--extra-index-url` / `-f` lines that pull the
CUDA-matched `torch` and `mmcv` wheels, so a single `pip install -r ...` is enough.

> **Why three environments and not one?** The two segmentation stacks are mutually
> incompatible by OpenMMLab design: InternImage needs `mmcv-full 1.5` (torch 1.11 / CUDA 11.3),
> Mask2Former needs `mmcv 2.0` (torch 2.0 / CUDA 11.8), and `mmcv` 1.x vs 2.x share one import
> namespace but have incompatible APIs. The pipeline env additionally uses numpy 2.x, which the
> torch-1.11 stack cannot load. Keep them separate; run each inference script inside its own env.

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
| `DATA_ROOT`        | unpacked data bundle — the only path you must set |
| `DATASET_PATH`     | ADE20K dataset (`DATA_ROOT/ade20k`) |
| `COCO_PATH`        | COCO dataset (`DATA_ROOT/coco`) |
| `SEED_SOURCE_PATH` | SEED vector source the build converts (`DATA_ROOT/SEED_MAP_v1.1`) |
| `RESULT_DIR`       | output folder name (`results`) — change it per run to keep runs apart |
| `EVAL_SPLITS`      | splits to evaluate (`validation`, `test`) |

Class metadata (`METAINFO`, `EXCLUDE_IDS`, `EVAL_CLASS_IDS`, …), the object-metric config
(`F1_IOUS`, `F1_PRIMARY`), the published combination (`BEST_MODEL`, `BEST_PARAMS`), and the
split-name maps (`ADE_SPLIT_DIR`, `COCO_IMG_DIR`) are also in `config.py`.

---

## 5. Running the pipeline

All commands run from `src/`.

### 5.1 Build the datasets (optional — the bundle already ships them)

```bash
python dataprep/build_dataset.py               # SEED_MAP_v1.1 -> ade20k/ + coco/
#   --split validation test     # only some splits
#   --skip images               # regenerate labels/color/coco only (e.g. after a rule change)
#   --coco-images               # also fill coco/{split}2017 with images (stand-alone COCO tree)
```

This is a pure format conversion: it rasterizes the ADE20K index + color labels and encodes
the lane instances into COCO RLE (writing `class_counts.csv` at the end). No geometry is
modified — the SEED source already holds one polyline per lane. Rebuilding `ade20k/` this way
gives slightly different label PNGs than the ones shipped in the bundle (see the bundle's
`README.md`); the COCO instance GT is reproduced exactly.

The merged SEED source itself is produced from the raw SEED release by
`dataprep/merge_annotation.py` (deduplicate → trim overlaps → chain fragments, then write the
result back in the SEED format). The bundle ships the merged revision, so this step only has
to run when the raw release changes:

```bash
python dataprep/merge_annotation.py --src <raw SEED dir> --dst <merged SEED dir>
#   --jobs 14        # worker processes (default: half the cores)
#   --count-only     # just report the raw vs merged lane counts
```

### 5.2 Segmentation inference → `<model>/pred_val`, `<model>/pred_test`

```bash
# run each inside its own inference env (§3.2 / §3.3):
python inference/infer_internimage.py      # .venv-internimage (mmcv-full 1.5 / mmseg 0.x)
python inference/infer_mask2former.py      # .venv-mask2former  (mmcv 2.0 / mmseg 1.x)
#   --validate        confirm the class mapping reproduces existing val predictions (agreement ~1.0)
#   --splits test     test split only
```

Reads images from `ade20k/images/<split>`. Deterministic — reuse existing predictions to
reproduce results without re-running.

### 5.3 Stitching pipeline + evaluation → `RESULT_PATH`

To reproduce the published numbers, run the single best combination (~50 min on validation,
~100 min on test):

```bash
python experiment/run_best_experiment.py --split validation
python experiment/run_best_experiment.py --split test
```

It uses the highest-F1 row of `total_performance.csv` when your own sweep exists in
`RESULT_PATH`, otherwise the published combination (`config.BEST_MODEL` / `BEST_PARAMS`).

To redo the search itself — the sweep tunes the hyper-parameters on validation, then
evaluates the best ones on test:

```bash
# single process (~48 min/combo, ~20 h for the full sweep):
python experiment/run_experiments.py --split validation test

# same result, fanned out across processes (recommended for a full re-run):
MAXJOBS=14 python experiment/run_parallel_sweep.py

# variants:
python experiment/run_experiments.py --fast         # skip windows/collages
python experiment/run_experiments.py --eval-only    # recompute F1 only from existing predictions
```

Both sweep drivers write `total_performance.csv` (and per-combo `eval_result.csv`) into
`RESULT_PATH`. The evaluator caches the EXCLUDE_IDS-filtered COCO GT next to the coco GT
(`..._selected.json`) and the per-model segmentation mIoU in `<model>/metrics_{split}.json`;
both are invalidated automatically when the GT changes.

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
