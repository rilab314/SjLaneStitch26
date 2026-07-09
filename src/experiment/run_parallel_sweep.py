"""
Parallel reproduction of the run_experiments sweep (run_parallel_sweep)

run_experiments.py runs every (model, param, split) combo in one process, back to back
(~48 min/combo -> ~20 h for the full 18+2+3 sweep). Detection is deterministic and each
combo writes only its own param folder, so the combos can run as independent subprocesses.
This driver fans them out (default 14 at a time) and reproduces the exact same
total_performance.csv, just much faster.

Phases (identical selection logic to run_experiments.run_experiments):
  0. pre-generate the two shared eval caches so parallel workers only read them:
       - coco/annotations/instances_{split}2017_selected.json   (COCO AP GT, EXCLUDE_IDS filtered)
       - <model>/metrics_{split}.json                            (segmentation-prediction mIoU)
  A. 18 best-model (mask2former_large) validation combos -> total_performance -> pick best param
  B. 2 other-model validation combos (best param) + 3 all-model test combos (best param)
  C. rewrite total_performance.csv with every column filled

Usage (from src/, after config.RESULT_DIR is set to the target run folder):
    MAXJOBS=14 python experiment/run_parallel_sweep.py
"""

import os
import sys
import glob
import time
import subprocess
import concurrent.futures as cf

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import _bootstrap  # noqa: F401
import config as cfg
from evaluator import _get_selected_annotation, evaluate_segm_pred_metrics
from run_experiments import (find_model_paths, find_best_model_and_params,
                             _best_param_combo, BEST_MODEL)

MAXJOBS = int(os.environ.get('MAXJOBS', '16'))
THICK = 3
STRIDES = [5, 10]
EXTENDS = [10, 20, 30]
TURNS = [0, 3, 5]


def model_token(model_path):
    return os.path.basename(model_path).replace(cfg.MODEL_PREFIX, '')


def wait_for_gt(split, timeout=10800):
    """Block until the built GT for a split exists (coco json + index labels + images).
    Lets the sweep overlap with a still-running build of a later split."""
    t0 = time.time()
    while True:
        ok = (os.path.exists(cfg.coco_anno_path(split))
              and glob.glob(os.path.join(cfg.label_dir(split), '*.png'))
              and glob.glob(os.path.join(cfg.image_dir(split), '*.png')))
        if ok:
            return
        if time.time() - t0 > timeout:
            raise TimeoutError(f'GT for split={split} not ready after {timeout}s')
        print(f'[wait] GT for split={split} not ready yet, retrying in 30s ...', flush=True)
        time.sleep(30)


def pregenerate_caches(model_paths, splits):
    """Build the two shared caches single-threaded so parallel workers are read-only on them."""
    for split in splits:
        print(f'[pregen] selected COCO GT cache ({split}) ...')
        _get_selected_annotation(cfg.coco_anno_path(split))
        print(f'[pregen] segmentation-prediction mIoU caches ({split}, clear stale first) ...')
        for mp in model_paths:
            stale = os.path.join(mp, f'metrics_{split}.json')
            if os.path.exists(stale):
                os.remove(stale)
            evaluate_segm_pred_metrics(mp, cfg.label_dir(split), split)


# One combo == one core: cap every math/vision library to a single thread so that N
# combos running in parallel use N cores instead of N*(2*ncpu) threads (huge oversubscription).
_THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}


def run_combo(logdir, model, s, e, tp, split):
    tag = f'{model}_s{s}_e{e}_tp{tp}_{split}'
    cmd = [sys.executable, os.path.join(SRC, 'experiment', 'run_experiments.py'),
           '--run-combo', model, str(THICK), str(s), str(e), str(tp), split]
    env = {**os.environ, **_THREAD_ENV}
    with open(os.path.join(logdir, tag + '.log'), 'w') as f:
        rc = subprocess.run(cmd, cwd=SRC, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return tag, rc


def run_pool(tasks, logdir):
    done = 0
    with cf.ThreadPoolExecutor(max_workers=MAXJOBS) as ex:
        futs = [ex.submit(run_combo, logdir, *t) for t in tasks]
        for fut in cf.as_completed(futs):
            tag, rc = fut.result()
            done += 1
            print(f'[{done}/{len(tasks)} rc={rc}] {tag}', flush=True)


def main():
    logdir = os.path.join(cfg.RESULT_PATH, '_combo_logs')
    os.makedirs(logdir, exist_ok=True)
    print(f'RESULT_DIR={cfg.RESULT_DIR}  MAXJOBS={MAXJOBS}  logs={logdir}')

    model_paths = find_model_paths(cfg.DATA_ROOT)
    best_dir = cfg.MODEL_PREFIX + BEST_MODEL
    best = next((p for p in model_paths if os.path.basename(p) == best_dir), model_paths[0])
    best_tok = model_token(best)
    other_toks = [model_token(p) for p in model_paths if p != best]

    # Phase A: best-model validation sweep (18 combos)
    wait_for_gt('validation')
    pregenerate_caches(model_paths, ['validation'])
    tasks = [(best_tok, s, e, tp, 'validation')
             for s in STRIDES for e in EXTENDS for tp in TURNS]
    print(f'\n=== Phase A: {len(tasks)} best-model validation combos ===')
    run_pool(tasks, logdir)
    find_best_model_and_params()
    bs, be, btp = _best_param_combo(best_tok)
    print(f'\nBest validation params: stride={bs} extend={be} turn={btp}')

    # Phase B: other-model validation (best param) + all-model test (best param).
    # test GT may still be building -> wait for it before pre-generating its caches.
    wait_for_gt('test')
    pregenerate_caches(model_paths, ['test'])
    tasks = [(tok, bs, be, btp, 'validation') for tok in other_toks]
    tasks += [(tok, bs, be, btp, 'test') for tok in [best_tok] + other_toks]
    print(f'\n=== Phase B: {len(tasks)} combos (other-val + all-test) ===')
    run_pool(tasks, logdir)
    find_best_model_and_params()
    print('\nParallel sweep completed. total_performance.csv written to', cfg.RESULT_PATH)


if __name__ == '__main__':
    main()
