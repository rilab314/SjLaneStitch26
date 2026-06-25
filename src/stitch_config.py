"""best 조합(모델·하이퍼파라미터) 로더.

total_performance.csv의 AP20 최고 행을 읽어 StitchConfig로 제공한다.
figure 생성(Figure/figure_base)과 단일 실험(run_best_experiment)이 공유한다.
"""
import os
import sys
from dataclasses import dataclass

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import config as cfg

DEFAULT_MODEL = "mask2former_large"


@dataclass
class StitchConfig:
    """LaneStitcher를 구성할 모델·하이퍼파라미터 조합."""

    model_name: str
    model_path: str
    pred_dir: str
    thickness: int
    sample_stride: int
    extend_len: int
    turn_penalty: float
    merge_count: int


def load_stitch_config(prefer_model=DEFAULT_MODEL):
    """total_performance.csv의 최고 AP20 조합을 읽어 StitchConfig로 반환(없으면 기본값)."""
    csv_path = os.path.join(cfg.RESULT_PATH, "total_performance.csv")
    if os.path.exists(csv_path):
        return build_config_from_csv(csv_path)
    print(f"[config] {csv_path} 없음 → 기본 파라미터({prefer_model}) 사용")
    return build_default_config(prefer_model)


def build_config_from_csv(csv_path):
    """CSV에서 AP20 최고 행을 골라 StitchConfig를 만든다."""
    frame = pd.read_csv(csv_path)
    best = frame.sort_values("AP20", ascending=False, na_position="last").iloc[0]
    model_name = str(best["model_name"])
    model_path = resolve_model_path(model_name)
    print(f"[config] best: {model_name} thick={int(best['thicknesses'])} "
          f"stride={int(best['sample_strides'])} extend={int(best['extend_lens'])} "
          f"turn={float(best['turn_penalties'])} merge={int(best['merge_count'])} "
          f"AP20={float(best['AP20']):.4f}")
    return StitchConfig(
        model_name=model_name,
        model_path=model_path,
        pred_dir=os.path.join(model_path, "prediction"),
        thickness=int(best["thicknesses"]),
        sample_stride=int(best["sample_strides"]),
        extend_len=int(best["extend_lens"]),
        turn_penalty=float(best["turn_penalties"]),
        merge_count=max(int(best["merge_count"]), 1),
    )


def build_default_config(model_name):
    """CSV가 없을 때 쓰는 기본 조합(LaneStitcher 기본값과 동일)."""
    model_path = resolve_model_path(model_name)
    return StitchConfig(model_name, model_path, os.path.join(model_path, "prediction"),
                        thickness=3, sample_stride=10, extend_len=20,
                        turn_penalty=3.0, merge_count=3)


def resolve_model_path(model_name):
    """모델 이름 → segmentation 예측이 저장된 모델 디렉토리 경로."""
    kind = "Internimage" if "internimage" in model_name.lower() else "mask2former"
    return os.path.join(cfg.DATA_ROOT, kind, cfg.MODEL_PREFIX + model_name)
