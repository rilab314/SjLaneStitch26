"""Table 1–5 공유 헬퍼: best 조합 조회, 예측 JSON 경로, 모델 표시명, 출력 포맷."""
import os

import pandas as pd

import config as cfg

# 본문 표 기준 작동점: merge×1 (merge×2는 Table 4 단계별 표에서만 별도로 보인다).
MERGE_COUNT = 1

# CSV의 내부 모델명 → 논문 표기명.
MODEL_DISPLAY = {
    "mask2former_large": "Mask2Former (Swin-L)",
    "mask2former_small": "Mask2Former (Swin-S)",
    "internimage_large": "InternImage-L",
}
# 표 출력 시 모델 행 순서.
MODEL_ORDER = ["mask2former_large", "mask2former_small", "internimage_large"]

BLANK = "–"  # en dash, 빈 칸 표기


def total_csv_path():
    return os.path.join(cfg.RESULT_PATH, "total_performance.csv")


def tables_dir():
    return os.path.join(cfg.RESULT_PATH, "Tables")


def with_val_aliases(df):
    """total_performance/eval의 (val) 지표 열에 무접미사 별칭을 추가한다.
    기존 표 스크립트가 df['AP20'] 등으로 validation 값을 그대로 읽게 하는 하위호환 shim.
    (test 열은 df['AP20(test)']로 직접 접근)"""
    for m in ("instances", "AP10", "AP20", "AP50", "mIoU"):
        v = cfg.mcol(m, "validation")
        if v in df.columns and m not in df.columns:
            df[m] = df[v]
    return df


def load_total_performance():
    """total_performance.csv를 읽어 (val) 별칭을 붙인 DataFrame을 반환한다."""
    return with_val_aliases(pd.read_csv(total_csv_path()))


def best_combo(df):
    """AP20(val) 최고 행의 model·하이퍼파라미터(dict)를 반환한다."""
    ap = cfg.mcol("AP20", "validation") if cfg.mcol("AP20", "validation") in df.columns else "AP20"
    best = df.sort_values(ap, ascending=False, na_position="last").iloc[0]
    return {
        "model_name": str(best["model_name"]),
        "thicknesses": int(best["thicknesses"]),
        "sample_strides": int(best["sample_strides"]),
        "extend_lens": int(best["extend_lens"]),
        "turn_penalties": int(best["turn_penalties"]),
    }


def param_dir_name(combo):
    return (f"thick={combo['thicknesses']},stride={combo['sample_strides']},"
            f"extend={combo['extend_lens']},turn={combo['turn_penalties']}")


def pred_json_path(combo, merge_count, split="validation"):
    """best 조합·merge 단계에 해당하는 예측 JSON 경로 (기본 validation)."""
    model_dir = cfg.MODEL_PREFIX + combo["model_name"]
    name = "origin" if merge_count == 0 else f"merge{merge_count}"
    return os.path.join(cfg.RESULT_PATH, model_dir, param_dir_name(combo),
                        f"coco_pred_{cfg.split_label(split)}_{name}.json")


def pct(value):
    """비율(0~1) → % 소수 둘째 자리."""
    return round(float(value) * 100, 2)


def save_csv(df, name):
    out_dir = tables_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"saved: {path}")
    print(df.to_string(index=False))
    return path
