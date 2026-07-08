import os

DATA_ROOT = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2026_LaneDetector/LaneDetector_on"
DATASET_PATH = DATA_ROOT + "/ade20k"  # 원본 위성 이미지·색상 라벨(그림용) 보관

# 출력(예측 JSON·CSV·Figure·Table) 저장 폴더명. 실행마다 바꿔 결과를 분리 보관한다.
RESULT_DIR = "results_260706"
RESULT_PATH = os.path.join(DATA_ROOT, RESULT_DIR)

# ────────────────────────────────────────────────────────────────────── #
# 원본 데이터셋 (권위 소스): 이미지·SEED 벡터 라벨·split 목록이 모두 여기 있다.
#   image/        : 전 split 혼재 위성 이미지 {basename}.png (train+val+test = 12828)
#   label/        : 전 split 혼재 SEED 벡터 라벨 {basename}.json
#   dataset.json  : split(train/validation/test) 별 basename 목록
# ADE20K(images/validation)는 val 서브셋만 있고 test는 없으므로, 추론·라벨 생성의
# 이미지·split 소스는 항상 이 원본 폴더를 쓴다.
# ────────────────────────────────────────────────────────────────────── #
SRC_DATASET_PATH = os.path.join(DATA_ROOT, "satellite_good_matching_250206")
SRC_IMAGE_DIR = os.path.join(SRC_DATASET_PATH, "image")
SRC_LABEL_DIR = os.path.join(SRC_DATASET_PATH, "label")
DATASET_SPLIT_JSON = os.path.join(SRC_DATASET_PATH, "dataset.json")

# 평가할 split과 split→모델 예측 폴더명 매핑.
# 모델 추론은 순수 클래스-색상 마스크를 <model>/pred_val, <model>/pred_test 에 저장한다
# (기존 단일 'prediction' 폴더를 split별로 분리).
EVAL_SPLITS = ['validation', 'test']
SPLIT_PRED_DIR = {'validation': 'pred_val', 'test': 'pred_test'}
# 산출물 파일명·CSV 열에 붙일 짧은 split 라벨 (coco_pred_val_*, AP20(val) 등)
SPLIT_LABEL = {'validation': 'val', 'test': 'test'}


def pred_dirname(split):
    """split -> 모델 예측 마스크 하위 폴더명 (pred_val / pred_test)."""
    return SPLIT_PRED_DIR[split]


def split_label(split):
    """split -> 산출물 파일명/열에 쓰는 짧은 라벨 (validation->val, test->test)."""
    return SPLIT_LABEL[split]


def mcol(metric, split):
    """split별 지표 열 이름. 예: mcol('AP20','test') -> 'AP20(test)'."""
    return f"{metric}({SPLIT_LABEL[split]})"


def pred_path(model_path, split):
    """모델 디렉토리 아래 split별 예측 마스크 폴더 경로."""
    return os.path.join(model_path, SPLIT_PRED_DIR[split])


def coco_anno_path(split):
    """split별 COCO GT(merged_annotations_{split}.json) 경로 (결과 폴더 내)."""
    return os.path.join(RESULT_PATH, f"merged_annotations_{split}.json")


def label_dir(split):
    """split별 ADE20K 인덱스 라벨(PNG) 디렉토리. SEED json에서 래스터화해 생성한다."""
    return os.path.join(RESULT_PATH, "labels", split)


def merge_compare_dir(split):
    """merge_annotation 전/후 비교 이미지 저장 디렉토리 (split별)."""
    return os.path.join(RESULT_PATH, "merge_compare", split)


def split_result_path(split):
    """알고리즘 결과 루트. val·test를 경로로 나누지 않고 같은 RESULT_PATH를 쓴다.
    두 split은 파일명(coco_pred_val_*/coco_pred_test_*)과 CSV 열 접미사(…(val)/…(test))로만
    구분한다. (인자는 호출부 호환을 위해 유지하되 반환값은 split과 무관하게 동일)"""
    return RESULT_PATH


# ── 기존 스크립트(Figure/Table 등) 호환용 validation 기본 별칭 ── #
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
EXCLUDE_IDS = [0, 8, 10]   # bicycle_lane(11) 평가 포함
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

# merge_annotation.py 전용: 분리된 SEED 차선 라벨을 병합해 COCO GT를 만든다.
# 이미지/라벨/ split 목록은 위 원본 데이터셋(SRC_*) 경로를 그대로 쓴다.
SEED_LABEL_PATH = SRC_LABEL_DIR
