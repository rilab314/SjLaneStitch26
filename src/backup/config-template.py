import os

DATA_ROOT = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2026_LaneDetector/LaneDetector_on"
DATASET_PATH = DATA_ROOT + "/ade20k"
MODEL_PATH = DATA_ROOT + "/Internimage/satellite_ade20k_250925_internimage_large"

# 출력(예측 JSON·CSV·Figure·Table) 저장 폴더명. 실행마다 바꿔 결과를 분리 보관한다.
RESULT_DIR = "results"
RESULT_PATH = os.path.join(DATA_ROOT, RESULT_DIR)

# GT(merged_annotations.json)가 있는 폴더명. 기본은 출력 폴더(RESULT_DIR)와 동일하게 두어
# 결과 폴더를 GT까지 포함한 자족적 단위로 만든다. 공유 GT를 쓰려면 별도 폴더명으로 지정.
ANNO_DIR = RESULT_DIR
COCO_MERGED_ANNO_PATH = os.path.join(DATA_ROOT, ANNO_DIR, "merged_annotations.json")

# BASE_PATH = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector_on/ade20k'
# DATA_PATH = os.path.join(BASE_PATH)
# LABEL_PATH = os.path.join(BASE_PATH, 'annotations', 'validation')
# PRED_PATH = os.path.join(BASE_PATH, 'prediction')
# COCO_ANNO_PATH = os.path.join(BASE_PATH.replace('ade20k', 'coco'), 'annotations', 'instances_validation2017.json')
#
# # BASE_PATH = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector_on/test/result/thickness=3/sample_stride=10/extend_len=20'
#
# ORIGIN_JSON_PATH = os.path.join(BASE_PATH, 'result', 'coco_pred_instances_origin.json')
# ORIGIN_EXCEPTED_JSON_PATH = os.path.join(BASE_PATH, 'result', 'coco_pred_instances_origin_excepted.json')
# MERGED_JSON_PATH = os.path.join(BASE_PATH, 'result', 'coco_pred_instances_merged.json')
# MERGED_EXCEPTED_JSON_PATH = os.path.join(BASE_PATH, 'result', 'coco_pred_instances_excepted.json')
# # RESULT_PATH = os.path.join(BASE_PATH, 'result')

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
EXCLUDE_IDS = [0, 8, 10, 11]
ID2BGR = {c['id']: (c['color'][2], c['color'][1], c['color'][0]) for c in METAINFO}
EVAL_CLASS_IDS = [c['id'] for c in METAINFO if c['id'] not in EXCLUDE_IDS]
ID2NAME = {c['id']: c['name'] for c in METAINFO}
MODEL_PREFIX = "satellite_ade20k_250925_"
