"""Figure 생성 공통 골격(FigureGenerator 베이스 클래스).

best 조합(stitch_config)으로 LaneStitcher를 구성하고, validation 프레임을 순회하며
조건을 만족하는 프레임만 저장한다. 하위 클래스는 `name`과 `build_figure`만 정의한다.
"""
import os
import sys
import glob

import cv2
from tqdm import tqdm

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Figure/ → src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as cfg
from lane_stitcher import LaneStitcher
from util import load_json, group_annotations_by_image
from stitch_config import load_stitch_config


class FigureGenerator:
    """validation 프레임을 순회하며 조건을 만족하는 figure만 저장하는 베이스.

    build_figure는 (이미지, 파일명 접미사) 또는 None(조건 불일치)을 반환한다.
    """

    name = "Figure"
    gap = 20

    def __init__(self):
        self._config = load_stitch_config()
        self._detector = self.build_detector()
        self._out_dir = os.path.join(cfg.RESULT_PATH, "Figure", self.name)
        os.makedirs(self._out_dir, exist_ok=True)
        self._val_files = sorted(glob.glob(
            os.path.join(cfg.DATASET_PATH, "images", "validation", "*.png")))
        self._gt_map = None

    def build_detector(self):
        """best 파라미터로 LaneStitcher 인스턴스를 구성한다."""
        conf = self._config
        detector = LaneStitcher(cfg.DATASET_PATH, conf.model_path, cfg.RESULT_PATH,
                                thickness=conf.thickness, sample_stride=conf.sample_stride,
                                extend_len=conf.extend_len, visualize=False)
        detector.turn_penalty = conf.turn_penalty
        return detector

    def run(self):
        """전 프레임을 순회하며 조건 만족분만 저장하고 결과를 보고한다."""
        kept = 0
        for path in tqdm(self.select_files(), desc=self.name):
            kept += int(self.save_if_match(path))
        self.report(kept)

    def select_files(self):
        """환경변수 FIG_LIMIT가 있으면 앞 N개만(스모크 테스트용)."""
        cap = os.environ.get("FIG_LIMIT")
        return self._val_files[:int(cap)] if cap else self._val_files

    def save_if_match(self, path):
        """조건을 만족하면 figure를 저장하고 True, 아니면 False."""
        image_id = os.path.basename(path)[:-4]
        result = self.build_figure(image_id, path)
        if result is None:
            return False
        image, suffix = result
        cv2.imwrite(os.path.join(self._out_dir, f"{image_id}{suffix}.png"), image)
        return True

    def build_figure(self, image_id, path):
        """(이미지, 접미사) 또는 None을 반환. 하위 클래스가 구현."""
        raise NotImplementedError

    def read_prediction(self, image_id):
        """모델 segmentation 예측 PNG를 읽는다(없으면 None)."""
        return cv2.imread(os.path.join(self._config.pred_dir, f"{image_id}.png"))

    def gt_annotations(self, image_id):
        """해당 image_id의 GT 어노테이션 리스트(최초 호출 시 한 번 로딩)."""
        if self._gt_map is None:
            data = load_json(cfg.COCO_ANNO_PATH)
            self._gt_map = group_annotations_by_image(data["annotations"]) if data else {}
        return self._gt_map.get(image_id, [])

    def final_merge(self, stage):
        """best merge_count 병합 단계 linestring. 짧은선 제거(_filter_short) 적용 → 평가 출력과 정합.

        stage_linestrings의 merges는 detect_lines 출력과 달리 짧은선 필터가 안 걸려 있어,
        평가에선 빠지는 min_lane_len 미만 조각이 figure에 남는다. 여기서 동일하게 걸러준다."""
        merges = stage["merges"]
        if not merges:
            return stage["refined"]
        idx = min(self._config.merge_count, len(merges)) - 1
        return self._detector._filter_short(merges[idx])

    def report(self, kept):
        """저장 결과 요약 출력."""
        print(f"[{self.name}] {kept}/{len(self._val_files)} frames -> {self._out_dir}")
