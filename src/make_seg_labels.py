"""
SEED 벡터 라벨 → ADE20K 인덱스 세그멘테이션 라벨(PNG) 생성 (make_seg_labels)

픽셀 단위 mIoU 평가는 GT로 ADE20K 형식 인덱스 라벨 PNG를 쓴다. 원래 validation
라벨(ade20k/annotations/validation)만 존재하고 test 라벨은 없으므로, val·test 모두
동일 규칙으로 SEED json에서 새로 래스터화해 결과 폴더(cfg.label_dir(split))에 만든다.

인코딩·렌더링은 **원본 ADE20K 생성기와 동일하게** 맞춘다(기존 validation 라벨을 재현):
  - 픽셀 값 = METAINFO 클래스 id + 1  (배경/도로 = 1, center_line = 2, ...)
    evaluator가 GT를 읽을 때 -1 시프트하므로 이 +1 규칙을 그대로 따른다.
  - 차선 폭: cv2 펜 두께가 아니라 **shapely buffer(반경 1.5px, round cap)** 로 중심선을
    폴리곤으로 확장한 뒤 cv2.fillPoly로 채운다. 굽은 구간에서도 폭이 일정한 "일정폭 띠"라,
    정점에서 면적이 부푸는 polylines 펜 두께로는 재현 불가(원본과 GT-GT IoU ≈ 97%가 한계).
    buffer 방식은 원본과 사실상 100% 일치한다.
  - 겹침 해소: 클래스 priority(config_converter.ADE20K_LANE_CATEGORIES)로 그리기 순서를
    정한다. priority 숫자가 작을수록 위에 그려짐(예: stop_line 최상단, center_line 최하단).

이 규칙으로 val 라벨은 기존 ade20k 라벨을 재현하고, test도 완전히 동일한 규칙으로 렌더링된다.
"""

import os
import json
import glob

import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import LineString

import config as cfg


class SegLabelRasterizer:
    buffer_size = 1.5    # 중심선 좌우 확장 반경(px). 원본 생성기와 동일(shapely buffer)
    cap_style = 'round'  # 선 끝 처리(원본과 동일)
    default_size = 768   # 이미지를 못 읽을 때 사용할 기본 캔버스 크기
    # 클래스 그리기 우선순위(원본 config_converter.ADE20K_LANE_CATEGORIES의 priority, id->priority).
    # priority 숫자가 작을수록 위에 그려진다(겹칠 때 우선). ignore(0)는 그리지 않는다.
    CLASS_PRIORITY = {1: 10, 2: 6, 3: 7, 4: 3, 5: 8, 6: 4, 7: 5, 8: 9, 9: 0, 10: 1, 11: 2}

    def __init__(self, split: str, image_ids, out_dir: str):
        self._split = split
        self._image_ids = image_ids
        self._out_dir = out_dir
        self._name2id = {c['name']: c['id'] for c in cfg.METAINFO}
        os.makedirs(out_dir, exist_ok=True)

    def run(self):
        made = 0
        skipped = 0
        for base in tqdm(self._image_ids, desc=f'seg-label[{self._split}]'):
            seed = os.path.join(cfg.SRC_LABEL_DIR, base + '.json')
            if not os.path.exists(seed):
                skipped += 1
                continue
            label = self._rasterize(seed, self._image_size(base))
            cv2.imwrite(os.path.join(self._out_dir, base + '.png'), label)
            made += 1
        print(f'[seg-label] split={self._split}: 생성 {made}개, 건너뜀 {skipped}개 -> {self._out_dir}')

    def _image_size(self, base: str):
        img_file = os.path.join(cfg.SRC_IMAGE_DIR, base + '.png')
        img = cv2.imread(img_file)
        if img is None:
            return self.default_size, self.default_size
        return img.shape[0], img.shape[1]

    def _rasterize(self, seed_json: str, hw) -> np.ndarray:
        h, w = hw
        label = np.ones((h, w), dtype=np.uint8)  # 배경/도로 = 1
        objs = self._load_line_objects(seed_json)
        # priority가 큰 클래스를 먼저 그려, priority가 작은(=우선) 클래스가 위에 남게 한다
        # (원본 ADE20K 생성기와 동일한 겹침 해소 규칙).
        objs.sort(key=lambda o: self.CLASS_PRIORITY.get(o[0], 0), reverse=True)
        for cid, pts in objs:
            for polygon in self._line_to_polygons(pts):
                cv2.fillPoly(label, [polygon], cid + 1)
        return label

    def _line_to_polygons(self, pts: np.ndarray):
        """중심선을 buffer_size 반경으로 확장한 폴리곤(외곽 정수 좌표) 리스트로 변환한다.
        원본 생성기의 expand_line_to_polygon(shapely buffer + fillPoly)과 동일하다."""
        buffered = LineString(pts).buffer(self.buffer_size, cap_style=self.cap_style)
        if buffered.is_empty:
            return []
        geoms = [buffered] if buffered.geom_type == 'Polygon' else list(buffered.geoms)
        polygons = []
        for geom in geoms:
            ext = [[int(round(x)), int(round(y))] for x, y in geom.exterior.coords]
            polygons.append(np.array(ext, dtype=np.int32))
        return polygons

    def _load_line_objects(self, seed_json: str):
        """SEED json에서 (클래스 id, 점배열) 리스트를 로드한다.
        RoadObject·LINE_STRING·점≥2·METAINFO 카테고리만 사용한다."""
        with open(seed_json, 'r') as f:
            data = json.load(f)
        objs = []
        for o in data:
            if o.get('class') != 'RoadObject' or o.get('geometry_type') != 'LINE_STRING':
                continue
            pts = o.get('image_points')
            if not pts or len(pts) < 2:
                continue
            cat = o.get('category')
            if cat not in self._name2id:
                continue
            objs.append((self._name2id[cat], np.array(pts, dtype=np.float64)))
        return objs


def main():
    with open(cfg.DATASET_SPLIT_JSON, 'r') as f:
        dataset = json.load(f)

    for split in cfg.EVAL_SPLITS:
        rasterizer = SegLabelRasterizer(
            split=split,
            image_ids=sorted(dataset[split]),
            out_dir=cfg.label_dir(split),
        )
        rasterizer.run()


if __name__ == '__main__':
    main()
