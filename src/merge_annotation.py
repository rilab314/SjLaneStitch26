"""
분리된 차선 annotation 병합 (merge_annotation)

위성영상 차선 annotation(json)에는 그림상으로는 하나의 차선이지만
여러 개의 LineString 객체로 분리되어 저장된 경우가 있다.
이 스크립트는 분리된 LineString들을 연결하여 통합된 차선 객체로 만든다.

LineStringDetector(lane_detector.py)의 끝점 겹침 병합 아이디어를 참고하되,
고가도로와 그 아래 교차도로처럼 서로 다른 객체가 교차할 수 있으므로
끝점 확장선이 겹치는 것뿐 아니라 "방향이 반대"인 경우에만 병합한다.

알고리즘 개요
1. 각 이미지에 대응하는 json 파일을 불러온다.
2. 각 LineString의 양 끝점을 기준으로 약 20px 확장(extrapolation)한다.
   - 끝점에서 중심 방향으로 약 20px 떨어진 기준 노드를 찾는다(짧으면 더 가까이).
   - (기준 노드 -> 끝점) 방향으로 끝점에서 바깥쪽으로 20px 확장한 점을 구한다.
3. LineString마다 양쪽 끝의 (기준 노드 -> 확장 점) 선분을 768x768 바이너리 이미지에 그린다.
4. 모든 바이너리 이미지를 비교하여 같은 클래스이면서 흰색이 겹치는 객체 쌍을 찾는다.
5. 겹친 선분들의 방향이 반대(정규화 방향 벡터 dot < -0.5)인지 확인한다.
6. 겹치고 방향이 반대인 LineString들을 통합(점 통합 후 순서 정렬)한다.
7. 겹침이 없는 LineString은 그대로 사용한다.
8. 적용 전/후 비교 이미지를 그려서 imshow로 보여주고 저장한다.
9. 통합 결과를 COCO instance segmentation 스타일 json으로 저장한다.

개발 중에는 전체 데이터를 끝까지 처리하지 않고 약 1분간만 실행 후 종료한다.
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Set

import cv2
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

import config as cfg


@dataclass
class EndSeg:
    """LineString 한쪽 끝의 확장 선분 정보."""
    tip: np.ndarray           # 끝점 (x, y)
    ref: np.ndarray           # 중심 방향 기준 노드
    ext: np.ndarray           # 바깥쪽으로 확장된 점
    direction: np.ndarray     # 바깥쪽(외측) 단위 방향 벡터 (tip - ref)
    pixels: Set[int] = field(default_factory=set)  # (ref -> ext) 선분의 래스터 픽셀 (y*W + x)


@dataclass
class Lane:
    """하나의 차선 LineString 인스턴스."""
    idx: int
    category: str
    category_id: int
    points: np.ndarray              # 순서가 있는 점들 (N, 2)
    ends: List[EndSeg] = field(default_factory=list)


class MergeAnnotator:
    extend_len = 20      # 끝점 바깥쪽 확장 길이 (px)
    ref_len = 20         # 끝점에서 안쪽 기준 노드까지의 길이 (px)
    seg_thickness = 5    # 겹침 검사용 확장 선분 두께
    overlap_thresh = 2   # 겹침으로 판정할 최소 픽셀 수
    dot_thresh = -0.5    # 방향이 반대라고 판정할 dot product 임계값
    align_thresh = 0.7   # 끝점 변위가 진행 방향과 정렬됐다고 볼 dot product 임계값(약 45도)
    align_eps = 2.0      # 두 끝점이 사실상 맞닿았다고 볼 거리(px)
    parallel_dist = 12   # 두 본체가 '가깝다'고 볼 측면 거리(px)
    parallel_ratio = 0.4 # 본체가 나란하다고 판정할, 가까운 점들의 비율
    dup_dist = 3.0       # 두 LineString이 '겹친 복제'인지 볼 점-본체 거리(px)
    dup_ratio = 0.8      # 한 선의 점이 상대 본체에 겹친 비율 임계 (이 이상이면 복제)
    mask_thickness = 3   # COCO 마스크 렌더링 두께
    dev_time_limit = None  # 개발 중 실행 제한 시간 (초). None 이면 전체 데이터셋 처리

    def __init__(self, split: str, image_ids: List[str], label_path: str,
                 image_path: str, compare_path: str, coco_path: str):
        self._split = split
        self._image_ids = image_ids
        self._label_path = label_path
        self._image_path = image_path
        self._compare_path = compare_path
        self._coco_path = coco_path
        self._img_shape = (768, 768)
        os.makedirs(self._compare_path, exist_ok=True)
        os.makedirs(os.path.dirname(self._coco_path), exist_ok=True)

        # SEED 카테고리 이름 -> METAINFO 클래스 id (출력 형식을 GT와 동일하게 맞춤)
        self._name2id = {c['name']: c['id'] for c in cfg.METAINFO}
        self._enabled = True       # imshow 사용 여부

    # ------------------------------------------------------------------ #
    # 메인 루프
    # ------------------------------------------------------------------ #
    def run(self):
        print(f'[run] split={self._split} 대상 이미지 {len(self._image_ids)}개')

        # 전체 데이터셋 배치 변환 시에는 imshow를 끄고 진행률만 표시한다.
        full_run = self.dev_time_limit is None
        if full_run:
            self._enabled = False

        coco_images = []
        coco_annotations = []
        total_before = 0
        total_after = 0
        processed = 0
        start_time = time.time()

        desc = f'merge[{self._split}]'
        iterator = tqdm(self._image_ids, desc=desc) if full_run else self._image_ids
        for base in iterator:
            image_id = base  # GT 형식과 동일하게 이미지 id를 basename 문자열로 사용
            img_file = os.path.join(self._image_path, base + '.png')
            json_file = os.path.join(self._label_path, base + '.json')
            if not os.path.exists(json_file):
                continue

            lanes = self._load_lanes(json_file)
            if len(lanes) == 0:
                continue

            image = cv2.imread(img_file)
            if image is not None:
                self._img_shape = image.shape[:2]

            # 끝점 병합 전에 겹친 복제(왕복 스트로크 등)를 먼저 하나로 정리한다.
            deduped = self._dedup_lanes(lanes)
            for lane in deduped:
                self._build_ends(lane)

            merged = self._merge_lanes(deduped)

            total_before += len(lanes)
            total_after += len(merged)
            processed += 1
            if not full_run:
                print(f'[{processed}] {base}: {len(lanes)} -> {len(merged)} 개')

            # COCO 누적 (GT merged_annotations.json 형식과 동일)
            h, w = self._img_shape
            coco_images.append({
                'license': 1,
                'file_name': base + '.png',
                'coco_url': '',
                'height': int(h),
                'width': int(w),
                'date_captured': '',
                'flickr_url': '',
                'id': image_id,
            })
            for lane in merged:
                ann = self._lane_to_annotation(lane, image_id)
                if ann is not None:
                    coco_annotations.append(ann)

            # 비교 이미지 표시 및 저장
            compare = self._make_compare_image(image, lanes, merged, base)
            self._save_compare(compare, base)
            if self._enabled:
                try:
                    cv2.imshow('merge compare (before | after)', compare)
                    cv2.waitKey(1)
                except cv2.error:
                    self._enabled = False

            if self.dev_time_limit is not None and (time.time() - start_time) > self.dev_time_limit:
                print(f'\n[run] 개발 제한 시간({self.dev_time_limit}s) 도달 -> 종료')
                break

        self._save_coco(coco_images, coco_annotations)

        print(f'\n===== 요약 (split={self._split}) =====')
        print(f'처리한 이미지: {processed}개')
        print(f'차선 객체: {total_before}개 -> {total_after}개 '
              f'({total_before - total_after}개 감소)')
        if self._enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    # ------------------------------------------------------------------ #
    # 데이터 로딩
    # ------------------------------------------------------------------ #
    def _load_lanes(self, json_file: str) -> List[Lane]:
        with open(json_file, 'r') as f:
            data = json.load(f)

        lanes = []
        idx = 0
        for obj in data:
            if obj.get('class') != 'RoadObject':
                continue
            if obj.get('geometry_type') != 'LINE_STRING':
                continue
            pts = obj.get('image_points')
            if pts is None or len(pts) < 2:
                continue
            points = np.array(pts, dtype=np.float64)
            category = obj.get('category', 'unknown')
            # METAINFO에 없는 카테고리(예: others)는 평가 대상이 아니므로 제외
            if category not in self._name2id:
                continue
            cat_id = self._name2id[category]
            lanes.append(Lane(idx=idx, category=category, category_id=cat_id, points=points))
            idx += 1
        return lanes

    # ------------------------------------------------------------------ #
    # 끝점 확장 (extrapolation)
    # ------------------------------------------------------------------ #
    def _build_ends(self, lane: Lane):
        lane.ends = []
        for from_head in (True, False):
            tip, ref = self._find_ref_node(lane.points, from_head, self.ref_len)
            direction = tip - ref
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction = direction / norm
            ext = tip + direction * self.extend_len
            pixels = self._rasterize_segment(ref, ext)
            lane.ends.append(EndSeg(tip=tip, ref=ref, ext=ext,
                                    direction=direction, pixels=pixels))

    @staticmethod
    def _find_ref_node(points: np.ndarray, from_head: bool, target: float):
        """끝점에서 중심 방향으로 target px 떨어진 기준 노드를 찾는다.
        선이 짧으면 도달 가능한 가장 먼 점(반대쪽 끝)을 사용한다."""
        seq = points if from_head else points[::-1]
        tip = seq[0].astype(np.float64)
        acc = 0.0
        ref = seq[-1].astype(np.float64)
        for k in range(1, len(seq)):
            acc += np.linalg.norm(seq[k] - seq[k - 1])
            if acc >= target:
                ref = seq[k].astype(np.float64)
                break
        return tip, ref

    def _rasterize_segment(self, p0: np.ndarray, p1: np.ndarray) -> Set[int]:
        """(p0 -> p1) 선분을 바이너리 이미지에 그린 뒤 픽셀 인덱스 집합 반환."""
        h, w = self._img_shape
        img = np.zeros((h, w), dtype=np.uint8)
        a = (int(round(p0[0])), int(round(p0[1])))
        b = (int(round(p1[0])), int(round(p1[1])))
        cv2.line(img, a, b, color=255, thickness=self.seg_thickness)
        ys, xs = np.nonzero(img)
        return set((ys * w + xs).tolist())

    # ------------------------------------------------------------------ #
    # 중복 제거 (dedup) - 끝점 병합 이전 단계
    # ------------------------------------------------------------------ #
    def _dedup_lanes(self, lanes: List[Lane]) -> List[Lane]:
        """같은 클래스이면서 본체가 거의 일치하는(겹친 복제) LineString들을 하나로 합친다.

        SEED 원본은 한 마킹을 정/역방향 왕복 스트로크 등 거의 동일한 위치의
        여러 LineString으로 저장한 경우가 많다(특히 u_turn_zone_line).
        _bodies_parallel 가드는 떨어진 평행 이중선과 겹친 복제를 구분하지 못해
        이들을 끝점 병합으로 합치지 못하므로, 끝점 병합 전에 별도로 정리한다.
        간격이 큰 진짜 이중선(예: 이중 중앙선)은 dup_dist(작음)에 걸리지 않아 보존된다."""
        n = len(lanes)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(n):
            for j in range(i + 1, n):
                if lanes[i].category_id != lanes[j].category_id:
                    continue
                if find(i) == find(j):
                    continue
                if self._is_duplicate(lanes[i], lanes[j]):
                    parent[find(j)] = find(i)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(lanes[i])

        deduped = []
        for members in groups.values():
            deduped.append(members[0] if len(members) == 1 else self._merge_group(members))
        return deduped

    def _is_duplicate(self, a: Lane, b: Lane) -> bool:
        """두 LineString이 거의 같은 위치에 겹쳐 그려진 복제인지 판정.
        한쪽 점들의 dup_ratio 이상이 상대 본체에서 dup_dist 이내이면 복제로 본다."""
        d_ab = self._point_to_polyline_dist(a.points, b.points)
        d_ba = self._point_to_polyline_dist(b.points, a.points)
        cover_ab = float(np.mean(d_ab < self.dup_dist))
        cover_ba = float(np.mean(d_ba < self.dup_dist))
        return max(cover_ab, cover_ba) >= self.dup_ratio

    # ------------------------------------------------------------------ #
    # 병합
    # ------------------------------------------------------------------ #
    def _merge_lanes(self, lanes: List[Lane]) -> List[Lane]:
        n = len(lanes)
        parent = list(range(n))
        members = {i: [i] for i in range(n)}
        par_cache = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def parallel(i, j):
            key = (i, j) if i < j else (j, i)
            if key not in par_cache:
                par_cache[key] = self._bodies_parallel(lanes[i], lanes[j])
            return par_cache[key]

        # 같은 클래스끼리 병합 후보 쌍 수집
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if lanes[i].category_id != lanes[j].category_id:
                    continue
                if self._should_merge(lanes[i], lanes[j]):
                    candidates.append((i, j))

        # 그룹 인식 union: 두 그룹을 합칠 때 교차 멤버 중 평행한 쌍이 있으면 거부한다.
        # (짧은 연결 segment가 분기점에서 평행한 두 strand를 전이적으로
        #  이어붙여 지그재그가 되는 것을 방지)
        for i, j in candidates:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            mi, mj = members[ri], members[rj]
            if any(parallel(p, q) for p in mi for q in mj):
                continue
            parent[rj] = ri
            members[ri] = mi + mj
            members.pop(rj, None)

        # 그룹 묶기
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(lanes[i])

        merged = []
        for members in groups.values():
            if len(members) == 1:
                merged.append(members[0])
            else:
                merged.append(self._merge_group(members))
        return merged

    def _should_merge(self, a: Lane, b: Lane) -> bool:
        """두 LineString이 끝과 끝으로 이어지면 True.

        조건: 확장선분이 겹치고(겹침), 두 끝의 외측 방향이 반대이며(방향),
        두 끝점을 잇는 변위가 진행 방향과 같은 축으로 정렬되어야 한다(정렬).
        정렬 조건이 없으면 나란히 평행한 이중 중앙선이 잘못 병합되어
        지그재그가 생기므로 반드시 필요하다."""
        for ea in a.ends:
            for eb in b.ends:
                if len(ea.pixels & eb.pixels) < self.overlap_thresh:
                    continue
                if float(np.dot(ea.direction, eb.direction)) >= self.dot_thresh:
                    continue  # 방향이 반대가 아님
                # 끝점 변위가 진행 방향과 정렬되었는지 검사 (평행 이중선 배제)
                gap = eb.tip - ea.tip
                dist = float(np.linalg.norm(gap))
                if dist > self.align_eps:
                    along = float(np.dot(gap / dist, ea.direction))
                    if along < self.align_thresh:
                        continue  # 옆으로 나란한(수직 변위) 평행선 -> 병합하지 않음
                # 끝점은 통과했지만 두 본체가 나란히 가까이 달리면 병합하지 않음
                if self._bodies_parallel(a, b):
                    return False
                return True
        return False

    def _bodies_parallel(self, a: Lane, b: Lane) -> bool:
        """두 LineString의 본체가 나란히 가까이 달리는지 검사.

        한쪽 점들이 다른 쪽 본체(polyline)에서 parallel_dist 이내인 비율이
        parallel_ratio 를 넘으면 평행한 이중선으로 보고 병합을 막는다.
        끝과 끝으로 이어지는 경우는 접점 부근만 가깝고 본체 대부분은 멀어
        비율이 낮으므로 구분된다."""
        d_ba = self._point_to_polyline_dist(b.points, a.points)
        d_ab = self._point_to_polyline_dist(a.points, b.points)
        ratio_b = float(np.mean(d_ba < self.parallel_dist))
        ratio_a = float(np.mean(d_ab < self.parallel_dist))
        return max(ratio_a, ratio_b) > self.parallel_ratio

    @staticmethod
    def _point_to_polyline_dist(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
        """pts(M,2)의 각 점에서 polyline(N,2)까지의 최소 거리 (M,) 반환.
        선분 위로 투영(클램프)하여 점-선분 거리를 계산한다."""
        if len(poly) < 2:
            return np.full(len(pts), np.inf)
        seg_a = poly[:-1]                         # (S, 2)
        seg_ab = poly[1:] - poly[:-1]             # (S, 2)
        seg_len2 = np.sum(seg_ab ** 2, axis=1)    # (S,)
        seg_len2[seg_len2 == 0] = 1e-9
        rel = pts[:, None, :] - seg_a[None, :, :]            # (M, S, 2)
        t = np.sum(rel * seg_ab[None, :, :], axis=2) / seg_len2[None, :]  # (M, S)
        t = np.clip(t, 0.0, 1.0)
        proj = seg_a[None, :, :] + t[:, :, None] * seg_ab[None, :, :]     # (M, S, 2)
        d = np.linalg.norm(pts[:, None, :] - proj, axis=2)               # (M, S)
        return d.min(axis=1)

    def _merge_group(self, members: List[Lane]) -> Lane:
        """여러 LineString의 점들을 통합한 뒤 순서대로 정렬한다."""
        all_points = np.vstack([m.points for m in members])
        ordered = self._order_points(all_points, members)
        base = members[0]
        return Lane(idx=base.idx, category=base.category,
                    category_id=base.category_id, points=ordered)

    @staticmethod
    def _order_points(points: np.ndarray, members: List[Lane]) -> np.ndarray:
        """통합된 점들을 nearest-neighbor 체이닝으로 순서대로 정렬한다.
        시작점은 그룹 내 모든 끝점(tip) 중 중심에서 가장 먼 점으로 선택한다."""
        # 시작점 결정
        tips = []
        for m in members:
            tips.append(m.points[0])
            tips.append(m.points[-1])
        tips = np.array(tips, dtype=np.float64)
        centroid = points.mean(axis=0)
        start = tips[np.argmax(np.linalg.norm(tips - centroid, axis=1))]

        remaining = points.astype(np.float64).copy()
        # 시작점에서 가장 가까운 실제 점부터 시작
        order = []
        cur_idx = int(np.argmin(np.linalg.norm(remaining - start, axis=1)))
        order.append(remaining[cur_idx])
        mask = np.ones(len(remaining), dtype=bool)
        mask[cur_idx] = False

        last = order[-1]
        while mask.any():
            idxs = np.nonzero(mask)[0]
            d = np.linalg.norm(remaining[idxs] - last, axis=1)
            nxt = idxs[int(np.argmin(d))]
            order.append(remaining[nxt])
            mask[nxt] = False
            last = remaining[nxt]
        return np.array(order, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 비교 이미지 (적용 전/후)
    # ------------------------------------------------------------------ #
    def _make_compare_image(self, image, before: List[Lane], after: List[Lane], base: str):
        h, w = self._img_shape
        if image is None:
            bg = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            bg = (image.astype(np.float32) * 0.4).astype(np.uint8)

        left = self._draw_lanes(bg.copy(), before, f'BEFORE ({len(before)})')
        right = self._draw_lanes(bg.copy(), after, f'AFTER ({len(after)})')

        sep = np.full((h, 4, 3), (255, 255, 255), dtype=np.uint8)
        return cv2.hconcat([left, sep, right])

    def _draw_lanes(self, canvas, lanes: List[Lane], title: str):
        for lane in lanes:
            # COCO segmentation과 동일하게 두께 3으로 그린다 (클래스별 색상)
            color = cfg.ID2BGR.get(lane.category_id, (255, 255, 255))
            pts = np.rint(lane.points).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=self.mask_thickness)
            # 양 끝점 표시
            for tip in (lane.points[0], lane.points[-1]):
                p = (int(round(tip[0])), int(round(tip[1])))
                cv2.circle(canvas, p, radius=5, color=(255, 255, 255), thickness=-1)
                cv2.circle(canvas, p, radius=5, color=color, thickness=2)
        cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        return canvas

    def _save_compare(self, compare, base: str):
        out_path = os.path.join(self._compare_path, base + '.png')
        cv2.imwrite(out_path, compare)

    # ------------------------------------------------------------------ #
    # COCO 출력
    # ------------------------------------------------------------------ #
    def _lane_to_annotation(self, lane: Lane, image_id: str):
        """LineString을 두께 3으로 그린 영역을 RLE segmentation으로 저장.
        GT merged_annotations.json과 동일하게 image_id/category_id/segmentation/score만 담는다."""
        h, w = self._img_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.rint(lane.points).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=self.mask_thickness)
        if mask.sum() == 0:
            return None

        fmask = np.asfortranarray(mask)
        rle = maskUtils.encode(fmask)
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')

        return {
            'image_id': image_id,
            'category_id': lane.category_id,
            'segmentation': rle,
            'score': 1.0,
        }

    def _save_coco(self, images: list, annotations: list):
        categories = [{'id': c['id'], 'name': c['name'], 'supercategory': 'segmentation'}
                      for c in cfg.METAINFO]
        coco = {
            'info': {'contributor': '', 'date_created': '2024/12/13', 'description': '',
                     'url': '', 'version': '1.0', 'year': 2024},
            'licenses': [],
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }
        with open(self._coco_path, 'w') as f:
            json.dump(coco, f)
        print(f'[save] merged_annotations json 저장: {self._coco_path} '
              f'(images={len(images)}, annotations={len(annotations)}, '
              f'categories={len(categories)})')


def _split_coco_path(split: str) -> str:
    """split 별 merged_annotations_{split}.json 경로를 만든다."""
    root, ext = os.path.splitext(cfg.MERGED_COCO_PATH)
    return f'{root}_{split}{ext}'


def count_class_instances(splits: List[str], csv_path: str):
    """split 별 merged json을 읽어 클래스별 개체수를 계산하고 csv로 저장한다.

    행: 클래스(이름), 열: split. 마지막에 합계 행/열을 추가한다."""
    id2name = {c['id']: c['name'] for c in cfg.METAINFO}
    # 평가에 쓰는 실제 차선 클래스(ignore 제외)만 행으로 사용
    class_ids = [c['id'] for c in cfg.METAINFO if c['id'] != 0]

    # counts[split][class_id]
    counts = {sp: {cid: 0 for cid in class_ids} for sp in splits}
    for sp in splits:
        path = _split_coco_path(sp)
        if not os.path.exists(path):
            print(f'[count] {path} 없음 -> {sp} 건너뜀')
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        for ann in data['annotations']:
            cid = ann['category_id']
            if cid in counts[sp]:
                counts[sp][cid] += 1

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write('class_id,class_name,' + ','.join(splits) + ',total\n')
        col_totals = {sp: 0 for sp in splits}
        for cid in class_ids:
            row = [counts[sp][cid] for sp in splits]
            for sp in splits:
                col_totals[sp] += counts[sp][cid]
            f.write(f'{cid},{id2name[cid]},' + ','.join(map(str, row)) +
                    f',{sum(row)}\n')
        total_row = [col_totals[sp] for sp in splits]
        f.write('-,total,' + ','.join(map(str, total_row)) +
                f',{sum(total_row)}\n')
    print(f'[count] 클래스별 개체수 csv 저장: {csv_path}')


def main():
    with open(cfg.DATASET_SPLIT_JSON, 'r') as f:
        dataset = json.load(f)

    # train, validation 두 split에 대해 같은 클래스 객체를 각각 만들어 처리한다.
    splits = ['train', 'validation']
    for split in splits:
        annotator = MergeAnnotator(
            split=split,
            image_ids=sorted(dataset[split]),
            label_path=cfg.SEED_LABEL_PATH,
            image_path=os.path.join(cfg.COCO_ROOT, cfg.SPLIT_IMAGE_DIR[split]),
            compare_path=os.path.join(cfg.MERGE_COMPARE_PATH, split),
            coco_path=_split_coco_path(split),
        )
        annotator.run()

    # 두 작업이 끝난 뒤 별도 함수로 클래스별 개체수를 계산해 csv로 저장한다.
    csv_path = os.path.join(os.path.dirname(cfg.MERGED_COCO_PATH), 'class_counts.csv')
    count_class_instances(splits, csv_path)


if __name__ == '__main__':
    main()
