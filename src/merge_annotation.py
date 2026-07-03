"""
분리된 차선 annotation 병합 (merge_annotation)

위성영상 차선 annotation(json)에는 그림상으로는 하나의 차선이지만
여러 개의 polyline 객체로 분리되어 저장된 경우가 있다.
이 스크립트는 분리된 polyline들을 연결하여 통합된 차선 객체로 만든다.

LaneStitcher(lane_stitcher.py)의 끝점 겹침 병합 아이디어를 참고하되,
고가도로와 그 아래 교차도로처럼 서로 다른 객체가 교차할 수 있으므로
끝점 확장선이 겹치는 것뿐 아니라 "방향이 반대"인 경우에만 병합한다.

알고리즘 개요 (이미지별)
1. json에서 polyline들을 Lane 리스트로 로드한다.
2. dedup: 거의 같은 위치의 복제선(왕복 스트로크 등)은 union-find로 묶어 가장 긴 것만
   남기고 나머지는 버린다(점을 섞지 않으므로 지그재그가 생기지 않음).
3. trim(center_line 한정): 가장 긴 선을 기준으로, 짧은 선에서 기준선과 겹치는 구간을
   Canny식 이중 임계(강 overlap_dist / 약 overlap_low)로 잘라낸다. 갈라지는 가지는
   별도 선으로 남고 완전 겹침 복제는 사라진다.
4. 끝점 병합: 각 선의 양 끝을 바깥쪽으로 extend_len 확장한 선분을 만들고, 같은 클래스이며
   확장선분이 겹치고 + 외측 방향이 반대이고 + 정렬된 끝끼리 직렬로 잇는다. 본체가 나란히
   달리는(평행 이중선) 쌍은 제외한다.
5. 통합 결과를 COCO instance segmentation json으로 저장하고, 전/후 비교 이미지를 남긴다.

개발 중에는 dev_time_limit 으로 일부만 처리할 수 있다(None 이면 전체 처리).
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import List, Set

import cv2
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

# 공유 그리기 함수(figure_render)는 Figure/ 하위에 있으므로 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figure"))
import config as cfg
import figure_render as fr


# ====================================================================== #
# 기하/정리 순수 함수 (구 polyline_merge 모듈을 merge_annotation이 자체 보유)
# dedup(is_duplicate)·trim(subtract_lane)에 쓰는 폴리라인 유틸.
# ====================================================================== #
def arc_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(points, float), axis=0), axis=1).sum())


def resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    """순서가 있는 폴리라인을 호길이 균일 간격(step)으로 재샘플한다."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    if total < 1e-6:
        return pts[:1]
    m = max(int(np.floor(total / step)), 1)
    s = np.linspace(0.0, total, m + 1)
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    return np.stack([x, y], axis=1)


def true_runs(mask: np.ndarray):
    """불리언 배열에서 연속된 True 구간의 (start, end) 리스트(end 배타적)를 반환한다."""
    m = np.asarray(mask)
    if m.size == 0:
        return []
    d = np.diff(m.astype(np.int8))
    starts = (np.flatnonzero(d == 1) + 1).tolist()
    ends = (np.flatnonzero(d == -1) + 1).tolist()
    if m[0]:
        starts.insert(0, 0)
    if m[-1]:
        ends.append(m.size)
    return list(zip(starts, ends))


def make_find(parent):
    """경로 압축 union-find의 find 함수를 만들어 반환한다."""
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    return find


def point_to_polyline_dist(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """pts(M,2)의 각 점에서 polyline(N,2)까지의 최소 거리 (M,) 반환."""
    if len(poly) < 2:
        return np.full(len(pts), np.inf)
    seg_a = poly[:-1]
    seg_ab = poly[1:] - poly[:-1]
    seg_len2 = np.sum(seg_ab ** 2, axis=1)
    seg_len2[seg_len2 == 0] = 1e-9
    rel = pts[:, None, :] - seg_a[None, :, :]
    t = np.sum(rel * seg_ab[None, :, :], axis=2) / seg_len2[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = seg_a[None, :, :] + t[:, :, None] * seg_ab[None, :, :]
    d = np.linalg.norm(pts[:, None, :] - proj, axis=2)
    return d.min(axis=1)


def bodies_parallel(a: np.ndarray, b: np.ndarray, overlap_thr: float, lateral_thr: float) -> bool:
    """두 폴리라인 본체가 나란히(평행 이중선) 달리는지 검사."""
    allp = np.vstack([a, b])
    center = allp.mean(axis=0)
    _, _, vt = np.linalg.svd(allp - center)
    axis, perp = vt[0], vt[1]
    proj_a, proj_b = (a - center) @ axis, (b - center) @ axis
    amin, amax = proj_a.min(), proj_a.max()
    bmin, bmax = proj_b.min(), proj_b.max()
    inter = max(0.0, min(amax, bmax) - max(amin, bmin))
    shorter = min(amax - amin, bmax - bmin)
    overlap = inter / shorter if shorter > 1e-6 else 0.0
    lateral = abs(float(np.median((a - center) @ perp) - np.median((b - center) @ perp)))
    return overlap > overlap_thr and lateral < lateral_thr


def is_duplicate(a: np.ndarray, b: np.ndarray, dup_dist: float, dup_ratio: float) -> bool:
    """두 폴리라인이 거의 같은 위치에 겹쳐 그려진 복제인지 판정."""
    cover_ab = float(np.mean(point_to_polyline_dist(a, b) < dup_dist))
    cover_ba = float(np.mean(point_to_polyline_dist(b, a) < dup_dist))
    return max(cover_ab, cover_ba) >= dup_ratio


def hysteresis_free(dmin: np.ndarray, high: float, low: float) -> np.ndarray:
    """Canny 이중 임계로 free(겹치지 않음) 마스크 생성."""
    strong = dmin > high
    weak = dmin > low
    free = np.zeros(len(dmin), dtype=bool)
    for s, e in true_runs(weak):
        if strong[s:e].any():
            free[s:e] = True
    return free


def bridge_runs(pts: np.ndarray, free: np.ndarray, bridge_gap: float) -> np.ndarray:
    """free 구간 사이의 짧은 겹침 단절(호길이 bridge_gap 이하)을 free로 메운다(내부만)."""
    free = free.copy()
    n = len(free)
    for s, e in true_runs(~free):
        if 0 < s and e < n and arc_length(pts[s - 1:e + 1]) <= bridge_gap:
            free[s:e] = True
    return free


def subtract_lane(pts, refs, *, overlap_high, overlap_low, min_free_len, bridge_gap, step):
    """pts에서 기준선(refs) 중 하나라도 측면거리 이내인 점을 제거하고 남은 구간 리스트 반환."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return []
    if not refs:
        return [pts]
    pts = resample_polyline(pts, step)
    dmin = np.full(len(pts), np.inf)
    for r in refs:
        dmin = np.minimum(dmin, point_to_polyline_dist(pts, r))
    free = hysteresis_free(dmin, overlap_high, overlap_low)
    free = bridge_runs(pts, free, bridge_gap)
    pieces = []
    for s, e in true_runs(free):
        run = pts[s:e]
        if len(run) >= 2 and arc_length(run) >= min_free_len:
            pieces.append(run)
    return pieces


@dataclass
class EndSeg:
    """polyline 한쪽 끝의 확장 선분 정보."""
    tip: np.ndarray           # 끝점 (x, y)
    ref: np.ndarray           # 중심 방향 기준 노드
    ext: np.ndarray           # 바깥쪽으로 확장된 점
    direction: np.ndarray     # 바깥쪽(외측) 단위 방향 벡터 (tip - ref)
    pixels: Set[int] = field(default_factory=set)  # (ref -> ext) 선분의 래스터 픽셀 (y*W + x)


@dataclass
class Lane:
    """하나의 차선 polyline 인스턴스."""
    idx: int
    category: str
    category_id: int
    points: np.ndarray              # 순서가 있는 점들 (N, 2)
    ends: List[EndSeg] = field(default_factory=list)


class MergeAnnotator:
    extend_len = 10      # 끝점 바깥쪽 확장 길이 (px)
    ref_len = 20         # 끝점에서 안쪽 기준 노드까지의 길이 (px)
    seg_thickness = 5    # 겹침 검사용 확장 선분 두께
    overlap_thresh = 2   # 겹침으로 판정할 최소 픽셀 수
    dot_thresh = -0.5    # 방향이 반대라고 판정할 dot product 임계값
    align_thresh = 0.7   # 끝점 변위가 진행 방향과 정렬됐다고 볼 dot product 임계값(약 45도)
    align_eps = 2.0      # 두 끝점이 사실상 맞닿았다고 볼 거리(px)
    parallel_overlap = 0.5  # 본체가 나란하다고 판정할 종축(진행 방향) 겹침 비율(짧은 쪽 기준)
    parallel_lateral = 30   # 평행 이중선으로 볼 최대 측면 간격(px). 이보다 멀면 곡선 연장으로 간주
    dup_dist = 3.0       # 두 polyline이 '겹친 복제'인지 볼 점-본체 거리(px)
    dup_ratio = 0.8      # 한 선의 점이 상대 본체에 겹친 비율 임계 (이 이상이면 복제)
    trim_class_id = 1        # 겹침 정리(trim)를 적용할 클래스 (center_line). 진짜 이중선 보존을 위해 한정
    trim_step = 3.0          # trim 대상 선의 균일 재샘플 간격(px). 불규칙 점간격을 정규화
    overlap_dist = 6.0       # 갈라짐 강임계(px). 이보다 멀어진 구간을 '확실히 갈라짐'으로 본다(시드)
    overlap_low = 3.0        # 갈라짐 약임계(px). hysteresis로 시드 구간을 이 거리까지 확장(시작부 복원)
    min_free_len = 20.0      # 잘린 뒤 새 선으로 남길 최소 구간 길이(px)
    bridge_gap = 10.0        # 짧은 겹침 단절을 free로 메우는(브리징) 최대 길이(px)
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
            # 나란히 겹쳐 달리는 center_line은 긴 선을 기준으로 겹침 구간을 잘라낸다.
            trimmed = self._trim_overlaps(deduped)
            for lane in trimmed:
                self._build_ends(lane)

            merged = self._merge_lanes(trimmed)

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
        if len(seq) < 2:
            return tip, seq[-1].astype(np.float64)
        # 누적 호길이가 처음으로 target 이상이 되는 점이 기준 노드. 없으면 반대쪽 끝.
        acc = np.cumsum(np.linalg.norm(np.diff(seq, axis=0), axis=1))
        k = int(np.searchsorted(acc, target, side='left'))
        ref = seq[k + 1] if k < len(acc) else seq[-1]
        return tip, ref.astype(np.float64)

    def _rasterize_segment(self, p0: np.ndarray, p1: np.ndarray) -> Set[int]:
        """(p0 -> p1) 선분을 그린 뒤 이미지 범위 안 픽셀 인덱스(y*W+x) 집합 반환.

        전체 이미지가 아니라 선분 바운딩박스 크롭에만 그려서 메모리/시간을 줄인다.
        크롭 밖 픽셀을 이미지 경계로 필터링하므로 전체 이미지에 그린 것과 결과가 동일하다."""
        h, w = self._img_shape
        ax, ay = int(round(p0[0])), int(round(p0[1]))
        bx, by = int(round(p1[0])), int(round(p1[1]))
        pad = self.seg_thickness  # 두께가 선분 양옆으로 번지는 만큼 여유
        x0, y0 = min(ax, bx) - pad, min(ay, by) - pad
        cw = max(ax, bx) + pad - x0 + 1
        ch = max(ay, by) + pad - y0 + 1
        img = np.zeros((ch, cw), dtype=np.uint8)
        cv2.line(img, (ax - x0, ay - y0), (bx - x0, by - y0), color=255, thickness=self.seg_thickness)
        ys, xs = np.nonzero(img)
        gx, gy = xs + x0, ys + y0
        inb = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        return set((gy[inb] * w + gx[inb]).tolist())

    # ------------------------------------------------------------------ #
    # 중복 제거 (dedup) - 끝점 병합 이전 단계
    # ------------------------------------------------------------------ #
    def _dedup_lanes(self, lanes: List[Lane]) -> List[Lane]:
        """같은 클래스이면서 본체가 거의 일치하는(겹친 복제) polyline들을 하나로 합친다.

        SEED 원본은 한 마킹을 정/역방향 왕복 스트로크 등 거의 동일한 위치의
        여러 polyline으로 저장한 경우가 많다(특히 u_turn_zone_line).
        _bodies_parallel 가드는 떨어진 평행 이중선과 겹친 복제를 구분하지 못해
        이들을 끝점 병합으로 합치지 못하므로, 끝점 병합 전에 별도로 정리한다.
        간격이 큰 진짜 이중선(예: 이중 중앙선)은 dup_dist(작음)에 걸리지 않아 보존된다."""
        n = len(lanes)
        parent = list(range(n))
        find = self._make_find(parent)

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

        # 복제는 거의 같은 위치의 동일 선이므로 NN으로 합치면(점 엮임) 지그재그가 생긴다.
        # 점을 섞지 않고 가장 긴 대표선만 남기고 나머지는 버린다.
        deduped = []
        for members in groups.values():
            deduped.append(max(members, key=lambda m: self._arc_length(m.points)))
        return deduped

    def _is_duplicate(self, a: Lane, b: Lane) -> bool:
        return is_duplicate(a.points, b.points, self.dup_dist, self.dup_ratio)

    # ------------------------------------------------------------------ #
    # 병합
    # ------------------------------------------------------------------ #
    def _merge_lanes(self, lanes: List[Lane]) -> List[Lane]:
        n = len(lanes)
        parent = list(range(n))
        members = {i: [i] for i in range(n)}
        par_cache = {}
        find = self._make_find(parent)

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
        for group in groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged.append(self._merge_group(group))

        return merged

    # ------------------------------------------------------------------ #
    # 평행 겹침 trim - 겹치는 구간을 잘라 기준선/조각으로 분리
    # ------------------------------------------------------------------ #
    def _trim_overlaps(self, lanes: List[Lane]) -> List[Lane]:
        """겹쳐 달리는 center_line을 정리한다.

        더 긴 선을 기준선으로 그대로 두고, 짧은 선에서 기준선과 겹치는(측면거리
        overlap_dist 이내) 구간을 잘라낸다. 남은(겹치지 않는) 구간만 새 선으로
        등록하므로, 부분적으로 갈라지는 가지는 별도 선으로 남고 완전히 겹치는
        복제는 사라진다. 이후 끝-끝 병합이 기준선 범위 밖 조각을 직렬로 잇는다.

        진짜 이중 마킹 보존을 위해 center_line(trim_class_id)에만 적용한다.
        점을 평균내지 않고 잘라내기만 하므로 지그재그가 생기지 않는다."""
        targets = [l for l in lanes if l.category_id == self.trim_class_id]
        others = [l for l in lanes if l.category_id != self.trim_class_id]
        # 긴 선부터 기준선으로 확정해 나간다.
        targets.sort(key=lambda l: self._arc_length(l.points), reverse=True)

        kept: List[Lane] = []
        for lane in targets:
            for piece in self._subtract_lane(lane.points, kept):
                kept.append(Lane(idx=lane.idx, category=lane.category,
                                 category_id=lane.category_id, points=piece))
        return others + kept

    def _subtract_lane(self, pts: np.ndarray, refs: List[Lane]) -> List[np.ndarray]:
        return subtract_lane(
            pts, [r.points for r in refs],
            overlap_high=self.overlap_dist, overlap_low=self.overlap_low,
            min_free_len=self.min_free_len, bridge_gap=self.bridge_gap, step=self.trim_step)

    @staticmethod
    def _make_find(parent: List[int]):
        return make_find(parent)

    @staticmethod
    def _arc_length(points: np.ndarray) -> float:
        return arc_length(points)

    def _should_merge(self, a: Lane, b: Lane) -> bool:
        """두 polyline이 끝과 끝으로 이어지면 True.

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
        return bodies_parallel(a.points, b.points, self.parallel_overlap, self.parallel_lateral)

    def _merge_group(self, members: List[Lane]) -> Lane:
        """여러 polyline의 점들을 통합한 뒤 순서대로 정렬한다."""
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
            # COCO segmentation과 동일하게 두께 3, 양 끝에 동그란 점 (figure와 공유 함수)
            color = cfg.ID2BGR.get(lane.category_id, (255, 255, 255))
            fr.draw_strand(canvas, lane.points, color, thickness=self.mask_thickness)
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
        """polyline을 두께 3으로 그린 영역을 RLE segmentation으로 저장.
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


def count_before_after(splits: List[str]):
    """merge 전(raw lane) vs 후(merged annotation) 개체수를 split별로 집계·출력한다.

    BEFORE: SEED_LABEL_PATH 원본 라벨 json의 유효 lane 수(_load_lanes와 동일 기준:
            RoadObject·LINE_STRING·점≥2·METAINFO 카테고리).
    AFTER : merged_annotations_{split}.json의 annotation 수.
    병합을 재실행하지 않고 파일만 읽어 계산한다."""
    name2id = {c['name']: c['id'] for c in cfg.METAINFO}
    with open(cfg.DATASET_SPLIT_JSON, 'r') as f:
        dataset = json.load(f)

    before = {sp: 0 for sp in splits}
    after = {sp: 0 for sp in splits}
    for sp in splits:
        for base in dataset.get(sp, []):
            json_file = os.path.join(cfg.SEED_LABEL_PATH, base + '.json')
            if not os.path.exists(json_file):
                continue
            with open(json_file, 'r') as f:
                data = json.load(f)
            before[sp] += sum(
                1 for o in data
                if o.get('class') == 'RoadObject' and o.get('geometry_type') == 'LINE_STRING'
                and o.get('image_points') is not None and len(o['image_points']) >= 2
                and o.get('category') in name2id)
        merged_path = _split_coco_path(sp)
        if os.path.exists(merged_path):
            with open(merged_path, 'r') as f:
                after[sp] = len(json.load(f)['annotations'])

    total_b, total_a = sum(before.values()), sum(after.values())
    print(f"\n{'split':12}{'before':>10}{'after':>10}{'reduced':>10}{'reduce%':>9}")
    for sp in splits:
        b, a = before[sp], after[sp]
        pct = (1 - a / b) * 100 if b else 0.0
        print(f"{sp:12}{b:>10}{a:>10}{b - a:>10}{pct:>8.1f}%")
    pct = (1 - total_a / total_b) * 100 if total_b else 0.0
    print(f"{'TOTAL':12}{total_b:>10}{total_a:>10}{total_b - total_a:>10}{pct:>8.1f}%")
    return before, after


def main():
    with open(cfg.DATASET_SPLIT_JSON, 'r') as f:
        dataset = json.load(f)

    # validation을 먼저 처리한 뒤 train을 처리한다.
    splits = ['validation', 'train']
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
