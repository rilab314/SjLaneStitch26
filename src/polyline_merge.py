"""순서가 있는 폴리라인(점 배열 (N,2))에 대한 공유 병합/정리 연산.

lane_stitcher(LaneStitch, 세그멘테이션 벡터화)와 merge_annotation(GT 통합)이 공통으로 쓰는
기하 유틸과 지그재그 제거 알고리즘을 모은 순수 함수 모듈이다. 데이터 소스에 무관하게
"이미 순서가 있는 폴리라인"만 입력으로 받는다(추출/로딩은 각 호출측 책임).

핵심 아이디어
- dedup_keep_longest: 거의 같은 위치의 복제선은 NN으로 합치지 말고 가장 긴 것만 남긴다.
- trim_overlaps: 평행하게 겹쳐 달리는 선은 가장 긴 기준선을 그대로 두고 짧은 선의 겹침
  구간만 잘라낸다(Canny식 이중 임계 hysteresis로 갈라짐 시작부 복원). 점을 평균내지 않아
  지그재그가 생기지 않고, 갈라지는 가지는 별도 선으로 보존된다.
- concat_polylines_in_series: 끝-끝으로 이어지는 폴리라인들을 점 단위 NN이 아니라
  폴리라인 단위로 체이닝해 이어 붙인다(각 선의 내부 점 순서를 보존 → 지그재그 없음).
"""

import numpy as np


# ------------------------------------------------------------------ #
# 기하 유틸
# ------------------------------------------------------------------ #
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


def smooth_polyline(points: np.ndarray, window: int = 5, iterations: int = 1) -> np.ndarray:
    """폴리라인 점들을 이동평균으로 평활화한다(끝점은 고정).

    지그재그(thinning/병합 경계 꺾임)를 완화해 선을 반듯하게 편다. 끝점을 고정해
    선이 짧아지거나 끝이 말려드는 것을 막는다."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= 2 or window < 3:
        return pts
    w = min(window, n - 1 if n % 2 == 0 else n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return pts
    half = w // 2
    kernel = np.ones(w) / w
    out = pts.copy()
    for _ in range(max(1, iterations)):
        padded = np.pad(out, ((half, half), (0, 0)), mode='edge')
        sx = np.convolve(padded[:, 0], kernel, mode='valid')
        sy = np.convolve(padded[:, 1], kernel, mode='valid')
        out = np.stack([sx, sy], axis=1)
        out[0], out[-1] = pts[0], pts[-1]  # 끝점 고정
    return out


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
    """pts(M,2)의 각 점에서 polyline(N,2)까지의 최소 거리 (M,) 반환.
    선분 위로 투영(클램프)하여 점-선분 거리를 계산한다."""
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
    """두 폴리라인 본체가 나란히(평행 이중선) 달리는지 검사.

    공통 주축(SVD 1번 축)으로 투영했을 때 종축 구간이 overlap_thr 이상 겹치고
    측면 간격이 lateral_thr 미만이면 평행으로 본다. 끝-끝 연장(collinear)은 서로 다른
    종축 구간을 차지해 겹침이 0에 가까우므로 구분된다."""
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
    """두 폴리라인이 거의 같은 위치에 겹쳐 그려진 복제인지 판정.
    한쪽 점들의 dup_ratio 이상이 상대 본체에서 dup_dist 이내이면 복제로 본다."""
    cover_ab = float(np.mean(point_to_polyline_dist(a, b) < dup_dist))
    cover_ba = float(np.mean(point_to_polyline_dist(b, a) < dup_dist))
    return max(cover_ab, cover_ba) >= dup_ratio


# ------------------------------------------------------------------ #
# dedup - 복제선은 가장 긴 대표만 남긴다 (점을 섞지 않음)
# ------------------------------------------------------------------ #
def dedup_keep_longest(polys, dup_dist: float, dup_ratio: float):
    """같은 클래스 폴리라인 리스트에서 복제 그룹마다 가장 긴 것의 인덱스만 남겨 반환한다.
    (점을 NN으로 합치면 지그재그가 생기므로 대표선만 남기고 나머지는 버린다.)"""
    n = len(polys)
    parent = list(range(n))
    find = make_find(parent)
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if is_duplicate(polys[i], polys[j], dup_dist, dup_ratio):
                parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [max(idxs, key=lambda k: arc_length(polys[k])) for idxs in groups.values()]


# ------------------------------------------------------------------ #
# trim - 평행 겹침 구간을 잘라 기준선/조각으로 분리 (hysteresis)
# ------------------------------------------------------------------ #
def hysteresis_free(dmin: np.ndarray, high: float, low: float) -> np.ndarray:
    """Canny 이중 임계로 free(겹치지 않음) 마스크 생성.
    강임계(high) 초과를 시드로, 약임계(low) 연결 구간이 시드를 포함하면 전체 채택."""
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
    """pts에서 기준선(refs: 점배열 리스트) 중 하나라도 측면거리 이내인 점을 제거하고,
    남은 연속 구간(점배열 리스트)을 반환한다. refs가 비면 원본을 그대로 보존한다."""
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


def trim_overlaps(polys, *, overlap_high, overlap_low, min_free_len, bridge_gap, step):
    """같은 클래스 폴리라인 리스트를 길이 내림차순으로 trim한다.
    각 결과 조각을 (원본 인덱스, 점배열)로 반환해 호출측이 메타데이터를 매핑할 수 있게 한다."""
    out = []
    kept = []
    for i in sorted(range(len(polys)), key=lambda k: arc_length(polys[k]), reverse=True):
        for piece in subtract_lane(polys[i], kept, overlap_high=overlap_high,
                                   overlap_low=overlap_low, min_free_len=min_free_len,
                                   bridge_gap=bridge_gap, step=step):
            kept.append(piece)
            out.append((i, piece))
    return out


# ------------------------------------------------------------------ #
# 직렬 연결 - 끝-끝 폴리라인을 점이 아니라 폴리라인 단위로 체이닝
# ------------------------------------------------------------------ #
def concat_polylines_in_series(polys):
    """끝-끝으로 이어지는 폴리라인들을 한 줄로 잇는다.

    점 단위 NN 정렬(merge_annotation의 옛 _order_points)과 달리, 각 폴리라인의 내부 점
    순서를 그대로 보존하고 폴리라인 사이만 연결한다. 따라서 평행 가닥을 잘못 엮어 생기던
    지그재그가 발생하지 않는다. 가장 바깥 끝점에서 시작해 가장 가까운 끝점을 가진
    폴리라인을 방향을 맞춰가며 차례로 이어 붙인다."""
    polys = [np.asarray(p, dtype=np.float64) for p in polys if len(p) >= 1]
    if len(polys) == 1:
        return polys[0]
    if not polys:
        return np.empty((0, 2), dtype=np.float64)

    # 시작 폴리라인/끝점: 모든 끝점 중 전체 무게중심에서 가장 먼 점
    centroid = np.vstack(polys).mean(axis=0)
    remaining = list(range(len(polys)))
    best = max(((i, end) for i in remaining for end in (0, -1)),
               key=lambda ie: np.linalg.norm(polys[ie[0]][ie[1]] - centroid))
    start_i, start_end = best
    # 시작 끝점이 머리로 오도록 정렬
    chain = polys[start_i] if start_end == 0 else polys[start_i][::-1]
    remaining.remove(start_i)

    while remaining:
        tail = chain[-1]
        # 남은 폴리라인의 양 끝점 중 현재 tail에 가장 가까운 것을 선택
        cand = min(((i, end) for i in remaining for end in (0, -1)),
                   key=lambda ie: np.linalg.norm(polys[ie[0]][ie[1]] - tail))
        i, end = cand
        seg = polys[i] if end == 0 else polys[i][::-1]  # 가까운 끝이 머리가 되도록
        chain = np.vstack([chain, seg])
        remaining.remove(i)
    return chain
