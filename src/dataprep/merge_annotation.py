"""
Merge fragmented lane annotations (merge_annotation)

In satellite-image lane annotations (json), what is visually a single lane is sometimes
stored split into several polyline objects.
This script connects the split polylines to form unified lane objects.

It borrows the endpoint-overlap merging idea from LaneStitcher (lane_stitcher.py), but
because different objects may cross (like an overpass and the crossing road beneath it),
it merges only when the endpoint extension lines overlap AND the "directions are opposite".

Algorithm overview (per image)
1. Load the polylines from json into a list of Lane objects.
2. dedup: nearly co-located duplicate lines (round-trip strokes, etc.) are grouped with union-find,
   keeping only the longest and discarding the rest (points are not mixed, so no zigzag arises).
3. trim (center_line only): using the longest line as the reference, cut the segments of shorter lines
   that overlap the reference via a Canny-style double threshold (strong overlap_dist / weak overlap_low).
   Diverging branches remain as separate lines and fully-overlapping duplicates disappear.
4. endpoint merge: build segments extending both ends of each line outward by extend_len, and join
   end-to-end in series when they are the same class AND the extension segments overlap + the outward
   directions are opposite + the ends are aligned. Pairs whose bodies run side by side (parallel double
   lines) are excluded.
5. Save the unified result as COCO instance segmentation json and leave before/after comparison images.

During development, dev_time_limit can process only a subset (None processes everything).
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg
import figure_render as fr


# ====================================================================== #
# Pure geometry/cleanup functions (merge_annotation carries its own copy of the old polyline_merge module)
# Polyline utilities used by dedup(is_duplicate) and trim(subtract_lane).
# ====================================================================== #
def arc_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(points, float), axis=0), axis=1).sum())


def resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    """Resample an ordered polyline at uniform arc-length intervals (step)."""
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
    """Return a list of (start, end) runs (end exclusive) of consecutive True in a boolean array."""
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
    """Create and return the find function of a path-compressed union-find."""
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    return find


def point_to_polyline_dist(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Return the minimum distance (M,) from each point of pts(M,2) to the polyline(N,2)."""
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
    """Check whether the two polyline bodies run side by side (parallel double line)."""
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
    """Decide whether the two polylines are duplicates drawn overlapping at nearly the same location."""
    cover_ab = float(np.mean(point_to_polyline_dist(a, b) < dup_dist))
    cover_ba = float(np.mean(point_to_polyline_dist(b, a) < dup_dist))
    return max(cover_ab, cover_ba) >= dup_ratio


def hysteresis_free(dmin: np.ndarray, high: float, low: float) -> np.ndarray:
    """Build the free (non-overlapping) mask using a Canny-style double threshold."""
    strong = dmin > high
    weak = dmin > low
    free = np.zeros(len(dmin), dtype=bool)
    for s, e in true_runs(weak):
        if strong[s:e].any():
            free[s:e] = True
    return free


def bridge_runs(pts: np.ndarray, free: np.ndarray, bridge_gap: float) -> np.ndarray:
    """Fill short overlap breaks between free runs (arc length <= bridge_gap) as free (interior only)."""
    free = free.copy()
    n = len(free)
    for s, e in true_runs(~free):
        if 0 < s and e < n and arc_length(pts[s - 1:e + 1]) <= bridge_gap:
            free[s:e] = True
    return free


def subtract_lane(pts, refs, *, overlap_high, overlap_low, min_free_len, bridge_gap, step):
    """Remove points in pts that are within lateral distance of any reference line (refs) and return the list of remaining runs."""
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
    """Extension-segment info for one end of a polyline."""
    tip: np.ndarray           # endpoint (x, y)
    ref: np.ndarray           # reference node toward the center
    ext: np.ndarray           # point extended outward
    direction: np.ndarray     # outward unit direction vector (tip - ref)
    pixels: Set[int] = field(default_factory=set)  # raster pixels of the (ref -> ext) segment (y*W + x)


@dataclass
class Lane:
    """A single lane polyline instance."""
    idx: int
    category: str
    category_id: int
    points: np.ndarray              # ordered points (N, 2)
    ends: List[EndSeg] = field(default_factory=list)


class MergeAnnotator:
    extend_len = 10      # outward endpoint extension length (px)
    ref_len = 20         # length from the endpoint to the inner reference node (px)
    seg_thickness = 5    # extension-segment thickness for overlap checking
    overlap_thresh = 2   # minimum number of pixels to judge as overlapping
    dot_thresh = -0.5    # dot-product threshold for judging directions as opposite
    align_thresh = 0.7   # dot-product threshold to treat the endpoint displacement as aligned with the travel direction (about 45 degrees)
    align_eps = 2.0      # distance to treat two endpoints as effectively touching (px)
    parallel_overlap = 0.5  # longitudinal (travel-direction) overlap ratio to judge bodies as side by side (relative to the shorter one)
    parallel_lateral = 30   # maximum lateral gap to treat as a parallel double line (px). Farther than this is treated as a curved extension
    dup_dist = 3.0       # point-to-body distance to see whether two polylines are an 'overlapping duplicate' (px)
    dup_ratio = 0.8      # ratio threshold of one line's points overlapping the other body (above this is a duplicate)
    trim_class_id = 1        # class to which overlap cleanup (trim) is applied (center_line). Limited to preserve genuine double lines
    trim_step = 3.0          # uniform resampling interval for the trim-target line (px). Normalizes irregular point spacing
    overlap_dist = 6.0       # divergence strong threshold (px). Segments farther than this are seen as 'definitely diverging' (seed)
    overlap_low = 3.0        # divergence weak threshold (px). Extend the seed run up to this distance via hysteresis (restores the start portion)
    min_free_len = 20.0      # minimum run length to keep as a new line after cutting (px)
    bridge_gap = 10.0        # maximum length for filling (bridging) a short overlap break as free (px)
    mask_thickness = 3   # COCO mask rendering thickness
    dev_time_limit = None  # dev run time limit (seconds). None processes the entire dataset

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

        # SEED category name -> METAINFO class id (matches the output format to the GT)
        self._name2id = {c['name']: c['id'] for c in cfg.METAINFO}
        self._enabled = True       # whether to use imshow

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def run(self):
        print(f'[run] split={self._split} target images: {len(self._image_ids)}')

        # For full-dataset batch conversion, turn off imshow and show only progress.
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
            image_id = base  # use the basename string as the image id, same as the GT format
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

            # Before endpoint merging, first consolidate overlapping duplicates (round-trip strokes, etc.).
            deduped = self._dedup_lanes(lanes)
            # For center_lines running side by side, cut the overlapping segments using the longer line as reference.
            trimmed = self._trim_overlaps(deduped)
            for lane in trimmed:
                self._build_ends(lane)

            merged = self._merge_lanes(trimmed)

            total_before += len(lanes)
            total_after += len(merged)
            processed += 1
            if not full_run:
                print(f'[{processed}] {base}: {len(lanes)} -> {len(merged)}')

            # COCO accumulation (same format as the GT merged_annotations.json)
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

            # display and save the comparison image
            compare = self._make_compare_image(image, lanes, merged, base)
            self._save_compare(compare, base)
            if self._enabled:
                try:
                    cv2.imshow('merge compare (before | after)', compare)
                    cv2.waitKey(1)
                except cv2.error:
                    self._enabled = False

            if self.dev_time_limit is not None and (time.time() - start_time) > self.dev_time_limit:
                print(f'\n[run] dev time limit ({self.dev_time_limit}s) reached -> stopping')
                break

        self._save_coco(coco_images, coco_annotations)

        print(f'\n===== summary (split={self._split}) =====')
        print(f'processed images: {processed}')
        print(f'lane objects: {total_before} -> {total_after} '
              f'({total_before - total_after} reduced)')
        if self._enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    # ------------------------------------------------------------------ #
    # data loading
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
            # exclude categories not in METAINFO (e.g. others) since they are not evaluated
            if category not in self._name2id:
                continue
            cat_id = self._name2id[category]
            lanes.append(Lane(idx=idx, category=category, category_id=cat_id, points=points))
            idx += 1
        return lanes

    # ------------------------------------------------------------------ #
    # endpoint extension (extrapolation)
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
        """Find the reference node target px away from the endpoint toward the center.
        If the line is short, use the farthest reachable point (the opposite end)."""
        seq = points if from_head else points[::-1]
        tip = seq[0].astype(np.float64)
        if len(seq) < 2:
            return tip, seq[-1].astype(np.float64)
        # The reference node is the first point where cumulative arc length reaches target. If none, the opposite end.
        acc = np.cumsum(np.linalg.norm(np.diff(seq, axis=0), axis=1))
        k = int(np.searchsorted(acc, target, side='left'))
        ref = seq[k + 1] if k < len(acc) else seq[-1]
        return tip, ref.astype(np.float64)

    def _rasterize_segment(self, p0: np.ndarray, p1: np.ndarray) -> Set[int]:
        """Draw the (p0 -> p1) segment and return the set of pixel indices (y*W+x) within the image bounds.

        Draw only on a crop of the segment's bounding box rather than the whole image to save memory/time.
        Pixels outside the crop are filtered by the image bounds, so the result is identical to drawing on the whole image."""
        h, w = self._img_shape
        ax, ay = int(round(p0[0])), int(round(p0[1]))
        bx, by = int(round(p1[0])), int(round(p1[1]))
        pad = self.seg_thickness  # margin for how far the thickness spreads on both sides of the segment
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
    # deduplication (dedup) - stage before endpoint merging
    # ------------------------------------------------------------------ #
    def _dedup_lanes(self, lanes: List[Lane]) -> List[Lane]:
        """Merge polylines of the same class whose bodies nearly coincide (overlapping duplicates) into one.

        The SEED originals often store one marking as several polylines at nearly the same location,
        such as forward/backward round-trip strokes (especially u_turn_zone_line).
        The _bodies_parallel guard cannot distinguish a separated parallel double line from an overlapping
        duplicate, so it fails to merge these via endpoint merging; therefore clean them up separately before it.
        Genuine double lines with a large gap (e.g. a double center line) do not trigger dup_dist (small) and are preserved."""
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

        # Duplicates are identical lines at nearly the same location, so merging by NN (interleaving points) causes zigzag.
        # Without mixing points, keep only the longest representative line and discard the rest.
        deduped = []
        for members in groups.values():
            deduped.append(max(members, key=lambda m: self._arc_length(m.points)))
        return deduped

    def _is_duplicate(self, a: Lane, b: Lane) -> bool:
        return is_duplicate(a.points, b.points, self.dup_dist, self.dup_ratio)

    # ------------------------------------------------------------------ #
    # merge
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

        # collect merge candidate pairs among the same class
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if lanes[i].category_id != lanes[j].category_id:
                    continue
                if self._should_merge(lanes[i], lanes[j]):
                    candidates.append((i, j))

        # group-aware union: when merging two groups, reject if any cross-member pair is parallel.
        # (prevents a short connecting segment from transitively joining two parallel
        #  strands at a junction, which would produce a zigzag)
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

        # group together
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
    # parallel-overlap trim - cut overlapping segments to separate reference line/pieces
    # ------------------------------------------------------------------ #
    def _trim_overlaps(self, lanes: List[Lane]) -> List[Lane]:
        """Clean up center_lines running overlapping.

        Keep the longer line as the reference line, and cut from the shorter line the segments that overlap
        the reference (within lateral distance overlap_dist). Only the remaining (non-overlapping) segments are
        registered as new lines, so a partially diverging branch remains as a separate line and a fully
        overlapping duplicate disappears. The subsequent end-to-end merge joins pieces outside the reference
        line's range in series.

        Applied only to center_line (trim_class_id) to preserve genuine double markings.
        It only cuts rather than averaging points, so no zigzag arises."""
        targets = [l for l in lanes if l.category_id == self.trim_class_id]
        others = [l for l in lanes if l.category_id != self.trim_class_id]
        # Confirm reference lines starting from the longest line.
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
        """True if the two polylines connect end to end.

        Conditions: the extension segments overlap (overlap), the two ends' outward directions are opposite
        (direction), and the displacement joining the two endpoints must be aligned along the same axis as
        the travel direction (alignment). Without the alignment condition, side-by-side parallel double center
        lines would be wrongly merged into a zigzag, so it is essential."""
        for ea in a.ends:
            for eb in b.ends:
                if len(ea.pixels & eb.pixels) < self.overlap_thresh:
                    continue
                if float(np.dot(ea.direction, eb.direction)) >= self.dot_thresh:
                    continue  # directions are not opposite
                # check whether the endpoint displacement is aligned with the travel direction (exclude parallel double lines)
                gap = eb.tip - ea.tip
                dist = float(np.linalg.norm(gap))
                if dist > self.align_eps:
                    along = float(np.dot(gap / dist, ea.direction))
                    if along < self.align_thresh:
                        continue  # a laterally side-by-side (perpendicular displacement) parallel line -> do not merge
                # endpoints passed, but do not merge if the two bodies run side by side close together
                if self._bodies_parallel(a, b):
                    return False
                return True
        return False

    def _bodies_parallel(self, a: Lane, b: Lane) -> bool:
        return bodies_parallel(a.points, b.points, self.parallel_overlap, self.parallel_lateral)

    def _merge_group(self, members: List[Lane]) -> Lane:
        """Combine the points of several polylines and then order them sequentially."""
        all_points = np.vstack([m.points for m in members])
        ordered = self._order_points(all_points, members)
        base = members[0]
        return Lane(idx=base.idx, category=base.category,
                    category_id=base.category_id, points=ordered)

    @staticmethod
    def _order_points(points: np.ndarray, members: List[Lane]) -> np.ndarray:
        """Order the combined points sequentially via nearest-neighbor chaining.
        The start point is chosen as the point farthest from the center among all endpoints (tips) in the group."""
        # determine the start point
        tips = []
        for m in members:
            tips.append(m.points[0])
            tips.append(m.points[-1])
        tips = np.array(tips, dtype=np.float64)
        centroid = points.mean(axis=0)
        start = tips[np.argmax(np.linalg.norm(tips - centroid, axis=1))]

        remaining = points.astype(np.float64).copy()
        # start from the actual point closest to the start point
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
    # comparison image (before/after)
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
            # thickness 3 with round dots at both ends, same as the COCO segmentation (function shared with figure)
            color = cfg.ID2BGR.get(lane.category_id, (255, 255, 255))
            fr.draw_strand(canvas, lane.points, color, thickness=self.mask_thickness)
        cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        return canvas

    def _save_compare(self, compare, base: str):
        out_path = os.path.join(self._compare_path, base + '.png')
        cv2.imwrite(out_path, compare)

    # ------------------------------------------------------------------ #
    # COCO output
    # ------------------------------------------------------------------ #
    def _lane_to_annotation(self, lane: Lane, image_id: str):
        """Save the region drawn from the polyline with thickness 3 as an RLE segmentation.
        Same as the GT merged_annotations.json, contains only image_id/category_id/segmentation/score."""
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
        print(f'[save] merged_annotations json saved: {self._coco_path} '
              f'(images={len(images)}, annotations={len(annotations)}, '
              f'categories={len(categories)})')


def _split_coco_path(split: str) -> str:
    """Per-split merged_annotations_{split}.json path (inside the result folder RESULT_PATH)."""
    return cfg.coco_anno_path(split)


def count_class_instances(splits: List[str], csv_path: str):
    """Read the per-split merged json, compute per-class instance counts, and save as csv.

    Rows: class (name), columns: split. Add a total row/column at the end."""
    id2name = {c['id']: c['name'] for c in cfg.METAINFO}
    # use as rows only the actual lane classes used for evaluation (excluding ignore)
    class_ids = [c['id'] for c in cfg.METAINFO if c['id'] != 0]

    # counts[split][class_id]
    counts = {sp: {cid: 0 for cid in class_ids} for sp in splits}
    for sp in splits:
        path = _split_coco_path(sp)
        if not os.path.exists(path):
            print(f'[count] {path} not found -> skipping {sp}')
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
    print(f'[count] per-class instance counts csv saved: {csv_path}')


def count_before_after(splits: List[str]):
    """Aggregate and print the instance counts before (raw lane) vs after (merged annotation) merging, per split.

    BEFORE: the number of valid lanes in the SEED_LABEL_PATH original label json (same criteria as _load_lanes:
            RoadObject / LINE_STRING / points>=2 / METAINFO categories).
    AFTER : the number of annotations in merged_annotations_{split}.json.
    Computed by reading files only, without rerunning the merge."""
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

    # Generate the COCO GT in the result folder for the evaluation splits (default: validation, test).
    # Images and SEED labels are looked up by basename from the original folders (SRC_*) where splits are mixed.
    splits = cfg.EVAL_SPLITS
    for split in splits:
        annotator = MergeAnnotator(
            split=split,
            image_ids=sorted(dataset[split]),
            label_path=cfg.SEED_LABEL_PATH,
            image_path=cfg.SRC_IMAGE_DIR,
            compare_path=cfg.merge_compare_dir(split),
            coco_path=_split_coco_path(split),
        )
        annotator.run()

    # After both tasks finish, compute per-class instance counts with a separate function and save as csv.
    csv_path = os.path.join(cfg.RESULT_PATH, 'class_counts.csv')
    count_class_instances(splits, csv_path)


if __name__ == '__main__':
    main()
