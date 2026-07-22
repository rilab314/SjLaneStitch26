"""
Geometry engine that merges fragmented lane annotations (lane_merger)

In satellite-image lane annotations (json), what is visually a single lane is sometimes
stored split into several polyline objects.
This module connects the split polylines to form unified lane objects.

It is a library: dataprep/merge_annotation.py drives it to turn the raw SEED release into the
merged one, and nothing else in the pipeline merges annotations.

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

merge_image() returns the (raw, merged) lane lists of one image; run() walks a whole split and
writes the before/after comparison images used to inspect the merge visually.
During development, dev_time_limit can process only a subset (None processes everything).
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Set

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path
import config as cfg
import seed_label
import figure_render as fr


# ====================================================================== #
# Pure geometry/cleanup functions.
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
    source: dict = None             # SEED object the lane originates from (attributes on rewrite)
    ends: List[EndSeg] = field(default_factory=list)


class LaneMerger:
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
    mask_thickness = 3   # lane rendering thickness of the before/after comparison image
    dev_time_limit = None  # dev run time limit (seconds). None processes the entire dataset

    def __init__(self, split: str, image_ids: List[str], label_path: str,
                 image_path: str, compare_path: str = None, write_compare: bool = True):
        self._split = split
        self._image_ids = image_ids
        self._label_path = label_path
        self._image_path = image_path
        self._compare_path = compare_path
        self._write_compare = write_compare  # save before/after overlay PNGs (off when only geometry is wanted)
        self._img_shape = (768, 768)
        if self._write_compare:
            os.makedirs(self._compare_path, exist_ok=True)
        self._enabled = True       # whether to use imshow

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def run(self):
        """Merge every image of the split and write the before/after comparison images."""
        print(f'[run] split={self._split} target images: {len(self._image_ids)}')

        # For full-dataset batch runs, turn off imshow and show only progress.
        full_run = self.dev_time_limit is None
        if full_run:
            self._enabled = False

        total_before = total_after = processed = 0
        start_time = time.time()
        iterator = tqdm(self._image_ids, desc=f'merge[{self._split}]') if full_run else self._image_ids
        for base in iterator:
            lanes, merged = self.merge_image(base)
            if not lanes:
                continue

            total_before += len(lanes)
            total_after += len(merged)
            processed += 1
            if not full_run:
                print(f'[{processed}] {base}: {len(lanes)} -> {len(merged)}')
            self._show_compare(base, lanes, merged)

            if self.dev_time_limit is not None and (time.time() - start_time) > self.dev_time_limit:
                print(f'\n[run] dev time limit ({self.dev_time_limit}s) reached -> stopping')
                break

        print(f'\n===== summary (split={self._split}) =====')
        print(f'processed images: {processed}')
        print(f'lane objects: {total_before} -> {total_after} '
              f'({total_before - total_after} reduced)')
        if self._enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    def merge_image(self, base: str):
        """Merge the lanes of one image and return the (raw, merged) lane lists.

        This is the public geometry API: merge_annotation stores the merged lanes back into the
        SEED format, run() only visualizes them."""
        json_file = os.path.join(self._label_path, base + '.json')
        if not os.path.exists(json_file):
            return [], []
        lanes = self._load_lanes(json_file)
        if len(lanes) == 0:
            return [], []

        self._read_image_shape(base)
        # Before endpoint merging, first consolidate overlapping duplicates (round-trip strokes, etc.).
        deduped = self._dedup_lanes(lanes)
        # For center_lines running side by side, cut the overlapping segments using the longer line as reference.
        trimmed = self._trim_overlaps(deduped)
        for lane in trimmed:
            self._build_ends(lane)
        return lanes, self._merge_lanes(trimmed)

    # ------------------------------------------------------------------ #
    # data loading
    # ------------------------------------------------------------------ #
    def _load_lanes(self, json_file: str) -> List[Lane]:
        """Read the SEED lane polylines (shared filter) as Lane instances."""
        return [Lane(idx=idx, category=seed.category, category_id=seed.category_id,
                     points=seed.points, source=seed.source)
                for idx, seed in enumerate(seed_label.load_lane_objects(json_file))]

    def _read_image_shape(self, base: str):
        """Keep the canvas size of the current image (the previous size is kept if unreadable)."""
        image = cv2.imread(os.path.join(self._image_path, base + '.png'))
        if image is not None:
            self._img_shape = image.shape[:2]

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
                                 category_id=lane.category_id, points=piece,
                                 source=lane.source))
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
        return Lane(idx=base.idx, category=base.category, category_id=base.category_id,
                    points=ordered, source=base.source)

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
    def _show_compare(self, base: str, before: List[Lane], after: List[Lane]):
        """Save and/or display the before|after overlay of one image (skipped when both are off)."""
        if not (self._write_compare or self._enabled):
            return
        image = cv2.imread(os.path.join(self._image_path, base + '.png'))
        compare = self._make_compare_image(image, before, after)
        if self._write_compare:
            cv2.imwrite(os.path.join(self._compare_path, base + '.png'), compare)
        if self._enabled:
            try:
                cv2.imshow('merge compare (before | after)', compare)
                cv2.waitKey(1)
            except cv2.error:
                self._enabled = False

    def _make_compare_image(self, image, before: List[Lane], after: List[Lane]):
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
