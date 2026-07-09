import os
import sys
import glob

import cv2
import numpy as np
import json
import copy
from pycocotools import mask as maskUtils
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass
from tqdm import tqdm

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import _bootstrap  # noqa: F401  # registers core/tables/figures on sys.path

from show_imgs import ImageShow
import config as cfg


# ====================================================================== #
# Pure geometry/merge functions (lane_stitch now holds these itself, from the former polyline_merge module)
# Utilities for trim (overlap removal) / series concatenation on ordered polylines (point array (N,2)).
# ====================================================================== #
def arc_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(points, float), axis=0), axis=1).sum())


def resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    """Resample an ordered polyline at a uniform arc-length interval (step)."""
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
    """Smooth the polyline points with a moving average (endpoints fixed)."""
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
        out[0], out[-1] = pts[0], pts[-1]  # fix endpoints
    return out


def true_runs(mask: np.ndarray):
    """Return a list of (start, end) ranges of consecutive True segments in a boolean array (end exclusive)."""
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
    """Build and return the find function of a path-compressed union-find."""
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    return find


def point_to_polyline_dist(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Return the minimum distance (M,) from each point of pts(M,2) to polyline(N,2)."""
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


def hysteresis_free(dmin: np.ndarray, pts: np.ndarray, high: float, low: float,
                    min_diverge_len: float) -> np.ndarray:
    """Create the free (non-overlapping) mask with Canny dual thresholds.

    A weak (>low) segment is accepted as free only when strong (>high) is maintained 'continuously'
    within it for at least an arc length of min_diverge_len. This prevents a single strong pixel from
    keeping an entire weak segment alive and producing fragmented double lines (with min_diverge_len=0,
    a single strong point is accepted as before)."""
    strong = dmin > high
    weak = dmin > low
    free = np.zeros(len(dmin), dtype=bool)
    for s, e in true_runs(weak):
        if _has_sustained_strong(strong[s:e], pts[s:e], min_diverge_len):
            free[s:e] = True
    return free


def _has_sustained_strong(strong_run: np.ndarray, pts_run: np.ndarray, min_len: float) -> bool:
    """Whether the strong mask contains at least one consecutive True segment with an arc length >= min_len."""
    if min_len <= 0:
        return bool(strong_run.any())
    for a, b in true_runs(strong_run):
        if arc_length(pts_run[a:b]) >= min_len:
            return True
    return False


def bridge_runs(pts: np.ndarray, free: np.ndarray, bridge_gap: float) -> np.ndarray:
    """Fill short overlap breaks between free segments (arc length <= bridge_gap) with free (interior only)."""
    free = free.copy()
    n = len(free)
    for s, e in true_runs(~free):
        if 0 < s and e < n and arc_length(pts[s - 1:e + 1]) <= bridge_gap:
            free[s:e] = True
    return free


def subtract_lane(pts, refs, *, overlap_high, overlap_low, min_free_len, bridge_gap, step,
                  min_diverge_len=0.0):
    """Remove points in pts that are within lateral distance of any reference line (refs) and return the list of remaining segments."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return []
    if not refs:
        return [pts]
    pts = resample_polyline(pts, step)
    dmin = np.full(len(pts), np.inf)
    for r in refs:
        dmin = np.minimum(dmin, point_to_polyline_dist(pts, r))
    free = hysteresis_free(dmin, pts, overlap_high, overlap_low, min_diverge_len)
    free = bridge_runs(pts, free, bridge_gap)
    pieces = []
    for s, e in true_runs(free):
        run = pts[s:e]
        if len(run) >= 2 and arc_length(run) >= min_free_len:
            pieces.append(run)
    return pieces


def trim_overlaps(polys, *, overlap_high, overlap_low, min_free_len, bridge_gap, step,
                  min_diverge_len=0.0):
    """Trim a list of same-class polylines in descending order of length.
    Return each resulting piece as (original index, point array) so the caller can map metadata."""
    out = []
    kept = []
    for i in sorted(range(len(polys)), key=lambda k: arc_length(polys[k]), reverse=True):
        for piece in subtract_lane(polys[i], kept, overlap_high=overlap_high,
                                   overlap_low=overlap_low, min_free_len=min_free_len,
                                   bridge_gap=bridge_gap, step=step,
                                   min_diverge_len=min_diverge_len):
            kept.append(piece)
            out.append((i, piece))
    return out


def concat_polylines_in_series(polys):
    """Chain end-to-end connecting polylines at the polyline level (not point level) into a single line."""
    polys = [np.asarray(p, dtype=np.float64) for p in polys if len(p) >= 1]
    if len(polys) == 1:
        return polys[0]
    if not polys:
        return np.empty((0, 2), dtype=np.float64)
    centroid = np.vstack(polys).mean(axis=0)
    remaining = list(range(len(polys)))
    best = max(((i, end) for i in remaining for end in (0, -1)),
               key=lambda ie: np.linalg.norm(polys[ie[0]][ie[1]] - centroid))
    start_i, start_end = best
    chain = polys[start_i] if start_end == 0 else polys[start_i][::-1]
    remaining.remove(start_i)
    while remaining:
        tail = chain[-1]
        cand = min(((i, end) for i in remaining for end in (0, -1)),
                   key=lambda ie: np.linalg.norm(polys[ie[0]][ie[1]] - tail))
        i, end = cand
        seg = polys[i] if end == 0 else polys[i][::-1]
        chain = np.vstack([chain, seg])
        remaining.remove(i)
    return chain


@dataclass
class Strand:
    id: int
    peak: Tuple[int, int]
    class_id: int
    points: np.ndarray = None  # sampled points on the original line (N,2)
    ext_points: np.ndarray = None  # points on the line extended on both sides ((N+M),2)
    src_range: Tuple[int, int] = None  # index range that the original points occupy within ext_points
    length: float = 0  # length of the line (cumulative Euclidean distance)


class LaneStitcher:
    id_offset = 10  # minimum offset of the peak ID
    overlap_thresh = 2  # number of overlapping pixels
    short_length = 30
    num_merges = 2  # number of merge iterations (2 is enough)

    # --- clean_lines(trim) / merge_lines parameters ---
    trim_class_id = 1        # class to apply overlap trim to (center_line)
    trim_step = 3.0          # resample interval for trim targets
    overlap_dist = 6.0       # divergence high threshold (px)
    overlap_low = 3.0        # divergence low threshold (px, hysteresis)
    min_diverge_len = 15.0   # accept divergence only when strong (>overlap_dist) continues for at least this arc length (validation-optimal value)
    min_free_len = 0.0       # minimum piece length to keep after trim (px). 0 = preserve short stepping-stone lines (connectivity up, center_line AP20 +1%p)
    bridge_gap = 10.0        # bridging of short overlap breaks in trim (px)
    parallel_overlap = 0.5   # longitudinal overlap threshold of parallel bodies
    parallel_lateral = 30.0  # maximum lateral gap of parallel bodies (px)
    turn_penalty = 3.0       # curvature penalty for next-point selection during sampling. Prefers the straighter continuation at branches (0 = distance only)
    dir_lookback_px = 30     # stabilize the next-point direction by taking the reference from a point ~this far back (using only the previous point causes wobble)
    min_lane_len = 10        # remove lines shorter than this after merging (once connection is done). 10px is optimal (removes only noise, AP +0.5%p)
    smooth_window = 5        # moving-average window size for point smoothing after merge (points)
    smooth_iters = 1         # number of smoothing iterations
    residual_pass = True     # extract once more from the seg region left after the first pass (recover the opposite side of a double line)
    residual_remove_width = 7 # thickness for erasing the first-pass line trace during residual extraction (px)

    def __init__(self, data_path: str, pred_path: str, result_path: str, thickness: int = 3, sample_stride: int = 10, extend_len: int = 20, visualize: bool = True, do_clean: bool = True, split: str = 'validation'):
        self.thickness = thickness
        self.sample_stride = sample_stride
        self.extend_len = extend_len
        self.do_clean = do_clean  # if False, skip clean_lines(trim) and run only the 3 stages
        self._data_path = data_path
        self._pred_path = pred_path
        self._result_path = result_path
        self._split = split  # split to process. Determines the input image list and prediction folder (pred_val/pred_test)
        self._visualize = visualize  # if False, skip window display and visualization collage (fast mode for evaluation)
        self._img_shape = (100, 100)
        self._palette = [info['color'][::-1] for info in cfg.METAINFO]
        # print('palette:', self._palette)
        self._id_count = 0
        self._imshow_base = ImageShow('base images', columns=3, scale=0.8, enabled=visualize)
        self._imshow_proc = ImageShow('processing images', columns=3, scale=0.8, enabled=visualize)
        self._imshow_save = ImageShow('save images', columns=3, scale=0.5, enabled=visualize)
        self._exclude_classes = [0]
        # self.figure_path = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector(copy)/ade20k/result/Figure'
        assert os.path.exists(self._data_path), f"data_path: {self._data_path} is not exists"
        print("make result path: ", self._result_path)
        os.makedirs(self._result_path, exist_ok=True)

    def _split_image_files(self):
        """Sorted satellite image paths for the current split (from the built ade20k dataset)."""
        return sorted(glob.glob(os.path.join(cfg.image_dir(self._split), '*.png')))

    def detect_lines(self, image_ids=None, desc=None):
        file_list = self._split_image_files()
        if image_ids is not None:  # process only a specific subset of images (for experiments/comparison)
            keep = set(image_ids)
            file_list = [f for f in file_list if os.path.basename(f)[:-4] in keep]
        result_jsons = [[] for _ in range(self.num_merges + 1)]  # index 0=origin, 1~3=merge1~3

        pbar = tqdm(enumerate(file_list), total=len(file_list), desc=desc or 'frames', dynamic_ncols=True)
        for i, file_name in pbar:
            # if i > 10:
            #     break
            pbar.set_postfix_str(os.path.basename(file_name))
            image, pred_img, anno_img = self._read_image(file_name)
            self._img_shape = image.shape[:2]
            self._id_count = self.id_offset
            image_id = os.path.basename(file_name).replace('.png', '')

            # Stage 1: first extraction
            first, line_img = self.extract_lines(pred_img, file_name)
            # Stage 2: residual extraction -- extract once more from the seg region with the first-pass line trace erased
            #          (recover missing lines such as the opposite side of a double center line)
            if self.residual_pass:
                res, _ = self.extract_lines(self._residual_pred(pred_img, first), file_name)
            else:
                res = []
            # Stage 3: clean_lines -- clean up the first pass and residual together without distinction.
            #          If do_clean=True, trim center_line parallel overlaps; if False, only re-id.
            combined = first + res
            lines = self._clean_lines(combined) if self.do_clean else self._reindex_lines(combined)
            result_jsons[0] += self.convert_to_json(self._smoothed_copies(lines), image_id)
            images_to_save = {'src_img': image, 'anno_img': anno_img, 'pred_img': pred_img, 'origin': line_img}

            # Stage 4: merge_lines -- serially connect (stitch) end-to-end overlapping lines. Repeat 3 times.
            for n in range(1, self.num_merges + 1):
                lines, line_img = self.merge_lines(lines, n - 1)
                # apply short-line removal only to the output (the next merge input uses unfiltered lines -> preserves connection opportunities)
                # + point smoothing (the merge itself proceeds with the original points)
                output = self._filter_short(lines)
                result_jsons[n] += self.convert_to_json(self._smoothed_copies(output), image_id)
                images_to_save[f'merge{n}'] = line_img

            if self._visualize:
                self._imshow_save.show_imgs(images_to_save, wait_ms=1)
            # self.save_images(self._imshow_save.update_whole_image(), file_name)
            # self._imshow_proc.display(1)
            counts = ', '.join(f'merge{n}={len(result_jsons[n])}' for n in range(self.num_merges + 1))
            pbar.write(f'[{i+1}/{len(file_list)}] {os.path.basename(file_name)} instance counts: {counts}')

        self._save_result_jsons(result_jsons)

    def get_linestrings_for_image(self, file_name: str):
        """Extract linestrings and image shape for a single validation image file."""
        image, pred_img, anno_img = self._read_image(file_name)
        self._img_shape = image.shape[:2]
        self._id_count = self.id_offset
        lines, _ = self.extract_lines(pred_img, file_name)
        return lines, self._img_shape

    def _filter_short(self, lines: List[Strand]) -> List[Strand]:
        """After merging is complete, remove lines with arc length below min_lane_len (0 = pass through)."""
        if self.min_lane_len <= 0:
            return lines
        return [l for l in lines if l.points is not None and len(l.points) >= 2
                and arc_length(l.points) >= self.min_lane_len]

    def _snapshot(self, lines: List[Strand]) -> List[Strand]:
        """A copy with the point arrays duplicated so it is not contaminated by in-place modifications in later stages (merge chaining, etc.)."""
        out = []
        for l in lines:
            if l.points is None:
                continue
            out.append(Strand(
                id=l.id, peak=l.peak, class_id=l.class_id,
                points=np.asarray(l.points).copy(),
                ext_points=None if l.ext_points is None else np.asarray(l.ext_points).copy(),
                src_range=l.src_range, length=l.length))
        return out

    def stage_linestrings(self, file_name: str, do_merge: bool = True, merge_iters: int = None):
        """For figure generation: return per-stage linestring copies of one image.

        Returned dict keys:
          image, pred_img, img_shape,
          first (first extraction), res (residual extraction), combined (first+residual, before refinement),
          refined (after clean_lines), merges (list: merge1..mergeN)
        Each stage copies its points via _snapshot, so they are independent of one another."""
        image, pred_img, _ = self._read_image(file_name)
        self._img_shape = image.shape[:2]
        self._id_count = self.id_offset

        first, _ = self.extract_lines(pred_img, file_name)
        first_snap = self._snapshot(first)
        if self.residual_pass:
            res, _ = self.extract_lines(self._residual_pred(pred_img, first), file_name)
        else:
            res = []
        res_snap = self._snapshot(res)

        combined = first + res
        combined_snap = self._snapshot(combined)
        refined = self._clean_lines(combined) if self.do_clean else self._reindex_lines(combined)
        refined_snap = self._snapshot(refined)

        merges = []
        if do_merge:
            lines = refined
            n_iter = self.num_merges if merge_iters is None else merge_iters
            for k in range(n_iter):
                lines, _ = self.merge_lines(lines, k)
                merges.append(self._snapshot(lines))

        return {
            'image': image, 'pred_img': pred_img, 'img_shape': self._img_shape,
            'first': first_snap, 'res': res_snap, 'combined': combined_snap,
            'refined': refined_snap, 'merges': merges,
        }

    def class_skeleton(self, pred_img: np.ndarray, class_id: int):
        """Return one class's segmentation blobs, the Zhang-Suen skeleton (label map), and per-blob strands.
        self._img_shape must be set before calling (after _read_image or stage_linestrings)."""
        color = self._palette[class_id]
        pred_class_map = np.all(pred_img == color, axis=-1).astype(np.uint8)
        self._id_count = self.id_offset
        line_map, line_strings = self._thin_image(pred_class_map, class_id)
        return pred_class_map, line_map, line_strings

    def _read_image(self, img_file: str):
        image = cv2.imread(img_file)
        # prediction mask: <model>/pred_val|pred_test/<basename>.png (per-split folder)
        base = os.path.basename(img_file)
        pred_file = os.path.join(cfg.pred_path(self._pred_path, self._split), base)
        pred_img = cv2.imread(pred_file)
        anno_img = None
        if self._visualize:
            # color GT overlay (visualization only). May be absent for the test split, so None is allowed.
            anno_file = os.path.join(cfg.color_label_dir(self._split), base)
            anno_img = cv2.imread(anno_file)
        # images = {'image': image, 'GT_img': anno_img, 'pred_img': pred_img}
        # self._imshow_base.show_imgs(images)
        return image, pred_img, anno_img
    
    def extract_lines(self, pred_img: np.ndarray, file_name=None) -> Tuple[List[Strand], np.ndarray]:
        line_string_list = []

        debug_img = np.full_like(pred_img, 255)
        for class_id, color in enumerate(self._palette):
            if class_id in self._exclude_classes:
                continue
            # for class_id in [1, 2, 4, 5, 7, 8, 9]:
            pred_class_map = np.all(pred_img == color, axis=-1).astype(np.uint8)
            line_map, line_strings = self._thin_image(pred_class_map, class_id)
            ext_lines = self._extend_lines(line_map, line_strings)
            line_string_list.extend(ext_lines)
            file_name = os.path.basename(file_name)

        line_img = None
        if self._visualize:
            line_img = np.zeros_like(pred_img)
            line_img = self._draw_colored_lines(line_img, line_string_list)
        # self._imshow_proc.show(line_img, 'extracted lines')
        return line_string_list, line_img
    
    def merge_lines(self, src_line_strings: List[Strand], iter: int) -> Tuple[List[Strand], np.ndarray]:
        dst_line_strings = []
        # print(f'=========== [merge_lines] iter={iter}, src_line_strings: {len(src_line_strings)}')
        for class_id, color in enumerate(self._palette):
            if class_id in self._exclude_classes:
                continue
            class_line_strings = [line for line in src_line_strings if line.class_id == class_id]
            merged_lines = self._merge_lines_by_class(class_line_strings, iter_count=1)
            # print(f'[merge_lines] class_id={class_id}, src lines={len(class_line_strings)}, merged={len(merged_lines)}')
            dst_line_strings.extend(merged_lines)

        line_img = None
        if self._visualize:
            line_img = np.zeros([self._img_shape[0], self._img_shape[1], 3], dtype=np.uint8)
            line_img = self._draw_colored_lines(line_img, dst_line_strings)
        # self._imshow_proc.show(line_img, f'merged_lines_{iter}')
        return dst_line_strings, line_img

    def _thin_image(self, seg_map: np.ndarray, class_id: int):
        # print(f'----- [thin_image] -----')
        line_strings = []
        line_map = np.zeros_like(seg_map, dtype=np.int32)
        line_blobs = np.zeros_like(seg_map, dtype=np.int32)
        y, x = np.nonzero(seg_map)
        fill_value = self.id_offset

        show_blobs = line_blobs.copy()

        for k, (y, x) in enumerate(zip(y, x)):
            if line_blobs[y, x] > 0:
                continue

            # fill the blob containing the seed via floodFill
            temp = seg_map.copy()
            mask = np.zeros((seg_map.shape[0] + 2, seg_map.shape[1] + 2), np.uint8)
            cv2.floodFill(temp, mask, (x, y), fill_value)
            # convert the filled region into a binary mask (0 or 255)
            line_blobs[temp == fill_value] = fill_value
            blob_mask = (temp == fill_value).astype(np.uint8) * 255

            show_blobs = line_blobs.astype(np.int16)

            # apply cv2.ximgproc.thinning (extract thin lines)
            # (cv2.ximgproc.thinning requires the input to be a binary image)
            line_img = cv2.ximgproc.thinning(blob_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

            # accumulate the result into line_map (overwrite overlapping regions)
            line_map[line_img > 0] = fill_value
            line_strings.append(Strand(id=fill_value, class_id=class_id, peak=(x, y)))
            fill_value += 1

        return line_map.astype(np.uint8), line_strings

    def _extend_lines(self, line_map: np.ndarray, line_strings: List[Strand]) -> List[Strand]:
        # print(f'----- [extend_lines] -----')
        id_list = np.unique(line_map)
        id_list = id_list[id_list >= self.id_offset]
        for line_string in line_strings:
            # binary image extracting only that label
            line_img = (line_map == line_string.id).astype(np.uint8)
            line_string.points = self._sample_points(line_img, self.sample_stride)
            if line_string.points.shape[0] < 2:
                line_string.id = None
                continue
            line_string.length = np.sum(np.linalg.norm(np.diff(line_string.points, axis=0), axis=1))
            if line_string.length < 3:
                line_string.id = None
                continue
            line_string = self._extrapolate_line(line_string, self.extend_len, self.sample_stride)

        line_strings = [ls for ls in line_strings if ls.id is not None]
        # sort in descending order by line length
        line_strings.sort(key=lambda ls: ls.length, reverse=True)
        return line_strings

    def _sample_points(self, line_img: np.ndarray, stride: int) -> np.ndarray:
        rows, cols = np.nonzero(line_img)
        points = np.stack((cols, rows), axis=1)
        if points.shape[0] < 2:
            return points
        sorted_points = [points[0]]
        direction = points[1] - points[0]
        sorted_points = self._sort_to_direction(points, sorted_points, True, direction, stride)
        sorted_points = self._sort_to_direction(points, sorted_points, False, -direction, stride)
        sorted_points = np.array(sorted_points).astype(np.int32)
        return sorted_points.astype(np.int32)

    def _sort_to_direction(self, src_points: np.ndarray, sorted_points: List[np.ndarray], to_tail: bool,
                           direction: np.ndarray, stride: int) -> List[np.ndarray]:
        points = src_points.copy()
        while len(points) > 0:
            last_point = sorted_points[-1] if to_tail else sorted_points[0]
            vecs = points - last_point
            distances = np.sqrt(np.sum(vecs ** 2, axis=1))
            dir_norm = float(np.linalg.norm(direction))
            # cosine of the angle between the travel direction and the candidate step
            with np.errstate(invalid='ignore', divide='ignore'):
                cos_ang = np.sum(vecs * direction, axis=1) / (distances * dir_norm + 1e-9)
            # valid candidates: distance [stride,30) & forward (90° cone). Does not cut the line with a hard gate.
            valid_mask = (distances < 30) & (distances >= stride) & (cos_ang >= 0)
            if np.sum(valid_mask) == 0:
                break
            # select by distance + curvature penalty -> prefer the candidate that continues straight at a branch (straight lines pass through as is)
            score = distances * (1.0 + self.turn_penalty * (1.0 - cos_ang))
            score[~valid_mask] = np.inf
            next_index = np.argmin(score)
            if to_tail:
                sorted_points.append(points[next_index])
            else:
                sorted_points.insert(0, points[next_index])
            # direction reference: not the immediately previous point but a point ~dir_lookback_px back -> tip (stable direction)
            direction = self._lookback_direction(sorted_points, to_tail, stride)
            distances = np.sqrt(np.sum((points - last_point) ** 2, axis=1))
            points = points[distances >= stride]
        return sorted_points

    def _lookback_direction(self, sorted_points: List[np.ndarray], to_tail: bool,
                            stride: int) -> np.ndarray:
        """Direction vector connecting the current tip and a point ~dir_lookback_px back (as far as possible if there are too few points)."""
        n = len(sorted_points)
        back = max(1, round(self.dir_lookback_px / max(stride, 1)))
        if to_tail:
            tip, ref = sorted_points[-1], sorted_points[max(0, n - 1 - back)]
        else:
            tip, ref = sorted_points[0], sorted_points[min(n - 1, back)]
        return np.asarray(tip) - np.asarray(ref)

    def _extrapolate_line(self, line_string: Strand, extend_len: int, stride: int) -> Strand:
        points = line_string.points  # (N,2) array
        N = len(points)
        n_ext = extend_len // stride

        head_prev_idx = min(max(N // 3, 3), N - 1)
        head_ext = self._extend_endpoint(points[0], points[head_prev_idx], n_ext, stride)
        tail_prev_idx = N - 1 - min(max(N // 3, 3), N - 1)
        tail_ext = self._extend_endpoint(points[-1], points[tail_prev_idx], n_ext, stride)

        n_head = len(head_ext)
        ext_points = np.vstack([head_ext[::-1], points, tail_ext]) if n_head > 0 else np.vstack([points, tail_ext])
        ext_points = np.clip(ext_points, [0, 0], [self._img_shape[1], self._img_shape[0]])

        line_string.ext_points = np.rint(ext_points).astype(np.int32)
        line_string.src_range = (n_head, n_head + N - 1)
        return line_string

    def _extend_endpoint(self, tip: np.ndarray, prev: np.ndarray, n_ext: int, stride: int) -> np.ndarray:
        direction = tip - prev
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.empty((0, 2), dtype=np.float64)
        direction = direction / norm
        return np.array([tip + direction * stride * i for i in range(1, n_ext + 1)])

    def _reindex_lines(self, lines: List[Strand]) -> List[Strand]:
        """Make the line ids unique (avoid label-map collisions) and recompute the extension points.
        Performs only minimal preprocessing so that merging works even when clean is skipped (do_clean=False)."""
        out = []
        for i, l in enumerate(lines):
            if l.points is None or len(l.points) < 2:
                continue
            l.id = self.id_offset + i
            out.append(self._extrapolate_line(l, self.extend_len, self.sample_stride))
        return out

    def _clean_lines(self, lines: List[Strand]) -> List[Strand]:
        """Clean up the combined first+residual lines (Stage 3, center_line parallel overlap trim).

        For center_line parallel overlaps, keep the longest reference line and cut only the overlapping
        segments of shorter lines (trim_overlaps, hysteresis) -> diverging branches are preserved as
        separate lines. Since the first pass and residual are processed together without distinction,
        cross-pass overlaps are cleaned by the same criterion. Points are not averaged/reordered, so no
        zigzag is produced.

        (Dedup for duplicate removal is removed here because it has zero effect in this pipeline: after
        erasing the residual thickly, duplicate lines at nearly the same position do not appear. Only trim
        has a real effect on center_line.)"""
        targets = [l for l in lines
                   if l.class_id == self.trim_class_id and l.points is not None and len(l.points) >= 2]
        others = [l for l in lines
                  if l.class_id != self.trim_class_id and l.points is not None and len(l.points) >= 2]

        trimmed = []
        for src_i, pts in trim_overlaps(
                [t.points for t in targets],
                overlap_high=self.overlap_dist, overlap_low=self.overlap_low,
                min_free_len=self.min_free_len, bridge_gap=self.bridge_gap, step=self.trim_step,
                min_diverge_len=self.min_diverge_len):
            base = targets[src_i]
            trimmed.append(Strand(id=base.id, peak=base.peak, class_id=self.trim_class_id,
                                  points=np.rint(pts).astype(np.int32),
                                  length=arc_length(pts)))

        return self._reindex_lines(others + trimmed)

    def _residual_pred(self, pred_img: np.ndarray, lines: List[Strand]) -> np.ndarray:
        """Residual prediction image with the first-pass lines' trace (thickness residual_remove_width) erased from the seg map.
        Erased pixels are set to background (0) so the same line is not picked up again during residual extraction."""
        mask = np.zeros((self._img_shape[0], self._img_shape[1]), dtype=np.uint8)
        for l in lines:
            if l.points is None or len(l.points) < 2:
                continue
            pts = l.points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=self.residual_remove_width)
        residual = pred_img.copy()
        residual[mask > 0] = 0
        return residual

    def _smoothed_copies(self, lines: List[Strand]) -> List[Strand]:
        """Make a copy with points smoothed by a moving average (the original lines are preserved for the next merge).
        Relaxes the zigzag that arises at thinning/merge boundaries to straighten the lines."""
        out = []
        for l in lines:
            if l.points is None or len(l.points) < 3:
                out.append(l)
                continue
            sp = smooth_polyline(l.points, self.smooth_window, self.smooth_iters)
            out.append(Strand(id=l.id, peak=l.peak, class_id=l.class_id,
                                  points=np.rint(sp).astype(np.int32), length=l.length))
        return out

    def _merge_lines_by_class(self, line_strings: List[Strand], iter_count: int = 0) -> List[Strand]:
        """Merge end-to-end connecting lines by series concatenation (no point-level NN reordering).

        Candidate pairs are found by extended end-segment overlap (_find_overlap) as before, but parallel
        body pairs are rejected (bodies_parallel) to prevent zigzag caused by mis-merging parallel strands.
        Each group is joined by polyline-level chaining (concat_polylines_in_series), preserving each line's
        internal point order."""
        lines = [l for l in line_strings
                 if l.id is not None and l.points is not None and len(l.points) >= 2]
        n = len(lines)
        if n == 0:
            return []
        id2idx = {l.id: i for i, l in enumerate(lines)}
        parent = list(range(n))
        members = {i: [i] for i in range(n)}
        find = make_find(parent)
        par_cache = {}

        def parallel(i, j):
            key = (i, j) if i < j else (j, i)
            if key not in par_cache:
                par_cache[key] = bodies_parallel(
                    lines[i].points, lines[j].points, self.parallel_overlap, self.parallel_lateral)
            return par_cache[key]

        # candidate pairs: same-class pairs whose extended end-segments overlap
        candidates = set()
        for i, line in enumerate(lines):
            for oid in self._find_overlap(lines, line):
                j = id2idx.get(int(oid))
                if j is not None and j != i:
                    candidates.add((i, j) if i < j else (j, i))

        # group-aware union: reject merging of parallel strands (prevents transitive parallel mis-merge)
        for i, j in candidates:
            if parallel(i, j):
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            mi, mj = members[ri], members[rj]
            if any(parallel(p, q) for p in mi for q in mj):
                continue
            parent[rj] = ri
            members[ri] = mi + mj
            members.pop(rj, None)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        out = []
        for idxs in groups.values():
            if len(idxs) == 1:
                out.append(lines[idxs[0]])
                continue
            base = lines[idxs[0]]
            chained = concat_polylines_in_series([lines[k].points for k in idxs])
            base.points = np.rint(chained).astype(np.int32)
            base.length = arc_length(base.points)
            out.append(self._extrapolate_line(base, self.extend_len, self.sample_stride))
        return out

    def _find_overlap(self, line_strings: List[Strand], this_line: Strand) -> Set[int]:
        src_line_id = this_line.id
        if this_line.ext_points is None:
            return set()
        line_strings_copy = line_strings.copy()
        line_strings_copy.remove(this_line)
        dilated_map = self._draw_line_strings(line_strings_copy, extend=True)
        dilated_map = cv2.dilate(dilated_map, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8))
        label_map = dilated_map.copy() if dilated_map.ndim == 2 else dilated_map[:, :, 0].copy()

        line_img = np.zeros_like(dilated_map, dtype=np.uint8)
        pts = this_line.ext_points.reshape((-1, 1, 2))
        cv2.polylines(line_img, [pts], isClosed=False, color=(255, 255, 255), thickness=3)
        line_map = line_img[:, :, 0]

        label_map[line_map == 0] = 0
        overlap_labels = label_map[label_map > 0]
        if overlap_labels.size == 0:
            return set()
        unique, counts = np.unique(overlap_labels, return_counts=True)

        mask_cond = (unique != 0) & (unique != src_line_id) & (counts > self.overlap_thresh)
        label_ids = set(unique[mask_cond].tolist())
        # print(f'unique: {unique}, counts: {counts}, label_ids: {label_ids}')
        return label_ids

    def _save_result_jsons(self, result_jsons: list):
        # put the split label in the filename to distinguish val/test within the same param folder
        # (coco_pred_val_origin.json, coco_pred_test_merge1.json, etc.).
        label = cfg.split_label(self._split)
        names = ['origin'] + [f'merge{n}' for n in range(1, self.num_merges + 1)]
        for name, data in zip(names, result_jsons):
            path = os.path.join(self._result_path, f'coco_pred_{label}_{name}.json')
            with open(path, 'w') as f:
                json.dump(data, f)
        counts = ', '.join(f'{name}={len(data)}' for name, data in zip(names, result_jsons))
        print(f'FINAL instance counts: {counts}\n')

    def _exclude_short_lines(self, line_strings: List[Strand]) -> Tuple[List[Strand], np.ndarray]:
        filtered_line_strings = []

        # [debug] print the number of input lines
        print(f'>> [DEBUG] _exclude_short_lines input count: {len(line_strings)}')

        rejected_count = 0
        min_len = float('inf')
        max_len = 0

        for line in line_strings:
            # existing logic: recompute length
            line.length = np.sum(np.linalg.norm(np.diff(line.points, axis=0), axis=1))

            # [debug] track min/max length
            if line.length < min_len: min_len = line.length
            if line.length > max_len: max_len = line.length

            if line.length > self.short_length:
                filtered_line_strings.append(line)
            else:
                # [debug] print info of the rejected line
                rejected_count += 1
                # print(f'   >> Rejected Line ID: {line.id}, Length: {line.length:.2f}')

        # [debug] print result summary
        print(f'>> [DEBUG] Threshold: {self.short_length}')
        print(f'>> [DEBUG] Length Range: Min={min_len:.2f}, Max={max_len:.2f}')
        print(f'>> [DEBUG] Rejected: {rejected_count}, Survived: {len(filtered_line_strings)}')

        line_img = np.zeros([self._img_shape[0], self._img_shape[1], 3], dtype=np.uint8)
        line_img = self._draw_colored_lines(line_img, filtered_line_strings)
        return filtered_line_strings, line_img

    def _draw_line_strings(self, line_strings: List[Strand], extend=False, color=None):
        image = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
        for line in line_strings:
            if line.id is None or line.points is None:
                continue
            if extend:
                pts = line.ext_points.reshape((-1, 1, 2))
            else:
                pts = line.points.reshape((-1, 1, 2))
            line_color = (line.id, line.id, line.id) if color is None else color
            cv2.polylines(image, [pts], isClosed=False, color=line_color, thickness=3)
        return image

    def _draw_single_line(self, line_string : Strand, thickness=None, extend=False):
        image = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
        if line_string.id is None or line_string.points is None:
            return image
        pts = line_string.points.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=(line_string.id, line_string.id, line_string.id), thickness=thickness)
        return image

    def _draw_colored_lines(self, pred_img, line_strings : List[Strand], extended=False, select=None):
        image = pred_img.copy()
        for line in line_strings:
            if line.id is None:
                continue
            color = self._palette[line.class_id]
            if extended:
                pts = line.ext_points.reshape((-1, 1, 2))
            else:
                pts = line.points.reshape((-1, 1, 2))
            if select is None:
                cv2.polylines(image, [pts], isClosed=False, color=color, thickness=3)
            elif select is not None:
                if line.class_id == select:
                    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=3)
                else:
                    continue
        return image

    def _draw_blobs_with_color(self, line_map):
        n_labels = int(np.max(line_map))
        rng = np.random.default_rng(42)  # fix the seed for reproducibility (remove if desired)
        H = rng.uniform(0, 180, size=n_labels + 1)  # [0,180)
        S = rng.uniform(170, 255, size=n_labels + 1)  # saturation up (170~255)
        V = rng.uniform(130, 220, size=n_labels + 1)  # value down (130~220)
        H[0], S[0], V[0] = 0, 0, 0
        hsv = np.stack([H, S, V], axis=1).astype(np.uint8).reshape(-1, 1, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)  # shape: (n_labels+1, 3)
        colorized = bgr[line_map]
        return colorized

    def convert_to_json(self, line_strings: List[Strand], image_id: str):
        pred_json = []
        for line in line_strings:
            mask = self._draw_single_line(line, self.thickness)
            mask = (np.all(mask > 0, axis=-1)).astype(np.uint8)
            mask = np.asfortranarray(mask)
            rle = maskUtils.encode(mask)
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            prediction = {
                "image_id": image_id,
                "category_id": line.class_id,
                "segmentation": rle,
                "score": 1.,
            }
            pred_json.append(prediction)
        return pred_json

    def save_images(self, save_image, img_file):
        filename = img_file.replace('/images/validation', '/result/results')
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        print('save filename:', filename)
        cv2.imwrite(filename, save_image)


def main():
    """Smoke test for LaneStitcher -- this is NOT the real pipeline entry point.

    The real lane-stitching run is done via experiment/run_experiments.py (full
    parameter sweep) or experiment/run_best_experiment.py (best config only). This
    test runs detection on a few validation images with the best config and writes
    the predictions to a throwaway subfolder, to confirm the class works end to end.
    """
    from stitch_config import load_stitch_config
    sc = load_stitch_config()
    out_dir = os.path.join(cfg.RESULT_PATH, "_smoketest", "lane_stitcher")
    detector = LaneStitcher(cfg.DATASET_PATH, sc.model_path, out_dir,
                            thickness=sc.thickness, sample_stride=sc.sample_stride,
                            extend_len=sc.extend_len, visualize=False, split='validation')
    detector.turn_penalty = sc.turn_penalty
    files = detector._split_image_files()[:3]
    image_ids = [os.path.basename(f)[:-4] for f in files]
    detector.detect_lines(image_ids=image_ids, desc="lane_stitcher smoke test")


if __name__ == '__main__':
    main()
