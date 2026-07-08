"""Reimplementation of the OpenSatMap baseline (watershed) post-processing -- defense for review point M1 (lack of a quantitative comparison against an external baseline).

Applies only watershed instance separation + PCA principal-axis-aligned vectorization on top of the same
segmentation predictions from our best model (Mask2Former Swin-L). The strengths of our pipeline (curvature
tracking, fragment merge, double-line trim, residual) are intentionally excluded so that only the difference
in the pure post-processing algorithm is revealed.

I/O, rasterization (3px), and RLE encoding reuse LaneStitcher via composition to match ours exactly.
Design rationale: baseline_opensatmap_design.md (§2 pseudocode, §6 recommended constants).
"""
import os
import json

import cv2
import numpy as np
from tqdm import tqdm

import config as cfg
from lane_stitcher import LaneStitcher, Strand, resample_polyline


class OpenSatMapBaseline:
    """seg prediction -> (watershed instance separation -> PCA-aligned vectorization) -> COCO prediction JSON.

    Holds a single LaneStitcher instance and calls _split_image_files, _read_image, _palette, and
    convert_to_json directly to keep the input image list, prediction loading, rasterization, and encoding
    identical to our pipeline.
    The baseline uses only the paper constants and is not retuned on validation (fairness)."""

    watershed_alpha = 0.5   # sure_fg threshold = alpha * dist.max() (OpenCV tutorial recipe, cited in paper §4.1)
    min_area = 100          # minimum instance pixel count (<100px removed, paper constant)

    def __init__(self, stitcher: LaneStitcher, sample_stride: int):
        self._stitcher = stitcher          # reuse I/O, rasterization, encoding (composition)
        self._sample_stride = sample_stride  # resample interval (same as our best, §6-3)

    def run(self):
        """Process the entire current split (validation) and return the list of baseline predictions."""
        files = self._stitcher._split_image_files()
        preds = []
        for file_name in tqdm(files, desc="OpenSatMap baseline"):
            image, pred_img, _ = self._stitcher._read_image(file_name)
            self._stitcher._img_shape = image.shape[:2]  # raster canvas size for convert_to_json
            image_id = os.path.basename(file_name)[:-4]
            strands = self._extract(pred_img)
            preds += self._stitcher.convert_to_json(strands, image_id)
        return preds

    def run_and_save(self, save_path: str):
        """Save the run() result as JSON and return the list of predictions."""
        preds = self.run()
        with open(save_path, "w") as fp:
            json.dump(preds, fp)
        print(f"saved: {save_path} (instances={len(preds)})")
        return preds

    def _extract(self, pred_img: np.ndarray):
        """Build a list of baseline strands per each of the 9 evaluation classes from the prediction image."""
        strands = []
        next_id = LaneStitcher.id_offset
        for class_id in cfg.EVAL_CLASS_IDS:
            color = self._stitcher._palette[class_id]  # extract class color the same way we do
            mask = np.all(pred_img == color, axis=-1).astype(np.uint8)
            for poly in self._vectorize_class(mask):
                strands.append(Strand(id=next_id, peak=(0, 0), class_id=class_id, points=poly))
                next_id += 1
        return strands

    def _vectorize_class(self, mask: np.ndarray):
        """One class binary mask -> list of PCA-aligned, resampled polylines per watershed instance."""
        labels = self._watershed_labels(mask)
        polys = []
        for label in np.unique(labels):
            if label <= 1:  # 1=background, -1=boundary
                continue
            instance = labels == label
            if int(instance.sum()) < self.min_area:  # paper constant: remove <100px
                continue
            poly = self._order_and_resample(instance)
            if poly is not None:
                polys.append(poly)
        return polys

    def _watershed_labels(self, mask: np.ndarray) -> np.ndarray:
        """Build an instance label map using the OpenCV tutorial recipe (cited in the paper).

        opening noise removal -> sure_bg (dilate) / sure_fg (distanceTransform>alpha*max) ->
        unknown region marker 0 -> watershed. For thin lines, sure_fg keeps only the central ridge,
        so each connected blob is separated into one instance (touching blobs such as double lines remain
        a single instance = the baseline limitation as is)."""
        binary = (mask > 0).astype(np.uint8) * 255
        if binary.max() == 0:
            return np.ones(binary.shape, dtype=np.int32)  # background only
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        sure_bg = cv2.dilate(opening, kernel)
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, self.watershed_alpha * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown > 0] = 0
        return cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)

    def _order_and_resample(self, instance_mask: np.ndarray):
        """Order the instance pixels by PCA principal-axis projection and uniformly resample by sample_stride."""
        rows, cols = np.nonzero(instance_mask)
        if len(cols) < 2:
            return None
        pts = np.stack([cols, rows], axis=1).astype(np.float64)  # (P,2) = (x,y), same as our point convention
        axis = self._principal_axis(pts)
        order = np.argsort(pts @ axis)  # order points by principal-axis projection (no curvature handling, keeps the baseline naive)
        poly = resample_polyline(pts[order], self._sample_stride)
        if len(poly) < 2:
            return None
        return np.rint(poly).astype(np.int32)

    @staticmethod
    def _principal_axis(pts: np.ndarray) -> np.ndarray:
        """Unit vector of the first PCA principal component (principal axis)."""
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt[0]
