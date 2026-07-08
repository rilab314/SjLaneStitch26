"""Figure rendering utilities: drawing polylines/endpoint dots and original-resolution collages.

merge_annotation.py (GT integration comparison images) and Figure/*.py share the same endpoint marker style.
"""
import sys
import os

import cv2
import numpy as np
from pycocotools import mask as mask_util

import config as cfg

ENDPOINT_DOT_RADIUS = 4   # endpoint marker radius (smaller than merge_annotation's original 5)
WHITE = (255, 255, 255)
GAP = 20                  # gap between collage panels (px)
SEPARATOR = 0             # gap color between panels (black) — separates white-background panels


def make_white_canvas(height, width):
    """Creates a white-background canvas."""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def draw_strand(canvas, points, color, thickness=3, dot_radius=ENDPOINT_DOT_RADIUS,
                draw_dots=True):
    """Draws a polyline and places round dots (white fill + colored border) at both endpoints."""
    pts = np.asarray(points)
    if len(pts) < 1:
        return canvas
    poly = np.rint(pts).astype(np.int32).reshape((-1, 1, 2))
    if len(poly) >= 2:
        cv2.polylines(canvas, [poly], isClosed=False, color=color, thickness=thickness)
    if draw_dots:
        for tip in (pts[0], pts[-1]):
            center = (int(round(float(tip[0]))), int(round(float(tip[1]))))
            cv2.circle(canvas, center, dot_radius, WHITE, thickness=-1)
            cv2.circle(canvas, center, dot_radius, color, thickness=2)
    return canvas


def draw_strands(canvas, strands, exclude_ids=None, thickness=3, dots=True,
                 color_map=None, only_class=None):
    """Draws a list of Strands in class colors (with endpoint dots). Modifies canvas in place and returns it."""
    exclude_ids = cfg.EXCLUDE_IDS if exclude_ids is None else exclude_ids
    color_map = cfg.RENDER_ID2BGR if color_map is None else color_map
    for strand in strands:
        if not _is_drawable(strand, exclude_ids, only_class):
            continue
        color = color_map.get(strand.class_id, cfg.ID2BGR.get(strand.class_id, (0, 0, 0)))
        draw_strand(canvas, strand.points, color, thickness=thickness, draw_dots=dots)
    return canvas


def draw_extension(canvas, ext_points, src_range, color, thickness=3):
    """Draws only the extrapolated part outside the original body range (src_range) as colored lines (body left as is)."""
    if ext_points is None or src_range is None:
        return canvas
    pts = np.rint(ext_points).astype(np.int32)
    head, tail = src_range
    if head > 0:
        cv2.polylines(canvas, [pts[0:head + 1].reshape(-1, 1, 2)], False, color, thickness)
    if tail < len(pts) - 1:
        cv2.polylines(canvas, [pts[tail:].reshape(-1, 1, 2)], False, color, thickness)
    return canvas


def _is_drawable(strand, exclude_ids, only_class):
    """Decides whether it is a drawable target (excluded-class/single-class filter/point count)."""
    if strand.class_id in exclude_ids:
        return False
    if only_class is not None and strand.class_id != only_class:
        return False
    return strand.points is not None and len(strand.points) >= 1


def concat_horizontal(images, gap=GAP):
    """Concatenates images horizontally at their original resolution with gaps in between."""
    panels = [img for img in images if img is not None]
    if not panels:
        return None
    height = max(img.shape[0] for img in panels)
    panels = [pad_to_height(img, height) for img in panels]
    separator = np.full((height, gap, 3), SEPARATOR, dtype=np.uint8)
    return np.hstack(_interleave(panels, separator))


def concat_vertical(images, gap=GAP):
    """Concatenates images vertically with gaps in between."""
    panels = [img for img in images if img is not None]
    if not panels:
        return None
    width = max(img.shape[1] for img in panels)
    panels = [pad_to_width(img, width) for img in panels]
    separator = np.full((gap, width, 3), SEPARATOR, dtype=np.uint8)
    return np.vstack(_interleave(panels, separator))


def _interleave(panels, separator):
    """Builds a list with separators inserted between panels."""
    stacked = []
    for index, panel in enumerate(panels):
        if index > 0:
            stacked.append(separator)
        stacked.append(panel)
    return stacked


def pad_to_height(image, target_height):
    """Pads the height with white to match target_height."""
    height, width = image.shape[:2]
    if height >= target_height:
        return image
    pad = np.full((target_height - height, width, 3), 255, dtype=np.uint8)
    return np.vstack([image, pad])


def pad_to_width(image, target_width):
    """Pads the width with white to match target_width."""
    height, width = image.shape[:2]
    if width >= target_width:
        return image
    pad = np.full((height, target_width - width, 3), 255, dtype=np.uint8)
    return np.hstack([image, pad])


def recolor_segmentation(pred_img, exclude_ids):
    """Converts the segmentation prediction image for figures: background/excluded classes -> white, class colors -> render colors."""
    out = pred_img.copy()
    out[np.all(out == [0, 0, 0], axis=-1)] = WHITE
    _whiten_excluded(out, pred_img, exclude_ids)
    _apply_render_colors(out, pred_img, exclude_ids)
    return out


def _whiten_excluded(out, pred_img, exclude_ids):
    """Paints excluded-class pixels white (both original and render colors)."""
    for class_id in exclude_ids:
        for color_map in (cfg.ID2BGR, cfg.RENDER_ID2BGR):
            color = color_map.get(class_id)
            if color is not None:
                out[np.all(pred_img == color, axis=-1)] = WHITE


def _apply_render_colors(out, pred_img, exclude_ids):
    """Replaces the original colors of eval classes with render colors."""
    for class_id, original in cfg.ID2BGR.items():
        if class_id in exclude_ids or class_id == 0:
            continue
        render = cfg.RENDER_ID2BGR.get(class_id)
        if render is not None and tuple(render) != tuple(original):
            out[np.all(pred_img == original, axis=-1)] = render


def overlay_segmentation(image, pred_img, exclude_ids, alpha=0.5):
    """Overlays the segmentation prediction on the original image in class render colors (alpha=opacity, excluded classes omitted)."""
    out = image.copy()
    for class_id, original in cfg.ID2BGR.items():
        if class_id == 0 or class_id in exclude_ids:
            continue
        mask = np.all(pred_img == original, axis=-1)
        if not mask.any():
            continue
        render = np.array(cfg.RENDER_ID2BGR.get(class_id, original), dtype=np.float32)
        out[mask] = (alpha * render + (1.0 - alpha) * out[mask]).astype(np.uint8)
    return out


def draw_annotations_on_image(img, annotations, exclude_ids):
    """Fills RLE/polygon annotations onto the image in class render colors (GT/prediction overlay)."""
    for ann in annotations:
        cat_id = ann.get("category_id")
        if cat_id in exclude_ids:
            continue
        seg = ann.get("segmentation")
        if seg is None:
            continue
        color = cfg.RENDER_ID2BGR.get(cat_id, cfg.ID2BGR.get(cat_id, WHITE))
        _fill_segmentation(img, seg, color)
    return img


def _fill_segmentation(img, seg, color):
    """Fills an RLE (dict) or polygon (list) segmentation with color."""
    if isinstance(seg, dict) and "counts" in seg:
        mask = mask_util.decode(seg)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        img[mask > 0] = color
    elif isinstance(seg, list):
        polys = [seg] if seg and isinstance(seg[0], (int, float)) else seg
        for poly in polys:
            if len(poly) >= 6:
                pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(img, [pts], color)
