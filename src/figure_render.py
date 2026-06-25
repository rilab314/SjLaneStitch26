"""Figure 렌더링 유틸: 폴리라인·끝점 점 그리기와 원본 해상도 콜라주.

merge_annotation.py(GT 통합 비교 이미지)와 Figure/*.py가 같은 끝점 마커 스타일을 공유한다.
"""
import sys
import os

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import config as cfg

ENDPOINT_DOT_RADIUS = 4   # 끝점 마커 반지름(merge_annotation 원본 5보다 작게)
WHITE = (255, 255, 255)
GAP = 20                  # 콜라주 패널 사이 간격(px)
SEPARATOR = 0             # 패널 사이 여백 색(검은색) — 흰 배경 패널을 구분


def make_white_canvas(height, width):
    """흰색 배경 캔버스를 만든다."""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def draw_strand(canvas, points, color, thickness=3, dot_radius=ENDPOINT_DOT_RADIUS,
                draw_dots=True):
    """폴리라인을 그리고 양 끝점에 동그란 점(흰 채움 + 색 테두리)을 찍는다."""
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
    """Strand 리스트를 클래스색으로 그린다(끝점 점 포함). canvas를 in-place 수정 후 반환."""
    exclude_ids = cfg.EXCLUDE_IDS if exclude_ids is None else exclude_ids
    color_map = cfg.RENDER_ID2BGR if color_map is None else color_map
    for strand in strands:
        if not _is_drawable(strand, exclude_ids, only_class):
            continue
        color = color_map.get(strand.class_id, cfg.ID2BGR.get(strand.class_id, (0, 0, 0)))
        draw_strand(canvas, strand.points, color, thickness=thickness, draw_dots=dots)
    return canvas


def draw_extension(canvas, ext_points, src_range, color, thickness=3):
    """원본 본체 구간(src_range) 바깥쪽 외삽 부분만 색선으로 그린다(본체는 그대로)."""
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
    """그릴 대상인지 판정(제외 클래스·단일 클래스 필터·점 개수)."""
    if strand.class_id in exclude_ids:
        return False
    if only_class is not None and strand.class_id != only_class:
        return False
    return strand.points is not None and len(strand.points) >= 1


def concat_horizontal(images, gap=GAP):
    """이미지들을 원본 해상도 그대로 가로로 붙이고 사이에 흰색 간격을 둔다."""
    panels = [img for img in images if img is not None]
    if not panels:
        return None
    height = max(img.shape[0] for img in panels)
    panels = [pad_to_height(img, height) for img in panels]
    separator = np.full((height, gap, 3), SEPARATOR, dtype=np.uint8)
    return np.hstack(_interleave(panels, separator))


def concat_vertical(images, gap=GAP):
    """이미지들을 세로로 붙이고 사이에 흰색 간격을 둔다."""
    panels = [img for img in images if img is not None]
    if not panels:
        return None
    width = max(img.shape[1] for img in panels)
    panels = [pad_to_width(img, width) for img in panels]
    separator = np.full((gap, width, 3), SEPARATOR, dtype=np.uint8)
    return np.vstack(_interleave(panels, separator))


def _interleave(panels, separator):
    """패널 사이에 separator를 끼워 넣은 리스트를 만든다."""
    stacked = []
    for index, panel in enumerate(panels):
        if index > 0:
            stacked.append(separator)
        stacked.append(panel)
    return stacked


def pad_to_height(image, target_height):
    """세로 높이를 흰색으로 패딩해 target_height에 맞춘다."""
    height, width = image.shape[:2]
    if height >= target_height:
        return image
    pad = np.full((target_height - height, width, 3), 255, dtype=np.uint8)
    return np.vstack([image, pad])


def pad_to_width(image, target_width):
    """가로 너비를 흰색으로 패딩해 target_width에 맞춘다."""
    height, width = image.shape[:2]
    if width >= target_width:
        return image
    pad = np.full((height, target_width - width, 3), 255, dtype=np.uint8)
    return np.hstack([image, pad])


def recolor_segmentation(pred_img, exclude_ids):
    """분할 예측 이미지를 figure용으로 변환: 배경/제외 클래스는 흰색, 클래스색→렌더색."""
    out = pred_img.copy()
    out[np.all(out == [0, 0, 0], axis=-1)] = WHITE
    _whiten_excluded(out, pred_img, exclude_ids)
    _apply_render_colors(out, pred_img, exclude_ids)
    return out


def _whiten_excluded(out, pred_img, exclude_ids):
    """제외 클래스 픽셀을 흰색으로 칠한다(원색·렌더색 모두)."""
    for class_id in exclude_ids:
        for color_map in (cfg.ID2BGR, cfg.RENDER_ID2BGR):
            color = color_map.get(class_id)
            if color is not None:
                out[np.all(pred_img == color, axis=-1)] = WHITE


def _apply_render_colors(out, pred_img, exclude_ids):
    """평가 클래스의 원색을 렌더색으로 치환한다."""
    for class_id, original in cfg.ID2BGR.items():
        if class_id in exclude_ids or class_id == 0:
            continue
        render = cfg.RENDER_ID2BGR.get(class_id)
        if render is not None and tuple(render) != tuple(original):
            out[np.all(pred_img == original, axis=-1)] = render
