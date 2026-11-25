"""Polygon 校验、栅格化与几何统计。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GeometryMetrics:
    bbox_xyxy: tuple[int, int, int, int]
    mask_area_px: int
    bbox_area_px: int
    fill_ratio: float
    aspect_ratio: float
    convexity: float
    vertex_count: int
    is_self_intersecting: bool
    out_of_bounds_points: int


def validate_polygon(points: np.ndarray, image_width: int, image_height: int) -> list[str]:
    issues: list[str] = []
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return ["invalid_vertex_count"]
    if not np.isfinite(points).all():
        issues.append("non_finite_coordinate")
    if ((points[:, 0] < 0) | (points[:, 0] >= image_width) | (points[:, 1] < 0) | (points[:, 1] >= image_height)).any():
        issues.append("out_of_bounds")
    if is_self_intersecting(points):
        issues.append("self_intersection")
    return issues


def rasterize_polygon(points: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if points.ndim != 2 or len(points) < 3:
        return mask
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, image_width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, image_height - 1)
    cv2.fillPoly(mask, [np.rint(clipped).astype(np.int32)], 255)
    return mask


def measure_polygon(points: np.ndarray, mask: np.ndarray, image_width: int, image_height: int) -> GeometryMetrics:
    x_min, y_min = np.floor(points.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(points.max(axis=0)).astype(int)
    x_min = int(np.clip(x_min, 0, image_width - 1))
    y_min = int(np.clip(y_min, 0, image_height - 1))
    x_max = int(np.clip(x_max, 0, image_width - 1))
    y_max = int(np.clip(y_max, 0, image_height - 1))
    bbox_width = max(1, x_max - x_min + 1)
    bbox_height = max(1, y_max - y_min + 1)
    bbox_area = bbox_width * bbox_height
    mask_area = int(np.count_nonzero(mask))
    contour = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(contour)
    contour_area = abs(float(cv2.contourArea(contour)))
    hull_area = abs(float(cv2.contourArea(hull)))
    return GeometryMetrics(
        bbox_xyxy=(x_min, y_min, x_max, y_max),
        mask_area_px=mask_area,
        bbox_area_px=bbox_area,
        fill_ratio=mask_area / bbox_area if bbox_area else 0.0,
        aspect_ratio=max(bbox_width, bbox_height) / min(bbox_width, bbox_height),
        convexity=contour_area / hull_area if hull_area > 0 else 0.0,
        vertex_count=len(points),
        is_self_intersecting=is_self_intersecting(points),
        out_of_bounds_points=int(
            np.count_nonzero(
                (points[:, 0] < 0)
                | (points[:, 0] >= image_width)
                | (points[:, 1] < 0)
                | (points[:, 1] >= image_height)
            )
        ),
    )


def is_self_intersecting(points: np.ndarray) -> bool:
    if len(points) < 4:
        return False
    segments = list(zip(points, np.roll(points, -1, axis=0)))
    count = len(segments)
    for index, (a1, a2) in enumerate(segments):
        for other_index in range(index + 1, count):
            if other_index in {index, (index - 1) % count, (index + 1) % count}:
                continue
            if index == 0 and other_index == count - 1:
                continue
            b1, b2 = segments[other_index]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _segments_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    o1, o2 = orientation(a1, a2, b1), orientation(a1, a2, b2)
    o3, o4 = orientation(b1, b2, a1), orientation(b1, b2, a2)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    return (
        (o1 == 0 and on_segment(a1, b1, a2))
        or (o2 == 0 and on_segment(a1, b2, a2))
        or (o3 == 0 and on_segment(b1, a1, b2))
        or (o4 == 0 and on_segment(b1, a2, b2))
    )
