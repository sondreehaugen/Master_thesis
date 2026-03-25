from __future__ import annotations

import cv2
import numpy as np


def centroid_xyxy(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bottom_center_xyxy(box: list[float], y_offset: int = 2) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2 - y_offset)


def tracking_point_xyxy(box: list[float], use_bottom_center: bool = True) -> tuple[float, float]:
    if use_bottom_center:
        return bottom_center_xyxy(box)
    return centroid_xyxy(box)


def point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    cnt = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(cnt, point, False) >= 0


def side_of_directed_line(
    p: tuple[int, int],
    a: tuple[int, int],
    b: tuple[int, int],
) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _ccw(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _intersect(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
    d: tuple[int, int],
) -> bool:
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def gate_cross_direction_hysteresis(
    p0: tuple[int, int],
    p1: tuple[int, int],
    a: tuple[int, int],
    b: tuple[int, int],
    band_px: int = 10,
) -> str | None:
    s0 = side_of_directed_line(p0, a, b)
    s1 = side_of_directed_line(p1, a, b)

    line_len = max(float(np.hypot(b[0] - a[0], b[1] - a[1])), 1e-6)
    d0 = s0 / line_len
    d1 = s1 / line_len

    clear_flip = (d0 > band_px and d1 < -band_px) or (d0 < -band_px and d1 > band_px)
    geometric_cross = _intersect(p0, p1, a, b)

    if not (clear_flip or geometric_cross):
        return None
    if d0 > 0 and d1 < 0:
        return "A->B"
    if d0 < 0 and d1 > 0:
        return "B->A"
    return "cross"
