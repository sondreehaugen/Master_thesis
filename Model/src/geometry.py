"""Geometry helpers for track-point extraction and polygon tests."""

from __future__ import annotations

import cv2
import numpy as np


def centroid_xyxy(box: list[float]) -> tuple[float, float]:
    """Return the centroid of an XYXY bounding box."""

    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bottom_center_xyxy(box: list[float], y_offset: int = 2) -> tuple[float, float]:
    """Return the bottom-center point of an XYXY bounding box."""

    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2 - y_offset)


def tracking_point_xyxy(box: list[float], use_bottom_center: bool = True) -> tuple[float, float]:
    """Return the tracking point used for zone assignment and trajectory drawing."""

    if use_bottom_center:
        return bottom_center_xyxy(box)
    return centroid_xyxy(box)


def point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    """Test whether a point lies inside or on the boundary of a polygon."""

    cnt = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(cnt, point, False) >= 0