from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


Point = tuple[int, int]


class ZoneDict(TypedDict):
    polygon: list[Point]
    dir: str


class GateDict(TypedDict):
    pts: list[Point]
    dir: str


@dataclass(slots=True)
class TrackingEvent:
    frame: int
    track_id: int
    origin: str
    dest: str
    cls_name: str
    conf: float
    count_method: str
    gate_cross: str | None = None
    origin_frame: int | None = None
