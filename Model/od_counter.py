from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict

from .src.geometry import point_in_polygon
from .types import TrackingEvent


class ODCounter:
    def __init__(
        self,
        zone_dwell_frames: int = 2,
        gate_band_px: int = 10,
        allow_unknown_origin: bool = False,
    ) -> None:
        self.zone_dwell_frames = max(zone_dwell_frames, 1)
        self.gate_band_px = gate_band_px
        self.allow_unknown_origin = allow_unknown_origin

        self.zone_hits_buffer = defaultdict(lambda: deque(maxlen=self.zone_dwell_frames))
        self.origin_candidate: dict[int, str] = {}
        self.first_dir: dict[int, str] = {}
        self.zone_entry_frame: dict[int, int] = {}

        self.last_point: dict[int, tuple[int, int]] = {}
        self.gate_seen_for_id: set[tuple[int, str]] = set()
        self.counted_trip: set[int] = set()

        self.unknown_origin_ids: set[int] = set()
        self.entered_zone_once = defaultdict(set)
        self.crossed_gate_once = defaultdict(set)

        self.od_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.od_events: list[TrackingEvent] = []

    def _update_origin(
        self,
        track_id: int,
        point: tuple[int, int],
        arm_zones: dict,
        frame_idx: int,
    ) -> None:
        current_zone_dir = None
        for zone_name, zone_info in arm_zones.items():
            poly = zone_info.get("polygon", [])
            if len(poly) < 3:
                continue
            if point_in_polygon(point, poly):
                current_zone_dir = zone_info.get("dir", zone_name)
                self.entered_zone_once[track_id].add(current_zone_dir)
                break

        self.zone_hits_buffer[track_id].append(current_zone_dir)

        if track_id in self.first_dir:
            return
        if len(self.zone_hits_buffer[track_id]) < self.zone_dwell_frames:
            return

        vals = [z for z in self.zone_hits_buffer[track_id] if z is not None]
        if len(vals) >= self.zone_dwell_frames and len(set(vals)) == 1:
            self.origin_candidate[track_id] = vals[-1]
            self.first_dir[track_id] = vals[-1]
            self.zone_entry_frame[track_id] = frame_idx

    def _update_destination(
        self,
        track_id: int,
        prev_point: tuple[int, int],
        curr_point: tuple[int, int],
        gates: dict,
        cls_name: str,
        conf: float,
        frame_idx: int,
    ) -> None:
        for gate_name, gate_info in gates.items():
            pts = gate_info.get("pts", [])
            if len(pts) < 2:
                continue

            a = tuple(pts[0])
            b = tuple(pts[1])
            # Simplified: check if current point is near gate line
            if len(pts) >= 2:
                # Use point_in_polygon as simplified check
                prev_in = point_in_polygon(prev_point, gate_info.get("polygon", []))
                curr_in = point_in_polygon(curr_point, gate_info.get("polygon", []))
                cross = "cross" if (prev_in != curr_in) else None
            else:
                cross = None
            if cross is None:
                continue

            key = (track_id, gate_name)
            if key in self.gate_seen_for_id:
                continue
            self.gate_seen_for_id.add(key)

            current_dir = gate_info.get("dir") or gate_name
            self.crossed_gate_once[track_id].add(current_dir)

            if track_id in self.counted_trip:
                continue

            if track_id in self.first_dir:
                origin = self.first_dir[track_id]
                dest = current_dir
                if origin == dest:
                    continue

                self.od_counts[dest][origin][cls_name] += 1
                self.od_events.append(
                    TrackingEvent(
                        frame=frame_idx,
                        track_id=track_id,
                        origin=origin,
                        dest=dest,
                        cls_name=cls_name,
                        conf=conf,
                        count_method="zone->gate",
                        gate_cross=cross,
                        origin_frame=self.zone_entry_frame.get(track_id),
                    )
                )
                self.counted_trip.add(track_id)

            elif self.allow_unknown_origin:
                self.unknown_origin_ids.add(track_id)
                self.od_counts[current_dir]["UNKNOWN"][cls_name] += 1
                self.od_events.append(
                    TrackingEvent(
                        frame=frame_idx,
                        track_id=track_id,
                        origin="UNKNOWN",
                        dest=current_dir,
                        cls_name=cls_name,
                        conf=conf,
                        count_method="unknown->gate",
                        gate_cross=cross,
                        origin_frame=None,
                    )
                )
                self.counted_trip.add(track_id)

    def update(
        self,
        track_id: int,
        point: tuple[int, int],
        gates: dict,
        arm_zones: dict,
        cls_name: str,
        conf: float,
        frame_idx: int,
    ) -> None:
        self._update_origin(track_id, point, arm_zones, frame_idx)

        if track_id in self.last_point:
            self._update_destination(
                track_id,
                self.last_point[track_id],
                point,
                gates,
                cls_name,
                conf,
                frame_idx,
            )

        self.last_point[track_id] = point

    def jsonable_counts(self) -> dict:
        out: dict[str, dict[str, dict[str, int]]] = {}
        for dest, by_origin in self.od_counts.items():
            out[dest] = {}
            for origin, by_class in by_origin.items():
                out[dest][origin] = dict(by_class)
        return out

    def events_as_dict(self) -> list[dict]:
        return [asdict(ev) for ev in self.od_events]
