from __future__ import annotations

import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import TrackingConfig
from .geometry import tracking_point_xyxy
from .od_counter import ODCounter


@dataclass(slots=True)
class TrackingArtifacts:
    mp4_path: str
    elapsed_time: float
    unique_ids: int
    track_ids: list[int]
    all_track_ids: list[int]
    filtered_out_ids: list[int]
    min_track_frames: int
    raw_detections: int
    frames_with_dets_no_ids: int
    od_counts: dict
    od_events: list[dict]


class TrackingRunner:
    def __init__(self, model, config: TrackingConfig) -> None:
        self.model = model
        self.config = config

    def _get_track_class(self, tid: int, class_conf_history: dict[int, dict[int, list[float]]]) -> str | None:
        if tid not in class_conf_history:
            return None

        class_stats = {
            cid: {"frames": len(cvals), "mean_conf": float(np.mean(cvals))}
            for cid, cvals in class_conf_history[tid].items()
            if len(cvals) > 0
        }
        if not class_stats:
            return None

        best_cid = max(
            class_stats,
            key=lambda cid: (class_stats[cid]["frames"], class_stats[cid]["mean_conf"]),
        )
        return self.model.names.get(best_cid, str(best_cid))

    def run(
        self,
        video_source: str,
        runs_dir: str,
        custom_tracker: str,
        gates: dict,
        arm_zones: dict,
    ) -> dict:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Kunne ikke åpne video: {video_source}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        timestamp = datetime.now().strftime("%H%M%S")
        source_stem = Path(video_source).stem
        out_dir = Path(runs_dir) / "instance"
        out_dir.mkdir(parents=True, exist_ok=True)
        avi_path = out_dir / f"{source_stem}_{timestamp}.avi"
        mp4_path = out_dir / f"{source_stem}_{timestamp}.mp4"

        out = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        x_margin = int((1 - self.config.inner_ratio_w) * w / 2)
        y_margin = int((1 - self.config.inner_ratio_h) * h / 2)
        x0, x1 = x_margin, w - x_margin
        y0, y1 = y_margin, h - y_margin

        frame_count = 0
        t0 = time.time()

        raw_ids_seen: set[int] = set()
        track_history: dict[int, list[tuple[int, int]]] = defaultdict(list)
        class_conf_history: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        track_frame_count: dict[int, int] = defaultdict(int)

        frames_with_dets_no_ids = 0
        total_raw_detections = 0

        od_counter = ODCounter(
            zone_dwell_frames=self.config.zone_dwell_frames,
            gate_band_px=self.config.gate_band_px,
            allow_unknown_origin=self.config.allow_unknown_origin,
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            roi = frame[y0:y1, x0:x1]
            results = self.model.track(
                roi,
                persist=True,
                conf=self.config.conf_thres,
                iou=self.config.iou_thres,
                tracker=custom_tracker,
                verbose=False,
            )

            annotated = frame.copy()
            annotated_roi = results[0].plot(color_mode="instance", line_width=2)
            annotated[y0:y1, x0:x1] = annotated_roi
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 2)

            if self.config.draw_zones and arm_zones:
                for zone_name, zone_info in arm_zones.items():
                    poly = zone_info.get("polygon", [])
                    if len(poly) < 3:
                        continue
                    poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [poly_np], color=(40, 90, 255))
                    annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)
                    cv2.polylines(annotated, [poly_np], isClosed=True, color=(40, 90, 255), thickness=2)
                    tx, ty = poly[0]
                    cv2.putText(
                        annotated,
                        f"ZONE {zone_name}",
                        (tx + 5, ty + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (40, 90, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if self.config.draw_gates and gates:
                for gate_name, gate_info in gates.items():
                    pts = gate_info.get("pts", [])
                    if len(pts) < 2:
                        continue
                    a = tuple(pts[0])
                    b = tuple(pts[1])
                    cv2.line(annotated, a, b, (255, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"GATE {gate_name}",
                        (a[0] + 5, a[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            has_boxes = (
                results[0].boxes is not None
                and results[0].boxes.xywh is not None
                and len(results[0].boxes) > 0
            )
            if has_boxes:
                total_raw_detections += len(results[0].boxes)
            if has_boxes and results[0].boxes.id is None:
                frames_with_dets_no_ids += 1

            if results[0].boxes.id is not None:
                boxes_xyxy = results[0].boxes.xyxy.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = (
                    results[0].boxes.cls.int().cpu().tolist()
                    if results[0].boxes.cls is not None
                    else [0] * len(track_ids)
                )
                confs = (
                    results[0].boxes.conf.cpu().tolist()
                    if results[0].boxes.conf is not None
                    else [1.0] * len(track_ids)
                )

                for box, tid, clsid, conf in zip(boxes_xyxy, track_ids, class_ids, confs):
                    tid = int(tid)
                    clsid = int(clsid)
                    box_full_frame = [
                        float(box[0] + x0),
                        float(box[1] + y0),
                        float(box[2] + x0),
                        float(box[3] + y0),
                    ]

                    raw_ids_seen.add(tid)
                    track_frame_count[tid] += 1
                    class_conf_history[tid][clsid].append(float(conf))

                    p = tracking_point_xyxy(
                        box_full_frame,
                        use_bottom_center=self.config.use_bottom_center,
                    )
                    p_int = (int(round(p[0])), int(round(p[1])))
                    cv2.circle(annotated, p_int, 3, (0, 0, 255), -1)

                    track = track_history[tid]
                    track.append(p_int)
                    if len(track) > self.config.track_display_len:
                        track.pop(0)
                    if len(track) > 1:
                        pts = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated, [pts], False, (0, 200, 255), 2)

                    cls_name = self._get_track_class(tid, class_conf_history)
                    if cls_name is None:
                        cls_name = self.model.names.get(clsid, str(clsid))

                    od_counter.update(
                        track_id=tid,
                        point=p_int,
                        gates=gates,
                        arm_zones=arm_zones,
                        cls_name=cls_name,
                        conf=float(conf),
                        frame_idx=frame_count - 1,
                    )

            if self.config.draw_count_overlay and od_counter.od_counts:
                y_text = 24
                for dest in sorted(od_counter.od_counts.keys()):
                    counts = od_counter.od_counts[dest]
                    summary = " | ".join(
                        f"{origin}: {sum(counts[origin].values())}"
                        for origin in sorted(counts.keys())
                    )
                    cv2.putText(
                        annotated,
                        f"{dest} <- {summary}",
                        (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (50, 255, 50),
                        2,
                        cv2.LINE_AA,
                    )
                    y_text += 22

            out.write(annotated)

            if frame_count % 100 == 0:
                elapsed_now = max(time.time() - t0, 1e-6)
                print(f"  {frame_count}/{total_frames} frames  |  {frame_count/elapsed_now:.1f} FPS", end="\r")

        cap.release()
        out.release()

        elapsed = max(time.time() - t0, 1e-6)
        self._convert_to_mp4(avi_path, mp4_path)

        stable_ids = [tid for tid in sorted(raw_ids_seen) if track_frame_count[tid] >= self.config.min_track_frames]
        short_ids = [tid for tid in sorted(raw_ids_seen) if track_frame_count[tid] < self.config.min_track_frames]

        print(f"\n✅ Lagret til: {mp4_path}")
        print(f"Tracker: {custom_tracker}")
        print(f"Tid: {elapsed:.1f}s  |  FPS: {frame_count/elapsed:.1f}")

        artifacts = TrackingArtifacts(
            mp4_path=str(mp4_path),
            elapsed_time=elapsed,
            unique_ids=len(stable_ids),
            track_ids=stable_ids,
            all_track_ids=sorted(raw_ids_seen),
            filtered_out_ids=short_ids,
            min_track_frames=self.config.min_track_frames,
            raw_detections=total_raw_detections,
            frames_with_dets_no_ids=frames_with_dets_no_ids,
            od_counts=od_counter.jsonable_counts(),
            od_events=od_counter.events_as_dict(),
        )
        return artifacts.__dict__

    @staticmethod
    def _convert_to_mp4(avi_path: Path, mp4_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(avi_path),
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(mp4_path),
            "-loglevel",
            "quiet",
        ]
        try:
            subprocess.run(cmd, check=True)
        finally:
            if avi_path.exists():
                os.remove(avi_path)
