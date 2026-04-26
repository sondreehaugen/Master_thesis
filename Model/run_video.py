import os
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np

from .src.geometry import centroid_xyxy, bottom_center_xyxy, tracking_point_xyxy, point_in_polygon

def run_video(
    model,
    video_source,
    runs_dir,
    custom_tracker,
    track_display_len: int = 150,
    conf_thres=0.25,
    iou_thres=0.45,
    inner_ratio_w=0.85,
    inner_ratio_h=0.85,
    min_track_frames: int = 5,
    gate_zones: dict | None = None,
    draw_gate_zones: bool = True,
    draw_roi_box: bool = True,
    use_roi: bool = True,
    draw_count_overlay: bool = True,
    use_bottom_center: bool = True,
    zone_dwell_frames: int = 1,
    is_live: bool = False,
    save_video: bool = False,
    max_frames: int | None = None,
    night_mode: bool = False,
    clahe_clip_limit=1.2,
    clahe_tile_grid_size=(10, 10),
    glare_threshold=230,
    glare_reduce_strength=0.7,
    # --- NEW: fallback for split IDs ---
    enable_split_id_fallback: bool = True,
    split_max_gap: int = 80,
    split_max_dist_raw: float = 180.0,
    split_max_dist_pred: float = 160.0,
    split_class_penalty: float = 40.0,
    split_min_fragment_frames: int = 8,
    split_require_same_class: bool = True,
    split_require_both_distances: bool = True,
) -> dict:
    """
    Én samlet funksjon for batch + live tracking med valgfri night mode.
    OD for kjøretøy telles som første stabile vehiclesone -> siste stabile vehiclesone.
    Inkluderer valgfri split-ID fallback (post-prosess) for occlusion/new-ID.
    """

    if is_live:
        import base64
        from IPython.display import HTML, display

    def preprocess_night_roi(roi):
        if not night_mode:
            return roi

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=tuple(clahe_tile_grid_size),
        )
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        alpha = float(np.clip(glare_reduce_strength, 0.0, 1.0))
        if alpha > 0:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            glare_mask = gray >= int(glare_threshold)
            if np.any(glare_mask):
                blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
                mixed = cv2.addWeighted(enhanced, 1.0 - alpha, blurred, alpha, 0)
                enhanced[glare_mask] = mixed[glare_mask]

        return enhanced

    # ---------- NEW helpers for split-ID fallback ----------
    def _euclid(a, b):
        return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))

    def _zone_of_point(pt):
        for zname, zinfo in gate_zones.items():
            poly = zinfo.get("polygon", [])
            if len(poly) < 3:
                continue
            if point_in_polygon((int(round(pt[0])), int(round(pt[1]))), poly):
                return zinfo.get("dir", zname)
        return None

    def _main_class(clsid_list):
        if not clsid_list:
            return -1
        vals, counts = np.unique(np.array(clsid_list, dtype=int), return_counts=True)
        return int(vals[np.argmax(counts)])

    def _build_fragments_from_tracks():
        by_tid = defaultdict(lambda: {"frames": [], "points": [], "clsids": []})

        for fidx, dets in all_tracks_by_frame.items():
            for d in dets:
                if len(d) < 4:
                    continue
                tid, box_full_frame, _, clsid = d[0], d[1], d[2], d[3]
                tid = int(tid)
                p = tracking_point_xyxy(box_full_frame)

                by_tid[tid]["frames"].append(int(fidx))
                by_tid[tid]["points"].append((float(p[0]), float(p[1])))
                by_tid[tid]["clsids"].append(int(clsid) if clsid is not None else -1)

        frags = []
        for tid, d in by_tid.items():
            if len(d["frames"]) < 2:
                continue

            order = np.argsort(d["frames"])
            frames = [d["frames"][i] for i in order]
            points = [d["points"][i] for i in order]
            clsids = [d["clsids"][i] for i in order]

            dt = max(1, frames[-1] - frames[0])
            vx = (points[-1][0] - points[0][0]) / dt
            vy = (points[-1][1] - points[0][1]) / dt

            frags.append(
                {
                    "id": tid,
                    "start_f": frames[0],
                    "end_f": frames[-1],
                    "n_frames": len(frames),
                    "start_p": points[0],
                    "end_p": points[-1],
                    "start_zone": _zone_of_point(points[0]),
                    "end_zone": _zone_of_point(points[-1]),
                    "main_clsid": _main_class(clsids),
                    "vel": (vx, vy),
                }
            )
        return frags

    def _apply_split_id_fallback():
        if not enable_split_id_fallback:
            return {"added_same_id": 0, "added_split_id": 0}
        if not gate_zones:
            return {"added_same_id": 0, "added_split_id": 0}
        if not all_tracks_by_frame:
            return {"added_same_id": 0, "added_split_id": 0}

        frags = _build_fragments_from_tracks()
        if not frags:
            return {"added_same_id": 0, "added_split_id": 0}

        counted_ids = set(int(e.get("track_id")) for e in od_events if "track_id" in e)

        # A) Person dwell fallback (for fragments that stayed in same person zone)
        for f in frags:
            tid = int(f["id"])
            if tid in counted_ids:
                continue
            # Only for person zones where start == end
            if f["start_zone"] is None or f["end_zone"] is None:
                continue
            if _get_zone_type(f["start_zone"]) != "person":
                continue
            if f["start_zone"] != f["end_zone"]:
                continue
            # Person zones must only count person class
            main_cls_name = model.names.get(int(f["main_clsid"]), str(int(f["main_clsid"])))
            if not _is_person_class(clsid=f["main_clsid"], cls_name=main_cls_name):
                continue
            # Check if dwell time is long enough
            duration = f["end_f"] - f["start_f"]
            if duration >= person_dwell_threshold:
                if tid not in person_counted[f["start_zone"]]:
                    person_counted[f["start_zone"]].add(tid)
                    od_events.append({
                        "frame": int(f["end_f"]),
                        "track_id": tid,
                        "zone": f["start_zone"],
                        "class": main_cls_name,
                        "count_method": "fallback_person_dwell",
                    })
                    counted_ids.add(tid)

        # B) Same-ID missing trip (vehicles only)
        added_same = 0
        for f in frags:
            tid = int(f["id"])
            if tid in counted_ids:
                continue
            if f["start_zone"] is None or f["end_zone"] is None:
                continue
            if f["start_zone"] == f["end_zone"]:
                continue

            cls_name = model.names.get(int(f["main_clsid"]), str(int(f["main_clsid"])))
            # Only count if origin and dest are same type
            if not _should_count_od(f["start_zone"], f["end_zone"], cls_name):
                continue
            
            od_counts[f["end_zone"]][f["start_zone"]][cls_name] += 1
            od_events.append(
                {
                    "frame": int(f["end_f"]),
                    "track_id": tid,
                    "origin": f["start_zone"],
                    "dest": f["end_zone"],
                    "class": cls_name,
                    "conf": None,
                    "count_method": "fallback_same_id",
                }
            )
            counted_ids.add(tid)
            added_same += 1

        # B) Split-ID matching
        starts = [
            f for f in frags
            if int(f["id"]) not in counted_ids
            and f["start_zone"] is not None
            and int(f.get("n_frames", 0)) >= int(split_min_fragment_frames)
        ]
        ends = [
            f for f in frags
            if int(f["id"]) not in counted_ids
            and f["end_zone"] is not None
            and int(f.get("n_frames", 0)) >= int(split_min_fragment_frames)
        ]

        starts.sort(key=lambda x: x["end_f"])
        ends.sort(key=lambda x: x["start_f"])

        used_end_ids = set()
        added_split = 0

        for a in starts:
            best = None
            best_score = float("inf")

            for b in ends:
                if int(b["id"]) == int(a["id"]):
                    continue
                if int(b["id"]) in used_end_ids:
                    continue

                gap = int(b["start_f"]) - int(a["end_f"])
                if gap < 1 or gap > split_max_gap:
                    continue

                if a["start_zone"] is None or b["end_zone"] is None:
                    continue
                if a["start_zone"] == b["end_zone"]:
                    continue

                if split_require_same_class and int(a["main_clsid"]) != int(b["main_clsid"]):
                    continue

                pred = (
                    a["end_p"][0] + a["vel"][0] * gap,
                    a["end_p"][1] + a["vel"][1] * gap,
                )

                d_raw = _euclid(a["end_p"], b["start_p"])
                d_pred = _euclid(pred, b["start_p"])

                if split_require_both_distances:
                    if d_raw > split_max_dist_raw or d_pred > split_max_dist_pred:
                        continue
                else:
                    if d_raw > split_max_dist_raw and d_pred > split_max_dist_pred:
                        continue

                score = 0.35 * d_raw + 0.65 * d_pred
                if (not split_require_same_class) and int(a["main_clsid"]) != int(b["main_clsid"]):
                    score += float(split_class_penalty)

                if score < best_score:
                    best_score = score
                    best = b

            if best is None:
                continue

            cls_name = model.names.get(int(best["main_clsid"]), str(int(best["main_clsid"])))
            # Only count if origin and dest are same type
            if not _should_count_od(a["start_zone"], best["end_zone"], cls_name):
                continue
            
            od_counts[best["end_zone"]][a["start_zone"]][cls_name] += 1
            od_events.append(
                {
                    "frame": int(best["end_f"]),
                    "track_id": int(a["id"]),
                    "origin": a["start_zone"],
                    "dest": best["end_zone"],
                    "class": cls_name,
                    "conf": None,
                    "count_method": "fallback_split_id",
                    "matched_with_id": int(best["id"]),
                    "match_gap": int(best["start_f"] - a["end_f"]),
                    "match_d_raw": float(_euclid(a["end_p"], best["start_p"])),
                    "match_d_pred": float(
                        _euclid(
                            (
                                a["end_p"][0] + a["vel"][0] * (best["start_f"] - a["end_f"]),
                                a["end_p"][1] + a["vel"][1] * (best["start_f"] - a["end_f"]),
                            ),
                            best["start_p"],
                        )
                    ),
                    "match_score": float(best_score),
                }
            )

            used_end_ids.add(int(best["id"]))
            counted_ids.add(int(a["id"]))
            added_split += 1

        return {"added_same_id": added_same, "added_split_id": added_split}

    gate_zones = gate_zones or {}

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Kunne ikke åpne video: {video_source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else -1

    if use_roi:
        x_margin = int((1 - inner_ratio_w) * w / 2)
        y_margin = int((1 - inner_ratio_h) * h / 2)
        x0, x1 = x_margin, w - x_margin
        y0, y1 = y_margin, h - y_margin
    else:
        x0, y0 = 0, 0
        x1, y1 = w, h

    out = None
    mp4_path = None
    avi_path = None

    if save_video:
        timestamp = datetime.now().strftime("%H%M%S")
        source_stem = os.path.splitext(os.path.basename(str(video_source)))[0]
        out_dir = os.path.join(runs_dir, "instance")
        os.makedirs(out_dir, exist_ok=True)
        mp4_path = os.path.join(out_dir, f"{source_stem}_{timestamp}.mp4")
        avi_path = None  # No longer using intermediate AVI file
        out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    raw_ids_seen = set()
    frame_count = 0
    t0 = time.time()

    track_history = defaultdict(list)
    class_conf_history = defaultdict(lambda: defaultdict(list))
    track_frame_count = defaultdict(int)

    frames_with_dets_no_ids = 0
    total_raw_detections = 0
    all_tracks_by_frame = defaultdict(list)

    zone_hist = defaultdict(lambda: deque(maxlen=max(zone_dwell_frames, 1)))
    first_vehicle_zone = {}
    last_vehicle_zone = {}
    last_vehicle_frame = {}
    last_vehicle_conf = {}
    last_vehicle_class = {}
    live_first_vehicle_zone = {}
    live_last_vehicle_zone = {}
    live_last_vehicle_class = {}
    live_track_last_seen_frame = {}
    live_track_finalized = set()
    live_counted_trip = set()
    live_od_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    live_finalize_gap = max(int(zone_dwell_frames) * 4, 10)
    seen_zones_for_id = defaultdict(set)
    od_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    od_events = []
    
    # Person dwell counting: {zone_name: {track_id: frame_count}}
    person_dwell_counts = defaultdict(lambda: defaultdict(int))
    person_counted = defaultdict(set)  # {zone_name: set of counted track_ids}
    person_dwell_threshold = 10  # frames

    def get_track_class(tid):
        if tid not in class_conf_history:
            return None

        class_stats = {
            cid: {
                "frames": len(cvals),
                "mean_conf": float(np.mean(cvals)),
            }
            for cid, cvals in class_conf_history[tid].items()
            if len(cvals) > 0
        }
        if not class_stats:
            return None

        stable_classes = {
            cid: stats for cid, stats in class_stats.items() if stats["frames"] >= 5
        }

        if stable_classes:
            best_cid = max(stable_classes, key=lambda cid: stable_classes[cid]["mean_conf"])
        else:
            best_cid = max(class_stats, key=lambda cid: class_stats[cid]["frames"])

        return model.names.get(best_cid, str(best_cid))

    def _get_zone_type(zone_name):
        """Hent sonetype (vehicle/person) fra gate_zones."""
        if not gate_zones or zone_name not in gate_zones:
            return "vehicle"  # Default
        return gate_zones[zone_name].get("type", "vehicle")

    def _should_count_od(origin, dest, cls_name):
        """
        Sjekk om OD-paret skal telles.
        - vehicle->vehicle: ja
        - person->person: ja
        - vehicle->person eller person->vehicle: nei (krysses ikke telles)
        """
        if origin is None or dest is None:
            return False
        origin_type = _get_zone_type(origin)
        dest_type = _get_zone_type(dest)
        return origin_type == dest_type

    def _is_person_class(clsid=None, cls_name=None):
        """Return True only for pedestrian classes."""
        if cls_name is None and clsid is not None:
            cls_name = model.names.get(int(clsid), str(clsid))
        n = str(cls_name).strip().lower() if cls_name is not None else ""
        return n in {"person", "pedestrian", "people"}

    def _finalize_first_last_vehicle_counts():
        """
        Count one OD event per track from first to last stable vehicle zone.
        This allows trajectories like E -> N -> S to be counted as E->S.
        """
        added = 0
        for tid, origin in first_vehicle_zone.items():
            dest = last_vehicle_zone.get(tid)
            if origin is None or dest is None or origin == dest:
                continue

            cls_name = last_vehicle_class.get(tid)
            if cls_name is None:
                cls_name = get_track_class(tid) or "unknown"

            if not _should_count_od(origin, dest, cls_name):
                continue

            od_counts[dest][origin][cls_name] += 1
            od_events.append(
                {
                    "frame": int(last_vehicle_frame.get(tid, -1)),
                    "track_id": int(tid),
                    "origin": origin,
                    "dest": dest,
                    "class": cls_name,
                    "conf": last_vehicle_conf.get(tid, None),
                    "count_method": "first_last_zone",
                }
            )
            added += 1

        return added

    def _update_live_vehicle_state(tid, stable_zone, cls_name, frame_idx):
        """Update live state for a track without counting immediately."""
        if stable_zone is None or _get_zone_type(stable_zone) != "vehicle":
            return

        live_track_last_seen_frame[tid] = int(frame_idx)

        if tid not in live_first_vehicle_zone:
            live_first_vehicle_zone[tid] = stable_zone
            live_last_vehicle_zone[tid] = stable_zone
            live_last_vehicle_class[tid] = cls_name
            return

        live_last_vehicle_zone[tid] = stable_zone
        live_last_vehicle_class[tid] = cls_name

    def _finalize_live_vehicle_counts(current_frame_idx, force: bool = False):
        """Finalize live preview counts after a track has been inactive long enough."""
        added = 0
        for tid, origin in list(live_first_vehicle_zone.items()):
            if tid in live_track_finalized:
                continue

            last_seen = live_track_last_seen_frame.get(tid)
            if last_seen is None:
                continue

            if not force and (int(current_frame_idx) - int(last_seen) < live_finalize_gap):
                continue

            dest = live_last_vehicle_zone.get(tid)
            if origin is None or dest is None or origin == dest:
                live_track_finalized.add(tid)
                continue

            cls_name = live_last_vehicle_class.get(tid)
            if cls_name is None:
                cls_name = get_track_class(tid) or "unknown"

            if not _should_count_od(origin, dest, cls_name):
                live_track_finalized.add(tid)
                continue

            live_od_counts[dest][origin][cls_name] += 1
            live_counted_trip.add(tid)
            live_track_finalized.add(tid)
            added += 1

        return added

    if is_live:
        display_handle = display(HTML("<img id='live'>"), display_id=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames is not None and frame_count >= max_frames:
            break

        frame_count += 1

        roi = frame[y0:y1, x0:x1]
        roi_proc = preprocess_night_roi(roi)

        results = model.track(
            roi_proc,
            persist=True,
            conf=conf_thres,
            iou=iou_thres,
            tracker=custom_tracker,
            verbose=False,
        )

        annotated = frame.copy()
        annotated_roi = results[0].plot(color_mode="instance", line_width=2)
        annotated[y0:y1, x0:x1] = annotated_roi

        if draw_roi_box and use_roi:
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 2)

        if draw_gate_zones and gate_zones:
            for zone_name, zone_info in gate_zones.items():
                poly = zone_info.get("polygon", [])
                if len(poly) < 3:
                    continue
                poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [poly_np], color=(255, 255, 0))
                annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)
                cv2.polylines(annotated, [poly_np], True, (255, 255, 0), 2)
                tx, ty = poly[0]
                cv2.putText(
                    annotated, zone_name, (tx + 5, ty + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA
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
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
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

                box_full_frame = [box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0]
                all_tracks_by_frame[frame_count - 1].append((tid, box_full_frame, conf, clsid, None))

                raw_ids_seen.add(tid)
                track_frame_count[tid] += 1
                class_conf_history[tid][clsid].append(float(conf))

                p = tracking_point_xyxy(box_full_frame)
                p_int = (int(round(p[0])), int(round(p[1])))

                cv2.circle(annotated, p_int, 3, (0, 0, 255), -1)

                track = track_history[tid]
                track.append(p_int)
                if len(track) > track_display_len:
                    track.pop(0)
                if len(track) > 1:
                    pts = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [pts], isClosed=False, color=(0, 200, 255), thickness=2)

                cls_name = get_track_class(tid)
                if cls_name is None:
                    cls_name = model.names.get(clsid, str(clsid))
                is_person_det = _is_person_class(clsid=clsid, cls_name=cls_name)

                current_zone = None
                for zone_name, zone_info in gate_zones.items():
                    poly = zone_info.get("polygon", [])
                    if len(poly) < 3:
                        continue
                    if point_in_polygon(p_int, poly):
                        current_zone = zone_info.get("dir", zone_name)
                        break

                zone_hist[tid].append(current_zone)

                # --- PERSON DWELL COUNTING ---
                if (
                    current_zone is not None
                    and _get_zone_type(current_zone) == "person"
                    and is_person_det
                ):
                    person_dwell_counts[current_zone][tid] += 1
                    # Check if threshold reached
                    if (tid not in person_counted[current_zone] and 
                        person_dwell_counts[current_zone][tid] >= person_dwell_threshold):
                        # Mark as counted in this zone
                        person_counted[current_zone].add(tid)
                        od_events.append({
                            "frame": frame_count - 1,
                            "track_id": tid,
                            "zone": current_zone,
                            "class": cls_name,
                            "count_method": "person_dwell",
                        })

                # --- VEHICLE ZONE TRACKING (first/last stable zone) ---
                stable_zone = None
                vals = [z for z in zone_hist[tid] if z is not None]
                if len(vals) >= zone_dwell_frames and len(vals[-zone_dwell_frames:]) == zone_dwell_frames:
                    tail = vals[-zone_dwell_frames:]
                    if len(set(tail)) == 1:
                        stable_zone = tail[-1]

                if stable_zone is not None and track_frame_count[tid] >= min_track_frames:
                    seen_zones_for_id[tid].add(stable_zone)

                    # Only keep first/last for vehicle zones
                    if _get_zone_type(stable_zone) != "vehicle":
                        continue

                    if tid not in first_vehicle_zone:
                        first_vehicle_zone[tid] = stable_zone

                    last_vehicle_zone[tid] = stable_zone
                    last_vehicle_frame[tid] = frame_count - 1
                    last_vehicle_conf[tid] = float(conf) if conf is not None else None
                    last_vehicle_class[tid] = cls_name

                    # Track live state; final counting happens after inactivity.
                    _update_live_vehicle_state(tid, stable_zone, cls_name, frame_count - 1)

        if is_live or save_video:
            _finalize_live_vehicle_counts(frame_count - 1, force=False)

        if draw_count_overlay:
            y_text = 24
            # Use preview counts while processing so the overlay is visible in live view
            # and in saved video output. Final od_counts are finalized after the loop.
            overlay_counts = live_od_counts if (is_live or save_video) else od_counts

            preview_total = 0
            for dest_counts in overlay_counts.values():
                for origin_counts in dest_counts.values():
                    preview_total += int(sum(origin_counts.values()))

            cv2.putText(
                annotated,
                f"Preview OD total: {preview_total}",
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y_text += 24
            
            # Vehicle counts
            if overlay_counts:
                for dest in sorted(overlay_counts.keys()):
                    counts = overlay_counts[dest]
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
            
            # person counts
            if person_counted:
                y_text += 10
                for zone_name in sorted(person_counted.keys()):
                    person_count = len(person_counted[zone_name])
                    if person_count > 0:
                        cv2.putText(
                            annotated,
                            f"{zone_name}: {person_count} persons",
                            (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (50, 255, 50),  # Same as vehicle direction overlay
                            2,
                            cv2.LINE_AA,
                        )
                        y_text += 22

        if night_mode:
            cv2.putText(
                annotated,
                "Night mode: ON",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        if out:
            out.write(annotated)

        if is_live and frame_count % 5 == 0:
            _, buffer = cv2.imencode(".jpg", annotated)
            jpg_as_text = base64.b64encode(buffer).decode()
            display_handle.update(HTML(f"<img src='data:image/jpeg;base64,{jpg_as_text}' width=900>"))

        if frame_count % 100 == 0:
            elapsed_now = max(time.time() - t0, 1e-6)
            status_str = f"{frame_count}/{total_frames} frames" if not is_live else f"{frame_count} frames"
            print(f"  {status_str}  |  {frame_count/elapsed_now:.1f} FPS", end="\r")

    cap.release()
    if out:
        out.release()

    elapsed = max(time.time() - t0, 1e-6)

    # Video already saved directly as MP4 - no conversion needed

    # Finalize primary vehicle OD counts from first->last stable zones
    first_last_added = _finalize_first_last_vehicle_counts()

    # Flush live preview counts for tracks still active at the end.
    live_preview_added = _finalize_live_vehicle_counts(frame_count, force=True)

    # Fallback for split IDs / occlusions (adds missing trips only)
    fallback_stats = _apply_split_id_fallback()

    stable_ids = [tid for tid in sorted(raw_ids_seen) if track_frame_count[tid] >= min_track_frames]
    short_ids = [tid for tid in sorted(raw_ids_seen) if track_frame_count[tid] < min_track_frames]

    print(f"Lagret til: {mp4_path}")
    print(f"\nTracking ferdig | Tid: {elapsed:.1f}s | FPS: {frame_count/elapsed:.1f}")
    print(f"Stabile track-IDer (>= {min_track_frames} frames): {len(stable_ids)}")
    print(f"Primær OD (first->last) lagt til: {first_last_added}")
    if is_live or save_video:
        print(f"Live preview OD lagt til: {live_preview_added}")
    if enable_split_id_fallback:
        print(
            f"Fallback lagt til -> same_id: {fallback_stats['added_same_id']}, "
            f"split_id: {fallback_stats['added_split_id']}"
        )

    return {
        "mp4_path": mp4_path,
        "elapsed_time": elapsed,
        "unique_ids": len(stable_ids),
        "track_ids": stable_ids,
        "all_track_ids": sorted(raw_ids_seen),
        "filtered_out_ids": short_ids,
        "min_track_frames": min_track_frames,
        "raw_detections": total_raw_detections,
        "frames_with_dets_no_ids": frames_with_dets_no_ids,
        "all_tracks_by_frame": all_tracks_by_frame,
        "od_counts": {d: {o: dict(c) for o, c in od_counts[d].items()} for d in od_counts},
        "person_counts": {zone: len(counted) for zone, counted in person_counted.items()},
        "od_events": od_events,
        "seen_zones_for_id": {tid: sorted(v) for tid, v in seen_zones_for_id.items()},
        "first_gate": dict(first_vehicle_zone),
        "last_gate": dict(last_vehicle_zone),
        "night_mode": night_mode,
        "split_id_fallback": bool(enable_split_id_fallback),
        "first_last_primary_count": int(first_last_added),
        "live_preview_count": int(live_preview_added),
        "fallback_stats": fallback_stats,
    }