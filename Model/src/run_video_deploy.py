"""
run_video_deploy.py
Deployment-optimized tracking with dynamic day/night tracker switching.

Orchestrates:
- Frame processing and detection
- Day/night mode detection and tracker switching
- Zone-based counting (O-D matrix)
- Split ID recovery across tracker switches
- Video output with annotations
"""

import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

import cv2
import numpy as np

from .day_night_detection import DayNightDetector, TrackerSwitcher
from .zone_counting import ZoneCounter
from .split_id_fallback import SplitIDRecovery
from .geometry import tracking_point_xyxy

def run_video_deploy(
    model,
    video_source,
    runs_dir: str,
    day_tracker: Optional[str] = None,
    night_tracker: Optional[str] = None,
    track_display_len: int = 150,
    conf_thres: float = 0.35,
    iou_thres: float = 0.35,
    inner_ratio_w: float = 0.9,
    inner_ratio_h: float = 0.9,
    min_track_frames: int = 5,
    gate_zones: Optional[Dict] = None,
    draw_gate_zones: bool = True,
    draw_count_overlay: bool = True,
    draw_tracker_info: bool = True,
    use_bottom_center: bool = True,
    zone_dwell_frames: int = 3,
    save_video: bool = True,
    max_frames: Optional[int] = None,
    enable_split_id_fallback: bool = True,
    split_max_dist_raw: float = 180.0,
    split_max_dist_pred: float = 160.0,
    split_class_penalty: float = 40.0,
    track_finalize_gap_seconds: float = 2.0,
    split_max_gap_seconds: float = 3.0,
    sat_threshold: float = 20.0,
) -> Dict[str, Any]:
    """
    Deploy-optimized tracking with dynamic day/night tracker switching.
    
    Supports three modes:
    - day_tracker only: Uses same tracker for entire video
    - night_tracker only: Uses same tracker for entire video  
    - Both: Dynamically switches based on lighting conditions
    
    Args:
        model: YOLO model instance
        video_source: Path to video file
        runs_dir: Output directory for results
        day_tracker: Path to day tracker config (e.g., "bytetrack.yaml"). Optional if night_tracker provided.
        night_tracker: Path to night tracker config (e.g., "night_botsort_conservative.yaml"). Optional if day_tracker provided.
        track_display_len: Trajectory history length
        conf_thres: Detection confidence threshold
        iou_thres: IOU threshold for NMS
        inner_ratio_w, inner_ratio_h: ROI dimensions as fraction of frame
        min_track_frames: Min frames to consider stable track
        gate_zones: Dict of counting zones {name: {polygon, dir, type}}
        draw_gate_zones: Draw zone overlays
        draw_count_overlay: Draw counting statistics
        draw_tracker_info: Draw active tracker name + metrics on frame
        use_bottom_center: Use bottom-center vs centroid for tracking point
        zone_dwell_frames: Frames in same zone = stable
        save_video: Write output video
        max_frames: Limit processing (None = all frames)
        enable_split_id_fallback: Post-process to recover split track IDs
        track_finalize_gap_seconds: Seconds of inactivity before finalizing track (default 2s)
        split_max_gap_seconds: Max gap in seconds for split ID matching (default 3s)
        sat_threshold: HSV saturation threshold for day/night detection (< threshold = night)
    
    Returns:
        Dict with tracking results, counts, and metadata
    """
    
    # ========================================================================
    # SETUP & INITIALIZATION
    # ========================================================================
    
    # Validate tracker inputs
    if day_tracker is None and night_tracker is None:
        raise ValueError("At least one of day_tracker or night_tracker must be provided")
    
    # Determine switching mode
    has_day = day_tracker is not None
    has_night = night_tracker is not None
    enable_switching = has_day and has_night  # Only switch if both provided
    
    if has_day and has_night:
        print(f"Dual-tracker mode: {day_tracker} (day) ↔ {night_tracker} (night)")
    elif has_day:
        print(f"Day-tracker only: {day_tracker}")
    else:
        print(f"Night-tracker only: {night_tracker}")
    
    # Validate inputs
    gate_zones = gate_zones or {}
    
    # Open video
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_source}.")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ROI extraction
    x_margin = int((1 - inner_ratio_w) * w / 2)
    y_margin = int((1 - inner_ratio_h) * h / 2)
    x0, x1 = x_margin, w - x_margin
    y0, y1 = y_margin, h - y_margin
    
    # Initialize components
    detector = DayNightDetector(
        sat_threshold=sat_threshold,
        window=int(fps * 2)  # 2 seconds of frames
    )
    switcher = TrackerSwitcher(initial_mode="day")
    counter = ZoneCounter(
        gate_zones=gate_zones,
        fps=fps,
        model=model,
        zone_dwell_frames=zone_dwell_frames,
        min_track_frames=min_track_frames,
        track_finalize_gap_seconds=track_finalize_gap_seconds,
    )
    fallback = SplitIDRecovery(
        model=model,
        gate_zones=gate_zones,
        fps=fps,
        enable=enable_split_id_fallback,
        max_dist_raw=split_max_dist_raw,
        max_dist_pred=split_max_dist_pred,
        class_penalty=split_class_penalty,
        max_gap_seconds=split_max_gap_seconds,
    )
    
    # Video output
    out = None
    mp4_path = None
    if save_video:
        timestamp = datetime.now().strftime("%H%M%S")
        source_stem = os.path.splitext(os.path.basename(str(video_source)))[0]
        out_dir = os.path.join(runs_dir, "instance")
        os.makedirs(out_dir, exist_ok=True)
        mp4_path = os.path.join(out_dir, f"{source_stem}_{timestamp}.mp4")
        out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    # Tracking state
    frame_count = 0
    t0 = time.time()
    
    raw_ids_seen = set()
    frames_with_dets_no_ids = 0
    total_raw_detections = 0
    all_tracks_by_frame = {}
    
    # Tracker mode statistics
    mode_frames = {"day": 0, "night": 0}
    mode_switches = []
    
    # ========================================================================
    # MAIN PROCESSING LOOP
    # ========================================================================
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames is not None and frame_count >= max_frames:
            break
        
        frame_count += 1
        
        # Detect day/night mode
        active_mode = "night" if has_night else "day"
        detection = None
        
        if enable_switching:
            detection = detector.analyze_frame(frame)
            active_mode = switcher.update(detection["mode"])
            mode_frames[active_mode] += 1
            
            if switcher.mode_switched:
                mode_switches.append({
                    "frame": frame_count,
                    "mode": active_mode,
                    "sat": detection["sat_mean"],
                    "lum": detection["lum_mean"],
                })
                print(
                    f"\n TRACKER SWITCH at frame {frame_count}: "
                    f"{active_mode.upper()} (sat={detection['sat_mean']:.1f}, "
                    f"lum={detection['lum_mean']:.1f})"
                )
        
        # Select tracker
        if enable_switching:
            active_tracker = night_tracker if active_mode == "night" else day_tracker
        elif has_night:
            active_tracker = night_tracker
        else:
            active_tracker = day_tracker
        
        # Extract ROI and run detection
        roi = frame[y0:y1, x0:x1]
        results = model.track(
            roi,
            persist=True,
            conf=conf_thres,
            iou=iou_thres,
            tracker=active_tracker,
            verbose=False,
        )
        
        # Prepare output frame
        annotated = frame.copy()
        annotated_roi = results[0].plot(color_mode="instance", line_width=2)
        annotated[y0:y1, x0:x1] = annotated_roi
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 2)
        
        # Draw gate zones
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
        
        # Process detections
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
            
            # Store detections for split ID fallback
            all_tracks_by_frame[frame_count - 1] = []
            
            for box, tid, clsid, conf in zip(boxes_xyxy, track_ids, class_ids, confs):
                tid = int(tid)
                clsid = int(clsid)
                
                box_full_frame = [box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0]
                all_tracks_by_frame[frame_count - 1].append(
                    (tid, box_full_frame, conf, clsid, None)
                )
                
                raw_ids_seen.add(tid)
                
                # Process zone counting
                counter.process_detection(
                    tid=tid,
                    box_full_frame=tuple(box_full_frame),
                    clsid=clsid,
                    conf=conf,
                    frame_idx=frame_count,
                )
                
                # Draw tracking point
                p = tracking_point_xyxy(box_full_frame)
                p_int = (int(round(p[0])), int(round(p[1])))
                cv2.circle(annotated, p_int, 3, (0, 0, 255), -1)
                
                # Draw track history
                track = counter.track_history[tid]
                if len(track) > 1:
                    pts = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [pts], isClosed=False, color=(0, 200, 255), thickness=2)
        
        # Draw overlays
        od_counts = counter.od_counts
        if draw_count_overlay and od_counts:
            y_text = 24
            for dest in sorted(od_counts.keys()):
                counts = od_counts[dest]
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
        
        # Draw tracker info
        if draw_tracker_info:
            if detection is not None:
                tracker_text = f"Tracker: {active_mode.upper()} | Sat: {detection['sat_mean']:.1f} | Lum: {detection['lum_mean']:.1f}"
            else:
                tracker_text = f"Tracker: {active_mode.upper()}"
            color = (0, 165, 255) if active_mode == "night" else (50, 255, 50)
            cv2.putText(
                annotated,
                tracker_text,
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        
        # Write output video
        if out:
            out.write(annotated)
        
        # Progress
        if frame_count % 100 == 0:
            elapsed_now = max(time.time() - t0, 1e-6)
            print(f"  {frame_count}/{total_frames} | {frame_count/elapsed_now:.1f} FPS", end="\r")
        
        # Finalize tracks periodically
        if frame_count % 30 == 0:
            counter.finalize_counts(frame_count, force_all=False)
    
    # ========================================================================
    # FINALIZATION
    # ========================================================================
    print("\n  Finalizing remaining tracks...")
    finalized = counter.finalize_counts(frame_count, force_all=True)
    print(f"  Finalized {finalized} remaining tracks")
    
    cap.release()
    if out:
        out.release()
    
    elapsed = max(time.time() - t0, 1e-6)
    
    # Apply split ID fallback
    fallback_stats = fallback.apply(all_tracks_by_frame, counter.od_events, counter.od_counts)
    
    # Summary statistics
    stable_ids = [
        tid for tid in sorted(raw_ids_seen) 
        if counter.track_frame_count.get(tid, 0) >= min_track_frames
    ]
    short_ids = [
        tid for tid in sorted(raw_ids_seen) 
        if counter.track_frame_count.get(tid, 0) < min_track_frames
    ]
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Tracking Complete")
    print(f"{'='*70}")
    print(f"Output: {mp4_path}")
    print(f"Duration: {elapsed:.1f}s | FPS: {frame_count/elapsed:.1f}")
    print(f"Frames: {frame_count} | Day: {mode_frames['day']} | Night: {mode_frames['night']}")
    print(f"Mode switches: {len(mode_switches)}")
    print(f"Detections: {total_raw_detections} | Stable tracks: {len(stable_ids)}")
    print(f"Counted trips: {len(counter.od_events)}")
    if counter.person_counted:
        person_summary = {z: len(ids) for z, ids in counter.person_counted.items()}
        print(f"Person counts by zone: {person_summary}")
    if enable_split_id_fallback:
        print(
            f"Fallback recovery: same_id={fallback_stats['added_same_id']}, "
            f"split_id={fallback_stats['added_split_id']}"
        )
    print(f"{'='*70}\n")
    
    # Get results from counter
    counter_results = counter.get_results()
    
    return {
        "mp4_path": mp4_path,
        "elapsed_time": elapsed,
        "frame_count": frame_count,
        "unique_ids": len(stable_ids),
        "track_ids": stable_ids,
        "all_track_ids": sorted(raw_ids_seen),
        "filtered_out_ids": short_ids,
        "min_track_frames": min_track_frames,
        "raw_detections": total_raw_detections,
        "frames_with_dets_no_ids": frames_with_dets_no_ids,
        "all_tracks_by_frame": all_tracks_by_frame,
        "od_counts": counter_results["od_counts"],
        "od_events": counter_results["od_events"],
        "person_counts": counter_results["person_counts"],
        "seen_zones_for_id": counter_results["seen_zones_for_id"],
        "first_gate": counter_results["first_gate"],
        "last_gate": counter_results["last_gate"],
        "mode_frames": mode_frames,
        "mode_switches": mode_switches,
        "split_id_fallback": bool(enable_split_id_fallback),
        "fallback_stats": fallback_stats,
    }
