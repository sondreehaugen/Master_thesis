"""
Vehicle tracking utility module for multi-object tracking with YOLO and custom trackers.
Provides functions for video processing, tracking configuration, and result analysis.
"""

import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from ultralytics import YOLO


def setup_tracking_context(
    model_name: str = "YOLO26m_final.pt",
    tracker_yaml: str = "day_botsort.yaml",
    video_name: str = "Fv587_Haukeland_stortelling.mp4",
    drive_base: str = "/content/drive/MyDrive/SVV/Master_Thesis/Model",
    apply_globals: bool = True,
) -> dict:
    """
    Setup tracking context by loading model and validating video source.
    
    Args:
        model_name: YOLO model file name
        tracker_yaml: Tracker configuration file name or path
        video_name: Video file name in Video/ directory
        drive_base: Base directory path
        apply_globals: Whether to update global variables
    
    Returns:
        Dictionary containing model, tracker, video paths and configuration
    """
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    # Build the core workspace paths used by the notebook and deployment scripts.
    runs_dir = f"{drive_base}/runs"
    model_path = f"{drive_base}/{model_name}"
    
    # Accept either an absolute path or a known tracker file name.
    if tracker_yaml.startswith("/") or tracker_yaml in ["botsort.yaml", "bytetrack.yaml"]:
        custom_tracker = tracker_yaml
    else:
        custom_tracker = f"{drive_base}/{tracker_yaml}"
    
    video_source = f"{drive_base}/Video/{video_name}"

    os.makedirs(runs_dir, exist_ok=True)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_source)
    video_ok = cap.isOpened()
    if video_ok:
        ret, frame = cap.read()
    else:
        ret, frame = False, None
    cap.release()

    cfg = {
        "model_name": model_name,
        "tracker_yaml": tracker_yaml,
        "video_name": video_name,
        "in_colab": in_colab,
        "drive_base": drive_base,
        "runs_dir": runs_dir,
        "model_path": model_path,
        "custom_tracker": custom_tracker,
        "video_source": video_source,
        "model": model,
        "class_names": model.names,
        "ret": ret,
        "frame": frame,
    }

    if apply_globals:
        globals().update(cfg)

    return cfg


def display_lines_on_frame(video_source, lines_to_draw=None, arm_zones=None):
    """
    Visualize detection zones on the first frame of a video.
    
    Args:
        video_source: Path to video file
        lines_to_draw: Dictionary of lines (not currently used)
        arm_zones: Dictionary of zone definitions with polygon coordinates
                   Expected format: {"zone_name": {"polygon": [...], "dir": "direction"}}
    
    Returns:
        The lines_to_draw dictionary (passed through)
    """
    if lines_to_draw is None:
        lines_to_draw = {}

    cap = cv2.VideoCapture(video_source)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame from: {video_source}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.imshow(frame_rgb)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    ax.yaxis.set_major_locator(plt.MaxNLocator(20))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Draw arm zones
    if arm_zones:
        zone_keys = list(arm_zones.keys())
        cmap = plt.cm.get_cmap("tab10", max(len(zone_keys), 1))
        zone_colors = {z: cmap(i) for i, z in enumerate(zone_keys)}

        for zone_name, zone_data in arm_zones.items():
            polygon = zone_data.get("polygon", [])
            if len(polygon) < 3:
                continue

            direction = zone_data.get("dir", zone_name)
            polygon_closed = polygon + [polygon[0]]
            poly_x, poly_y = zip(*polygon_closed)
            color = zone_colors[zone_name]

            ax.fill(poly_x, poly_y, color=color, alpha=0.28, zorder=2)
            zone_line, = ax.plot(
                poly_x, poly_y,
                linestyle="-", linewidth=1.5, color=color, alpha=1.0,
                label=f"Zone {zone_name} ({direction})", zorder=3
            )
            zone_line.set_path_effects([pe.Stroke(linewidth=7, foreground="black", alpha=0.9), pe.Normal()])

            center_x = sum(p[0] for p in polygon) / len(polygon)
            center_y = sum(p[1] for p in polygon) / len(polygon)
            txt = ax.text(
                center_x, center_y, str(direction),
                color="white", fontsize=14, ha="center", va="center",
                fontweight="bold",
                bbox=dict(facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.95),
                zorder=4
            )
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    return lines_to_draw