"""
Example: Using run_video_deploy.py

This shows how to use the deployment-optimized tracker with dynamic day/night switching.
"""

from ultralytics import YOLO
from run_video_deploy import run_video_deploy

# Load model
model = YOLO("YOLO26m_final.pt")

# Define gate zones (if needed)
GATE_ZONES = {
    "Zone_A": {
        "polygon": [[100, 100], [300, 100], [300, 300], [100, 300]],
        "dir": "A→B",
        "type": "vehicle"  # or "person"
    },
    "Zone_B": {
        "polygon": [[400, 100], [600, 100], [600, 300], [400, 300]],
        "dir": "B→A",
        "type": "vehicle"
    },
}

# Run with dynamic tracker switching
results = run_video_deploy(
    model=model,
    video_source="path/to/video.mp4",
    runs_dir="./runs",
    day_tracker="bytetrack.yaml",              # Day mode tracker
    night_tracker="night_botsort_conservative.yaml",  # Night mode tracker
    track_display_len=150,
    conf_thres=0.25,
    iou_thres=0.45,
    inner_ratio_w=0.9,
    inner_ratio_h=0.9,
    min_track_frames=5,
    gate_zones=GATE_ZONES,
    draw_gate_zones=True,
    draw_count_overlay=True,
    draw_tracker_info=True,  # Shows active tracker + metrics
    save_video=True,
    sat_threshold=15.0,        # HSV saturation < 15 = night
    brightness_threshold=70,   # Grayscale mean < 70 = night (backup)
)

# Results include:
# - mp4_path: Output video path
# - mode_frames: {day: N, night: M}
# - mode_switches: List of switch events with frame/metrics
# - od_counts: Origin-destination counts
# - od_events: List of counting events
# - fallback_stats: Split ID recovery statistics

print(f"Mode switches: {results['mode_switches']}")
print(f"OD Counts: {results['od_counts']}")
