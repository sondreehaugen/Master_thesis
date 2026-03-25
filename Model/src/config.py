from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrackingConfig:
    track_display_len: int = 150
    conf_thres: float = 0.6
    iou_thres: float = 0.4
    inner_ratio_w: float = 0.8
    inner_ratio_h: float = 0.8
    min_track_frames: int = 5
    zone_dwell_frames: int = 2
    gate_band_px: int = 10
    use_bottom_center: bool = True
    allow_unknown_origin: bool = False
    draw_gates: bool = True
    draw_zones: bool = True
    draw_count_overlay: bool = True


@dataclass(slots=True)
class AppConfig:
    model_name: str = "yolo26_fine-tuned.pt"
    tracker_name: str = "my_new_botsort.yaml"
    video_name: str = "Fv587_Haukeland_1229050_00.mp4"


def build_paths(base_dir: Path, app: AppConfig) -> dict[str, Path]:
    return {
        "model_path": base_dir / app.model_name,
        "custom_tracker": base_dir / app.tracker_name,
        "video_source": base_dir / "Video" / app.video_name,
        "runs_dir": base_dir / "runs",
    }
