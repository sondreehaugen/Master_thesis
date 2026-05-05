"""Configuration objects and path helpers for the tracking pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path



PROJECT_ROOT = Path("/content/drive/MyDrive/SVV/Master_Thesis/Model")

@dataclass
class AppConfig:
    """File names used to locate the model, tracker, and input video."""

    model_name: str = "yolo26_fine-tuned.pt"
    tracker_name: str = "my_new_botsort.yaml"
    video_name: str = "Fv587_Haukeland_1229050_00.mp4"


def build_paths(base_dir: Path, app: AppConfig) -> dict[str, Path]:
    """Build absolute paths for the model, tracker, video, and run directory."""

    return {
        "model_path": base_dir / app.model_name,
        "custom_tracker": base_dir / app.tracker_name,
        "video_source": base_dir / "Video" / app.video_name,
        "runs_dir": base_dir / "runs",
    }
