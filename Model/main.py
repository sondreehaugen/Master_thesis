from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.config import AppConfig, TrackingConfig, build_paths
from src.tracker_runner import TrackingRunner
from src.visualization import plot_od_routes, print_od_matrix


ARM_ZONES = {
    "S": {
        "polygon": [(50, 275), (430, 275), (320, 375), (50, 375)],
        "dir": "S",
    },
    "N": {
        "polygon": [(240, 125), (400, 125), (400, 210), (240, 210)],
        "dir": "N",
    },
    "E": {
        "polygon": [(425, 170), (790, 170), (790, 305), (425, 305)],
        "dir": "E",
    },
}

GATES = {
    "S": {"pts": [(90, 300), (400, 300)], "dir": "S"},
    "N": {"pts": [(420, 180), (250, 180)], "dir": "N"},
    "E": {"pts": [(420, 300), (420, 180)], "dir": "E"},
}


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore

        return True
    except Exception:
        return False


def maybe_mount_drive() -> None:
    if not in_colab():
        return
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive", force_remount=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vehicle tracking pipeline.")
    parser.add_argument("--colab", action="store_true", help="Use Google Colab paths and mount Drive.")
    parser.add_argument(
        "--drive-base",
        type=str,
        default="/content/drive/MyDrive/SVV/Master_Thesis/Model",
        help="Base folder in Google Drive when using --colab.",
    )
    parser.add_argument("--model", type=str, default="yolo26_fine-tuned.pt")
    parser.add_argument("--tracker", type=str, default="my_new_botsort.yaml")
    parser.add_argument("--video", type=str, default="Fv587_Haukeland_1229050_00.mp4")
    parser.add_argument("--run-bytetrack", action="store_true", help="Also run ByteTrack after primary tracker.")
    return parser.parse_args()


def run_once(tracker_name: str, app: AppConfig, cfg: TrackingConfig, base_dir: Path) -> dict:
    app_run = AppConfig(model_name=app.model_name, tracker_name=tracker_name, video_name=app.video_name)
    paths = build_paths(base_dir, app_run)
    paths["runs_dir"].mkdir(parents=True, exist_ok=True)

    model = YOLO(str(paths["model_path"]))
    runner = TrackingRunner(model=model, config=cfg)

    return runner.run(
        video_source=str(paths["video_source"]),
        runs_dir=str(paths["runs_dir"]),
        custom_tracker=str(paths["custom_tracker"] if tracker_name.endswith(".yaml") and tracker_name != "bytetrack.yaml" else tracker_name),
        gates=GATES,
        arm_zones=ARM_ZONES,
    )


def main() -> None:
    args = parse_args()

    if args.colab:
        maybe_mount_drive()
        base_dir = Path(args.drive_base)
    else:
        base_dir = Path(__file__).resolve().parent

    app = AppConfig(
        model_name=args.model,
        tracker_name=args.tracker,
        video_name=args.video,
    )
    cfg = TrackingConfig(
        conf_thres=0.6,
        iou_thres=0.4,
        inner_ratio_w=0.8,
        inner_ratio_h=0.8,
        min_track_frames=5,
        zone_dwell_frames=2,
    )

    print(f"\nBase dir: {base_dir}")
    print("\n=== Primary tracker ===")
    results_primary = run_once(app.tracker_name, app, cfg, base_dir)
    print_od_matrix(results_primary["od_counts"])

    print("\nOutput video:")
    print(f"  {app.tracker_name}: {results_primary['mp4_path']}")

    if args.run_bytetrack:
        print("\n=== ByteTrack ===")
        results_bytetrack = run_once("bytetrack.yaml", app, cfg, base_dir)
        print_od_matrix(results_bytetrack["od_counts"])

        print("\n=== Plot fra siste kjøring ===")
        plot_od_routes(results_bytetrack["od_counts"])
        print(f"  ByteTrack: {results_bytetrack['mp4_path']}")


if __name__ == "__main__":
    main()
