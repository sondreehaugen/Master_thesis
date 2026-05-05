"""
day_night_detection.py
Day/night detection and dynamic tracker switching.

Features:
- HSV saturation-based day/night detection (infrared camera discriminator)
- Hysteresis-based tracker switching to prevent flickering
- Smooth transitions across lighting conditions
"""

from collections import deque
from typing import Dict, Any
import matplotlib.pyplot as plt
import cv2
import numpy as np


class DayNightDetector:
    """
    Detects day/night conditions using HSV color saturation.
    Optimized for infrared night camera (pure grayscale, saturation ≈ 0)
    vs color day camera (saturation ≈ 60+).
    
    Parameters:
        sat_threshold: Mean HSV saturation below this (night mode threshold)
        window: Frames for rolling average (approx 2s at 30fps)
    """
    
    def __init__(
        self,
        sat_threshold: float = 20.0,
        window: int = 60
    ):
        self.sat_threshold = sat_threshold
        self.sat_history = deque(maxlen=window)
        self.lum_history = deque(maxlen=window)

    def analyze_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Analyze frame and return day/night mode with metrics.
        
        Returns:
            Dict with keys:
                - mode: "day" or "night"
                - sat_mean: Smoothed HSV saturation
                - lum_mean: Smoothed brightness
                - trigger: What triggered the detection
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        self.sat_history.append(float(hsv[:, :, 1].mean()))
        self.lum_history.append(float(gray.mean()))

        smoothed_sat = np.mean(self.sat_history)
        smoothed_lum = np.mean(self.lum_history)

        # Primary: saturation only (IR night cam = grayscale with low saturation)
        is_night = smoothed_sat < self.sat_threshold

        return {
            "mode": "night" if is_night else "day",
            "sat_mean": smoothed_sat,
            "lum_mean": smoothed_lum,
            "trigger": "saturation",
        }


class TrackerSwitcher:
    """
    Manages tracker mode switching with hysteresis to prevent flickering.
    Only commits to a new mode after it persists for a configurable number of seconds.
    """

    def __init__(self, initial_mode: str = "day", fps: float = 30.0, hold_seconds: float = 5.0):
        self.active_mode = initial_mode
        self.pending_mode = None
        self.hold_count = 0
        self.mode_switched = False
        self.hold_frames = max(1, int(round(hold_seconds * fps)))

    def update(self, proposed_mode: str) -> str:
        """
        Update with proposed mode. Returns actual active mode.
        Mode switches only after the configured hold time of persistence.
        """
        self.mode_switched = False
        
        if proposed_mode == self.active_mode:
            # Stable in current mode
            self.pending_mode = None
            self.hold_count = 0
        else:
            # Mode change proposed
            if proposed_mode != self.pending_mode:
                # New mode proposed; start counting
                self.pending_mode = proposed_mode
                self.hold_count = 0
            
            self.hold_count += 1
            if self.hold_count >= self.hold_frames:
                # Threshold reached; commit to new mode
                self.active_mode = proposed_mode
                self.pending_mode = None
                self.hold_count = 0
                self.mode_switched = True
        
        return self.active_mode


def plot_saturation_with_switch_points(
    video_path: str,
    sat_threshold: float = 20.0,
    initial_mode: str = "day",
    title: str | None = None,
    figsize: tuple[int, int] = (14, 5),
    show_plot: bool = True,
) -> Dict[str, Any]:
    """
    Analyze saturation over time and mark tracker mode switch points.

    This utility mirrors the day/night switching logic used in deployment:
    - rolling-window saturation from DayNightDetector
    - hysteresis switching from TrackerSwitcher

    Args:
        video_path: Path to input video
        sat_threshold: Threshold used for night/day classification
        initial_mode: Initial tracker mode for hysteresis switcher
        title: Optional custom plot title
        figsize: Matplotlib figure size
        show_plot: Whether to render the plot

    Returns:
        Dictionary containing:
            - fps: Video FPS
            - frame_count: Number of processed frames
            - sat_series: Rolling saturation values
            - lum_series: Rolling luminance values
            - mode_series: Active mode per frame
            - switch_points: List[(frame_idx, mode, sat_mean)]
            - switch_points_seconds: List[(time_s, mode, sat_mean)]
            - sat_threshold: Threshold used in analysis
    """


    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    detector = DayNightDetector(sat_threshold=sat_threshold, window=int(fps * 2))
    switcher = TrackerSwitcher(initial_mode=initial_mode, fps=fps)

    sat_series = []
    lum_series = []
    mode_series = []
    switch_points = []  # (frame_idx, mode, sat_mean)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        detection = detector.analyze_frame(frame)
        active_mode = switcher.update(detection["mode"])

        sat_series.append(float(detection["sat_mean"]))
        lum_series.append(float(detection["lum_mean"]))
        mode_series.append(active_mode)

        if switcher.mode_switched:
            switch_points.append((frame_idx, active_mode, float(detection["sat_mean"])))

    cap.release()

    if not sat_series:
        raise RuntimeError("No frames were read from the video.")

    t = np.arange(1, len(sat_series) + 1) / float(fps)

    if show_plot:
        plt.figure(figsize=figsize)
        plt.plot(t, sat_series, label="sat_mean (rolling)", linewidth=1.8, color="#2276b2")
        plt.axhline(
            sat_threshold,
            color="crimson",
            linestyle="--",
            linewidth=1.6,
            label=f"sat_threshold = {sat_threshold}",
        )

        # Mark switch points
        for f_idx, mode, sat_val in switch_points:
            ts = f_idx / float(fps)
            color = "#ff7f0e" if mode == "night" else "#2ca02c"
            plt.axvline(ts, color=color, linestyle=":", alpha=0.85)
            plt.scatter([ts], [sat_val], color=color, s=35, zorder=5)

        plt.title(title or f"sat_mean over time with switch points\n{video_path}")
        plt.xlabel("Time (s)")
        plt.ylabel("Mean HSV saturation")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    switch_points_seconds = [
        (f_idx / float(fps), mode, sat_val)
        for f_idx, mode, sat_val in switch_points
    ]

    return {
        "fps": float(fps),
        "frame_count": int(len(sat_series)),
        "sat_series": sat_series,
        "lum_series": lum_series,
        "mode_series": mode_series,
        "switch_points": switch_points,
        "switch_points_seconds": switch_points_seconds,
        "sat_threshold": float(sat_threshold),
    }
