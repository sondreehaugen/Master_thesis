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
    Only commits to a new mode after it persists for a threshold (≈5s at 30fps).
    """
    
    HOLD_FRAMES = 150  # ~5 seconds at 30fps before switching
    
    def __init__(self, initial_mode: str = "day"):
        self.active_mode = initial_mode
        self.pending_mode = None
        self.hold_count = 0
        self.mode_switched = False

    def update(self, proposed_mode: str) -> str:
        """
        Update with proposed mode. Returns actual active mode.
        Mode switches only after HOLD_FRAMES frames of persistence.
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
            if self.hold_count >= self.HOLD_FRAMES:
                # Threshold reached; commit to new mode
                self.active_mode = proposed_mode
                self.pending_mode = None
                self.hold_count = 0
                self.mode_switched = True
        
        return self.active_mode
