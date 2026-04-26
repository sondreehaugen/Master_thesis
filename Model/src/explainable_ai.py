"""
Explainable AI module for YOLO object detection and tracking.
Simple, practical methods for understanding model predictions.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import seaborn as sns
from matplotlib.patches import Rectangle


class DetectionVisualizer:
    """
    Visualize YOLO detections with confidence scores and bounding boxes.
    This is the most reliable way to explain what the model detected.
    """

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def visualize_detections(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.1,
    ) -> Tuple[np.ndarray, list]:
        """
        Visualize detections with bounding boxes and confidence.

        Args:
            image: Input image (BGR)
            target_size: YOLO input size
            conf_threshold: Detection confidence threshold

        Returns:
            annotated_image: Image with bounding boxes
            detections: List of detected objects
        """
        # Run inference
        results = self.model(image, verbose=False, conf=conf_threshold)
        detections = results[0]
        
        # Plot results
        annotated = detections.plot()
        
        # Convert to BGR for consistency
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        
        # Extract detection info
        det_info = []
        if len(detections.boxes) > 0 and detections.boxes.conf is not None:
            for i, box in enumerate(detections.boxes.xyxy):
                conf = detections.boxes.conf[i] if detections.boxes.conf is not None else 0.0
                cls = detections.boxes.cls[i] if detections.boxes.cls is not None else 0
                det_info.append({
                    "box": box.cpu().numpy(),
                    "confidence": float(conf),
                    "class": int(cls),
                    "class_name": self.model.names[int(cls)]
                })
        
        return annotated, det_info

    @staticmethod
    def create_confidence_heatmap(
        image: np.ndarray,
        detections: list,
    ) -> np.ndarray:
        """
        Create a heatmap showing detection confidence across the image.
        Brighter areas = higher confidence detections.
        """
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = det["confidence"]
            # Fill detection region with confidence value
            heatmap[y1:y2, x1:x2] = max(heatmap[y1:y2, x1:x2].max(), conf)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap

    @staticmethod
    def visualize_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        cmap: str = "YlOrRd",
    ) -> np.ndarray:
        """Overlay confidence heatmap on image."""
        colormap = plt.get_cmap(cmap)
        heatmap_colored = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        heatmap_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
        result = cv2.addWeighted(image, 1 - alpha, heatmap_bgr, alpha, 0)
        return result


class SimpleAttentionMap:
    """
    Create simple attention maps based on detection density.
    Shows regions where objects are being detected.
    """

    def __init__(self, model):
        self.model = model

    def create_attention_map(
        self,
        image: np.ndarray,
        window_size: int = 32,
    ) -> np.ndarray:
        """
        Divide image into windows and count detections in each.
        """
        h, w = image.shape[:2]
        attention = np.zeros((h // window_size, w // window_size), dtype=np.float32)
        
        results = self.model(image, verbose=False)
        detections = results[0]
        
        if len(detections.boxes) > 0:
            for box in detections.boxes.xyxy:
                x1, y1, x2, y2 = box.cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                row = min(int(cy // window_size), attention.shape[0] - 1)
                col = min(int(cx // window_size), attention.shape[1] - 1)
                attention[row, col] += 1
        
        # Normalize
        if attention.max() > 0:
            attention = attention / attention.max()
        
        # Upsample to image size
        attention = cv2.resize(attention, (w, h), interpolation=cv2.INTER_LINEAR)
        return attention

    @staticmethod
    def visualize_attention(
        image: np.ndarray,
        attention: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay attention map on image."""
        colormap = plt.get_cmap("hot")
        attention_colored = (colormap(attention)[:, :, :3] * 255).astype(np.uint8)
        attention_bgr = cv2.cvtColor(attention_colored, cv2.COLOR_RGB2BGR)
        result = cv2.addWeighted(image, 1 - alpha, attention_bgr, alpha, 0)
        return result



class ExplainableAIPipeline:
    """Complete pipeline for visual explanation of YOLO detections."""

    def __init__(self, model):
        self.model = model
        self.detector = DetectionVisualizer(model)
        self.attention = SimpleAttentionMap(model)

    def explain_detection(
        self,
        image: np.ndarray,
        methods: List[str] = ["detections", "confidence"],
        figsize: Tuple[int, int] = (15, 5),
    ) -> Tuple[plt.Figure, Dict]:
        """
        Create visual explanations for model predictions.
        
        Args:
            image: Input image (BGR)
            methods: List of visualization methods
            figsize: Figure size
        
        Returns:
            Figure and results dictionary
        """
        results = {}
        num_methods = len(methods) + 1  # +1 for original
        
        fig, axes = plt.subplots(1, num_methods, figsize=figsize)
        if num_methods == 1:
            axes = [axes]

        # Original image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image", fontweight="bold")
        axes[0].axis("off")

        # Method 1: Detection boxes
        det_info = None
        if "detections" in methods:
            annotated, det_info = self.detector.visualize_detections(image)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            axes[1].imshow(annotated_rgb)
            axes[1].set_title(f"Detections ({len(det_info)} objects)", fontweight="bold")
            axes[1].axis("off")
            results["detections"] = det_info
            results["annotated"] = annotated
            idx = 2
        else:
            idx = 1

        # Method 2: Confidence heatmap
        if "confidence" in methods and "detections" in methods:
            heatmap = self.detector.create_confidence_heatmap(image, det_info)
            heatmap_viz = self.detector.visualize_heatmap(image, heatmap)
            heatmap_rgb = cv2.cvtColor(heatmap_viz, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(heatmap_rgb)
            axes[idx].set_title("Confidence Heatmap", fontweight="bold")
            axes[idx].axis("off")
            results["confidence_heatmap"] = heatmap
            idx += 1

        # Method 3: Attention/detection density
        if "attention" in methods:
            attention = self.attention.create_attention_map(image)
            attention_viz = self.attention.visualize_attention(image, attention)
            attention_rgb = cv2.cvtColor(attention_viz, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(attention_rgb)
            axes[idx].set_title("Detection Density", fontweight="bold")
            axes[idx].axis("off")
            results["attention"] = attention

        plt.tight_layout()
        return fig, results