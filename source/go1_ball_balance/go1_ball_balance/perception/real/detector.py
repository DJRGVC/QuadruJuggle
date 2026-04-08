"""Ball detection from D435i depth frames.

Two-stage pipeline:
1. YOLOv8n+P2 (TensorRT FP16) detects ball bounding box in depth image
2. Median depth within bbox -> camera-frame 3D position

Fallback: Hough circle detection when YOLO confidence < 0.4.

See docs/lit_review_yolo_ball_detection.md for model training recipe.
See docs/lit_review_realsense_d435i_noise.md for depth lookup details.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraIntrinsics


@dataclass
class Detection:
    """Single ball detection result."""

    pos_cam: np.ndarray  # (3,) ball position in camera optical frame [X, Y, Z] metres
    confidence: float  # YOLO confidence score (0-1)
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coordinates
    method: str = "yolo"  # "yolo" or "hough"


class BallDetector:
    """Detect 40mm ping-pong ball in D435i depth frame.

    Usage::

        detector = BallDetector("models/ball_detector.engine")
        detection = detector.detect(depth_frame, intrinsics)
        if detection is not None:
            ball_pos_cam = detection.pos_cam  # (3,) in camera frame
    """

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.5,
        hough_fallback: bool = True,
        min_depth: float = 0.168,
        max_depth: float = 2.0,
    ) -> None:
        self._model_path = model_path
        self._conf_thresh = conf_thresh
        self._hough_fallback = hough_fallback
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._model = None  # TensorRT engine, loaded lazily

    def detect(
        self,
        depth_frame: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Detection | None:
        """Detect ball in depth frame.

        Args:
            depth_frame: (H, W) uint16 depth in millimetres.
            intrinsics: Camera intrinsic parameters.

        Returns:
            Detection with pos_cam in camera optical frame, or None if
            ball not detected.
        """
        raise NotImplementedError(
            "BallDetector.detect() is a stub — implement when YOLO model "
            "is trained. See docs/hardware_pipeline_architecture.md §3.2."
        )

    def _detect_yolo(
        self, depth_frame: np.ndarray, intrinsics: CameraIntrinsics
    ) -> Detection | None:
        """YOLO-based detection (primary path)."""
        raise NotImplementedError("_detect_yolo is a stub.")

    def _detect_hough(
        self, depth_frame: np.ndarray, intrinsics: CameraIntrinsics
    ) -> Detection | None:
        """Hough circle fallback when YOLO confidence < 0.4."""
        raise NotImplementedError("_detect_hough is a stub.")

    @staticmethod
    def _median_depth_in_bbox(
        depth_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        min_depth_mm: int = 168,
        max_depth_mm: int = 2000,
    ) -> float | None:
        """Median of valid depth pixels within bounding box.

        Returns depth in metres, or None if no valid pixels.
        Uses median (not mean) to reject single-pixel outliers from
        background bleed — a known D435i artifact at object edges.
        """
        x1, y1, x2, y2 = bbox
        roi = depth_frame[y1:y2, x1:x2].flatten()
        valid = roi[(roi >= min_depth_mm) & (roi <= max_depth_mm)]
        if len(valid) == 0:
            return None
        return float(np.median(valid)) / 1000.0  # mm -> metres
