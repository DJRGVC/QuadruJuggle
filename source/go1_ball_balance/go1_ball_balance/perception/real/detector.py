"""Ball detection from D435i depth frames.

Two-stage pipeline:
1. YOLOv8n+P2 (TensorRT FP16) detects ball bounding box in depth image
2. Median depth within bbox -> camera-frame 3D position

Fallback: Hough circle detection on depth image when YOLO is unavailable
or confidence < hough_fallback_thresh.

See docs/lit_review_yolo_ball_detection.md for model training recipe.
See docs/lit_review_realsense_d435i_noise.md for depth lookup details.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    cv2 = None

from .camera import CameraIntrinsics

# Expected ball radius in metres (40mm ping-pong ball)
_BALL_RADIUS_M = 0.020


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
        model_path: str | None = None,
        conf_thresh: float = 0.5,
        hough_fallback: bool = True,
        hough_fallback_thresh: float = 0.4,
        min_depth: float = 0.168,
        max_depth: float = 2.0,
        ball_radius_m: float = _BALL_RADIUS_M,
    ) -> None:
        self._model_path = model_path
        self._conf_thresh = conf_thresh
        self._hough_fallback = hough_fallback
        self._hough_fallback_thresh = hough_fallback_thresh
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._ball_radius_m = ball_radius_m
        self._model = None  # TensorRT engine, loaded lazily

    def detect(
        self,
        depth_frame: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Detection | None:
        """Detect ball in depth frame.

        Tries YOLO first (if model loaded). Falls back to Hough circle
        detection if YOLO is unavailable, returns None, or confidence is
        below hough_fallback_thresh.

        Args:
            depth_frame: (H, W) uint16 depth in millimetres.
            intrinsics: Camera intrinsic parameters.

        Returns:
            Detection with pos_cam in camera optical frame, or None if
            ball not detected.
        """
        # Primary: YOLO (if model is available)
        yolo_det = None
        if self._model is not None:
            yolo_det = self._detect_yolo(depth_frame, intrinsics)
            if yolo_det is not None and yolo_det.confidence >= self._conf_thresh:
                return yolo_det

        # Fallback: Hough circle detection
        if self._hough_fallback:
            hough_det = self._detect_hough(depth_frame, intrinsics)
            if hough_det is not None:
                return hough_det

        # Return low-confidence YOLO result if nothing else worked
        if yolo_det is not None:
            return yolo_det

        return None

    def _detect_yolo(
        self, depth_frame: np.ndarray, intrinsics: CameraIntrinsics
    ) -> Detection | None:
        """YOLO-based detection (primary path)."""
        raise NotImplementedError("_detect_yolo is a stub.")

    def _detect_hough(
        self, depth_frame: np.ndarray, intrinsics: CameraIntrinsics
    ) -> Detection | None:
        """Hough circle detection on depth image.

        Converts depth to 8-bit image, applies Gaussian blur, then
        HoughCircles with radius bounds derived from known ball size
        and depth range. Selects the circle closest to expected ball
        radius at its measured depth.

        Works well for isolated spherical objects (ping-pong ball) against
        a low-texture background (typical for upward-facing paddle camera).
        """
        if cv2 is None:
            return None

        min_mm = int(self._min_depth * 1000)
        max_mm = int(self._max_depth * 1000)

        # Create valid-depth mask
        valid = (depth_frame >= min_mm) & (depth_frame <= max_mm)
        if not np.any(valid):
            return None

        # Normalise valid depth to 8-bit for Hough transform
        # Map [min_mm, max_mm] → [1, 255], invalid = 0
        depth_f = depth_frame.astype(np.float32)
        img8 = np.zeros(depth_frame.shape, dtype=np.uint8)
        img8[valid] = np.clip(
            1 + (depth_f[valid] - min_mm) * 254.0 / (max_mm - min_mm),
            1, 255,
        ).astype(np.uint8)

        # Gaussian blur to suppress depth noise
        blurred = cv2.GaussianBlur(img8, (5, 5), 1.5)

        # Compute pixel radius bounds from ball size + depth range
        # r_px = fx * ball_radius / depth
        r_min_px = max(3, int(intrinsics.fx * self._ball_radius_m / self._max_depth))
        r_max_px = max(r_min_px + 1, int(intrinsics.fx * self._ball_radius_m / self._min_depth) + 2)

        # dp=1.5 gives good results for small circles; minDist prevents
        # double-detections of the same ball
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=r_max_px * 3,
            param1=50,
            param2=25,
            minRadius=r_min_px,
            maxRadius=r_max_px,
        )

        if circles is None:
            return None

        circles = circles[0]  # shape (N, 3): [cx, cy, radius]

        # Score each circle: prefer radius closest to expected at its depth
        best_det = None
        best_score = -1.0

        for cx, cy, r_px in circles:
            u, v = float(cx), float(cy)
            iu, iv = int(round(u)), int(round(v))

            # Clamp to image bounds
            iv = max(0, min(iv, depth_frame.shape[0] - 1))
            iu = max(0, min(iu, depth_frame.shape[1] - 1))

            # Get depth at circle centre — use median in a small patch
            patch_r = max(1, int(r_px * 0.5))
            y1 = max(0, iv - patch_r)
            y2 = min(depth_frame.shape[0], iv + patch_r + 1)
            x1 = max(0, iu - patch_r)
            x2 = min(depth_frame.shape[1], iu + patch_r + 1)
            bbox = (x1, y1, x2, y2)

            depth_m = self._median_depth_in_bbox(depth_frame, bbox, min_mm, max_mm)
            if depth_m is None:
                continue

            # Expected pixel radius at this depth
            expected_r = intrinsics.fx * self._ball_radius_m / depth_m
            # Score: how close is detected radius to expected?
            radius_ratio = min(r_px, expected_r) / max(r_px, expected_r)

            if radius_ratio > best_score:
                best_score = radius_ratio
                # Deproject circle centre to 3D
                pos_cam = intrinsics.deproject(u, v, depth_m)
                # Confidence: radius_ratio mapped to [0, 1]
                confidence = float(np.clip(radius_ratio, 0.0, 1.0))

                # Full bounding box from circle
                bx1 = max(0, int(u - r_px))
                by1 = max(0, int(v - r_px))
                bx2 = min(depth_frame.shape[1], int(u + r_px) + 1)
                by2 = min(depth_frame.shape[0], int(v + r_px) + 1)

                best_det = Detection(
                    pos_cam=pos_cam,
                    confidence=confidence,
                    bbox=(bx1, by1, bx2, by2),
                    method="hough",
                )

        return best_det

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
