"""Mock camera and detector for integration testing without hardware.

Provides MockCamera (generates synthetic depth frames with a ball at a
specified 3D position) and MockDetector (deterministic detection with
optional noise and dropout). These can be used to validate the full
RealPerceptionPipeline chain without a physical D435i or trained YOLO model.

Usage::

    from perception.real.mock import MockCamera, MockDetector

    cam = MockCamera()
    cam.start()
    cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))  # 50cm in front
    frame, ts = cam.get_frame()
    # frame has a ball-shaped blob at the correct pixel location
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .camera import CameraIntrinsics, D435iCamera
from .detector import BallDetector, Detection


# Default D435i 848x480 intrinsics (approximate factory values)
_DEFAULT_INTRINSICS = CameraIntrinsics(
    fx=425.0, fy=425.0,
    cx=424.0, cy=240.0,
    width=848, height=480,
)

_BALL_RADIUS_M = 0.020  # 40mm ping-pong ball


@dataclass
class MockCameraConfig:
    """Configuration for synthetic depth frame generation."""
    intrinsics: CameraIntrinsics = field(default_factory=lambda: _DEFAULT_INTRINSICS)
    ball_radius_m: float = _BALL_RADIUS_M
    depth_noise_std_mm: float = 2.0  # per-pixel Gaussian noise
    background_depth_mm: int = 0  # 0 = no return (like real D435i for far objects)


class MockCamera:
    """Synthetic D435i that renders depth frames with a ball at a known position.

    Implements the same interface as D435iCamera (start/get_frame/stop/get_intrinsics)
    so it can be swapped in for integration testing.
    """

    def __init__(self, config: MockCameraConfig | None = None) -> None:
        self._config = config or MockCameraConfig()
        self._started = False
        self._ball_pos_cam: np.ndarray | None = None  # (3,) in camera frame
        self._frame_count = 0
        self._start_time = 0.0

    def set_ball_pos_cam(self, pos_cam: np.ndarray) -> None:
        """Set the ball's 3D position in camera optical frame [X, Y, Z] metres."""
        self._ball_pos_cam = np.asarray(pos_cam, dtype=np.float64)

    def clear_ball(self) -> None:
        """Remove ball from scene (simulates ball out of view)."""
        self._ball_pos_cam = None

    def start(self) -> None:
        """No-op start (no hardware to initialise)."""
        self._started = True
        self._start_time = time.monotonic()
        self._frame_count = 0

    def get_frame(self) -> tuple[np.ndarray, float] | None:
        """Generate a synthetic depth frame with ball at current position.

        Returns (depth_u16, timestamp_s) matching D435iCamera interface.
        depth_u16 is (H, W) uint16 in millimetres.
        """
        if not self._started:
            return None

        intr = self._config.intrinsics
        cfg = self._config

        # Background depth frame
        depth = np.full(
            (intr.height, intr.width), cfg.background_depth_mm, dtype=np.uint16
        )

        if self._ball_pos_cam is not None:
            bx, by, bz = self._ball_pos_cam
            if bz > 0:
                # Project ball centre to pixel coordinates
                u_c = intr.fx * bx / bz + intr.cx
                v_c = intr.fy * by / bz + intr.cy

                # Ball apparent radius in pixels
                r_px = intr.fx * cfg.ball_radius_m / bz

                # Rasterise a filled circle at ball depth
                ball_depth_mm = int(bz * 1000)
                u_min = max(0, int(u_c - r_px - 1))
                u_max = min(intr.width, int(u_c + r_px + 2))
                v_min = max(0, int(v_c - r_px - 1))
                v_max = min(intr.height, int(v_c + r_px + 2))

                for v in range(v_min, v_max):
                    for u in range(u_min, u_max):
                        if (u - u_c) ** 2 + (v - v_c) ** 2 <= r_px ** 2:
                            noise = 0
                            if cfg.depth_noise_std_mm > 0:
                                noise = int(
                                    np.random.normal(0, cfg.depth_noise_std_mm)
                                )
                            d = max(0, ball_depth_mm + noise)
                            depth[v, u] = min(d, 65535)

        timestamp = time.monotonic() - self._start_time
        self._frame_count += 1
        return depth, timestamp

    def get_intrinsics(self) -> CameraIntrinsics:
        if not self._started:
            raise RuntimeError("Camera not started — call start() first.")
        return self._config.intrinsics

    def stop(self) -> None:
        self._started = False

    @property
    def frame_count(self) -> int:
        return self._frame_count


class MockDetector:
    """Deterministic ball detector for testing.

    Instead of running YOLO, returns the known ball position with optional
    noise and configurable dropout. Uses the same Detection interface as
    BallDetector.
    """

    def __init__(
        self,
        pos_noise_std: float = 0.002,  # metres, added to each axis
        dropout_rate: float = 0.0,  # probability of returning None
        min_depth: float = 0.168,
        max_depth: float = 2.0,
    ) -> None:
        self._pos_noise_std = pos_noise_std
        self._dropout_rate = dropout_rate
        self._min_depth = min_depth
        self._max_depth = max_depth

    def detect(
        self,
        depth_frame: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Detection | None:
        """Detect ball from synthetic depth frame.

        Finds the non-zero region in the depth frame (the ball blob rendered
        by MockCamera), computes its centroid, and returns a Detection with
        the deprojected 3D position plus optional noise.
        """
        if np.random.random() < self._dropout_rate:
            return None

        # Find ball pixels (non-background, within valid range)
        min_mm = int(self._min_depth * 1000)
        max_mm = int(self._max_depth * 1000)
        mask = (depth_frame >= min_mm) & (depth_frame <= max_mm)

        if not np.any(mask):
            return None

        # Compute centroid of ball pixels
        vs, us = np.where(mask)
        u_c = float(np.mean(us))
        v_c = float(np.mean(vs))

        # Bounding box
        x1, y1 = int(np.min(us)), int(np.min(vs))
        x2, y2 = int(np.max(us)) + 1, int(np.max(vs)) + 1

        # Median depth in bbox (same method as real detector)
        bbox = (x1, y1, x2, y2)
        roi = depth_frame[y1:y2, x1:x2].flatten()
        valid = roi[(roi >= min_mm) & (roi <= max_mm)]
        if len(valid) == 0:
            return None
        depth_m = float(np.median(valid)) / 1000.0

        # Deproject to camera frame
        pos_cam = intrinsics.deproject(u_c, v_c, depth_m)

        # Add noise
        if self._pos_noise_std > 0:
            pos_cam = pos_cam + np.random.normal(
                0, self._pos_noise_std, size=3
            ).astype(np.float32)

        return Detection(
            pos_cam=pos_cam,
            confidence=0.95,
            bbox=bbox,
            method="mock",
        )
