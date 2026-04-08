"""D435i depth camera driver wrapper.

Non-blocking depth-only streaming at 848x480 @ 90fps via pyrealsense2.
No ROS2 in the hot path (adds 2-8ms latency per frame).

Requires: pip install pyrealsense2
Hardware: Intel RealSense D435i, USB-C at 5V/3A on Orin NX.

See docs/lit_review_d435i_ros2_integration.md for setup details.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore[import-untyped]
except ImportError:
    rs = None  # allow import on machines without the SDK


@dataclass
class CameraIntrinsics:
    """D435i depth camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def deproject(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """Pixel (u, v, depth) -> camera-frame 3D point [X, Y, Z]."""
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        return np.array([x, y, depth_m], dtype=np.float32)


class D435iCamera:
    """Non-blocking D435i depth stream at 848x480 @ 90fps.

    Usage::

        cam = D435iCamera()
        cam.start()
        while running:
            result = cam.get_frame()
            if result is not None:
                depth_u16, timestamp = result
                # depth_u16 is (480, 848) uint16 in millimetres
        cam.stop()
    """

    def __init__(
        self,
        serial: str | None = None,
        width: int = 848,
        height: int = 480,
        fps: int = 90,
    ) -> None:
        if rs is None:
            raise ImportError(
                "pyrealsense2 not installed. "
                "Install with: pip install pyrealsense2"
            )
        self._serial = serial
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline: rs.pipeline | None = None
        self._intrinsics: CameraIntrinsics | None = None

    def start(self) -> None:
        """Enable depth stream. Blocks until first frame arrives."""
        raise NotImplementedError(
            "D435iCamera.start() is a stub — implement when hardware is available. "
            "See docs/hardware_pipeline_architecture.md §3.1 for spec."
        )

    def get_frame(self) -> tuple[np.ndarray, float] | None:
        """Non-blocking poll. Returns (depth_u16, timestamp_s) or None.

        depth_u16: (H, W) uint16 array, values in millimetres.
        timestamp_s: hardware timestamp in seconds (monotonic).
        """
        raise NotImplementedError("D435iCamera.get_frame() is a stub.")

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return depth stream intrinsics. Call after start()."""
        if self._intrinsics is None:
            raise RuntimeError("Camera not started — call start() first.")
        return self._intrinsics

    def stop(self) -> None:
        """Release device and pipeline resources."""
        raise NotImplementedError("D435iCamera.stop() is a stub.")
