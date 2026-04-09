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
        cfg = rs.config()
        if self._serial:
            cfg.enable_device(self._serial)
        cfg.enable_stream(
            rs.stream.depth, self._width, self._height, rs.format.z16, self._fps
        )

        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(cfg)

        # Extract intrinsics from the depth stream profile
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = depth_stream.get_intrinsics()
        self._intrinsics = CameraIntrinsics(
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            width=intr.width,
            height=intr.height,
        )

        # Block until we receive the first valid frame (confirms device is live)
        self._pipeline.wait_for_frames(timeout_ms=5000)

    def get_frame(self) -> tuple[np.ndarray, float] | None:
        """Non-blocking poll. Returns (depth_u16, timestamp_s) or None.

        depth_u16: (H, W) uint16 array, values in millimetres.
        timestamp_s: hardware timestamp in seconds (monotonic).
        """
        if self._pipeline is None:
            return None

        frames = self._pipeline.poll_for_frames()
        if not frames:
            return None

        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return None

        # Hardware timestamp in milliseconds → convert to seconds
        timestamp_s = depth_frame.get_timestamp() / 1000.0

        # Zero-copy view of the depth data as uint16 (millimetres)
        depth_u16 = np.asanyarray(depth_frame.get_data(), dtype=np.uint16).reshape(
            self._height, self._width
        )

        return depth_u16, timestamp_s

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return depth stream intrinsics. Call after start()."""
        if self._intrinsics is None:
            raise RuntimeError("Camera not started — call start() first.")
        return self._intrinsics

    def stop(self) -> None:
        """Release device and pipeline resources."""
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
