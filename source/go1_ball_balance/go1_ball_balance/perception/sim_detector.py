"""Ball detection from Isaac Lab TiledCamera output (sim only).

Detects the ball in float32 depth frames by finding the closest small blob
within the expected depth range. Much simpler than the real detector — in sim
the ball is the only small close object above the robot.

Returns camera-frame 3D position compatible with the EKF pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    cv2 = None


@dataclass
class SimDetection:
    """Ball detection result from sim camera."""

    pos_cam: np.ndarray  # (3,) camera-frame [X, Y, Z] metres
    pixel_uv: tuple[float, float]  # (u, v) pixel centre
    depth_m: float  # depth at detection centre
    num_pixels: int  # blob size in pixels


class SimBallDetector:
    """Detect ball in Isaac Lab TiledCamera depth output.

    The detector works on float32 depth images (metres) as produced by
    Isaac Lab's ``distance_to_image_plane`` data type.

    Strategy:
      1. Threshold depth to [min_depth, max_depth] to isolate nearby objects
      2. Find connected components in the valid-depth mask
      3. Select the component closest to expected ball pixel size
      4. Deproject blob centroid to 3D using camera intrinsics

    Args:
        focal_length_px: Camera focal length in pixels (fx ≈ fy).
        cx: Principal point x (pixels). Defaults to width/2.
        cy: Principal point y (pixels). Defaults to height/2.
        min_depth: Minimum valid depth (metres). Default 0.05.
        max_depth: Maximum valid depth (metres). Default 2.0.
        ball_radius_m: Known ball radius (metres). Default 0.020 (40mm ball).
        min_blob_px: Minimum blob area to consider (pixels). Default 4.
    """

    def __init__(
        self,
        focal_length_px: float,
        cx: float | None = None,
        cy: float | None = None,
        min_depth: float = 0.05,
        max_depth: float = 2.0,
        ball_radius_m: float = 0.020,
        min_blob_px: int = 4,
    ) -> None:
        self._fx = focal_length_px
        self._fy = focal_length_px  # square pixels
        self._cx = cx
        self._cy = cy
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._ball_radius_m = ball_radius_m
        self._min_blob_px = min_blob_px

    def detect(self, depth: np.ndarray) -> SimDetection | None:
        """Detect ball in a float32 depth image.

        Args:
            depth: (H, W) float32 depth in metres from TiledCamera.

        Returns:
            SimDetection or None if no ball found.
        """
        if cv2 is None:
            raise ImportError("cv2 required for SimBallDetector")

        h, w = depth.shape[:2]
        cx = self._cx if self._cx is not None else w / 2.0
        cy = self._cy if self._cy is not None else h / 2.0

        # Step 1: valid depth mask
        valid = np.isfinite(depth) & (depth >= self._min_depth) & (depth <= self._max_depth)
        if not valid.any():
            return None

        mask = valid.astype(np.uint8) * 255

        # Step 2: connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        if num_labels <= 1:
            return None  # only background

        # Step 3: score each component by how close its pixel area matches
        # expected ball area at its median depth
        best_score = -1.0
        best_idx = -1
        best_depth = 0.0

        for i in range(1, num_labels):  # skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self._min_blob_px:
                continue

            # Median depth of this component
            comp_mask = labels == i
            comp_depths = depth[comp_mask]
            median_d = float(np.median(comp_depths))

            # Expected ball area at this depth
            r_px = self._fx * self._ball_radius_m / median_d
            expected_area = np.pi * r_px ** 2

            # Score: ratio of smaller/larger area (1.0 = perfect match)
            ratio = min(area, expected_area) / max(area, expected_area)

            if ratio > best_score:
                best_score = ratio
                best_idx = i
                best_depth = median_d

        if best_idx < 0:
            return None

        # Step 4: deproject centroid
        cu, cv_pt = centroids[best_idx]
        x = (cu - cx) * best_depth / self._fx
        y = (cv_pt - cy) * best_depth / self._fy
        pos_cam = np.array([x, y, best_depth], dtype=np.float32)

        return SimDetection(
            pos_cam=pos_cam,
            pixel_uv=(float(cu), float(cv_pt)),
            depth_m=best_depth,
            num_pixels=int(stats[best_idx, cv2.CC_STAT_AREA]),
        )

    @classmethod
    def from_tiled_camera_cfg(
        cls,
        focal_length_cm: float = 11.24,
        horizontal_aperture_cm: float = 20.955,
        width: int = 640,
        height: int = 480,
        **kwargs,
    ) -> "SimBallDetector":
        """Create detector matching the D435i TiledCamera config.

        Converts focal_length (cm) and horizontal_aperture (cm) to pixel
        focal length using: fx_px = focal_length / horizontal_aperture * width
        """
        fx_px = focal_length_cm / horizontal_aperture_cm * width
        return cls(
            focal_length_px=fx_px,
            cx=width / 2.0,
            cy=height / 2.0,
            **kwargs,
        )
