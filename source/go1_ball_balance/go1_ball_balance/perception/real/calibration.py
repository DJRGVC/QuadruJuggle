"""Camera-to-body extrinsic calibration.

Provides the rigid transform from camera optical frame to robot body (IMU)
frame. This transform is used to convert ball detections from camera frame
to world frame (via body-to-world from IMU).

Two loading modes:
1. from_yaml(): load pre-calibrated extrinsics (production)
2. from_checkerboard(): run interactive calibration routine (setup)

See docs/hardware_pipeline_architecture.md §3.3 for specification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraExtrinsics:
    """Camera-to-body (IMU) rigid transform.

    Convention: transforms points FROM camera optical frame TO body frame.
    Camera optical frame: Z forward, X right, Y down.
    Body frame: X forward, Y left, Z up (REP-103 / Isaac Lab convention).
    """

    R_cam_body: np.ndarray  # (3, 3) rotation: camera -> body
    t_cam_body: np.ndarray  # (3,) translation: camera origin in body frame (metres)

    def transform_to_body(self, pos_cam: np.ndarray) -> np.ndarray:
        """Transform a point from camera frame to body frame.

        Args:
            pos_cam: (3,) or (N, 3) point(s) in camera optical frame.

        Returns:
            (3,) or (N, 3) point(s) in body frame.
        """
        if pos_cam.ndim == 1:
            return self.R_cam_body @ pos_cam + self.t_cam_body
        return (pos_cam @ self.R_cam_body.T) + self.t_cam_body


class CameraCalibrator:
    """Compute and store camera-to-body extrinsics."""

    @staticmethod
    def from_yaml(path: str) -> CameraExtrinsics:
        """Load pre-calibrated extrinsics from YAML file.

        Expected format::

            R_cam_body: [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
            t_cam_body: [tx, ty, tz]

        Args:
            path: Path to YAML file.

        Returns:
            CameraExtrinsics with loaded transform.
        """
        raise NotImplementedError(
            "CameraCalibrator.from_yaml() is a stub — implement when "
            "extrinsics YAML format is finalised. "
            "See docs/hardware_pipeline_architecture.md §3.3."
        )

    @staticmethod
    def from_checkerboard(
        camera,  # D435iCamera instance
        board_size: tuple[int, int] = (7, 5),
        square_size_m: float = 0.025,
        num_frames: int = 20,
    ) -> CameraExtrinsics:
        """Interactive checkerboard calibration routine.

        Guides user to hold a checkerboard at various poses relative to
        the robot body. Computes camera-to-body transform via PnP + IMU
        gravity alignment.

        Args:
            camera: Started D435iCamera instance.
            board_size: Inner corners (cols, rows).
            square_size_m: Checkerboard square side length.
            num_frames: Number of frames to capture.

        Returns:
            CameraExtrinsics from the calibration.
        """
        raise NotImplementedError(
            "CameraCalibrator.from_checkerboard() is a stub — implement "
            "during hardware setup phase."
        )

    @staticmethod
    def from_known_mount(
        mount_position_body: np.ndarray,
        mount_orientation_rpy: np.ndarray,
    ) -> CameraExtrinsics:
        """Compute extrinsics from known mount geometry.

        For the QuadruJuggle setup, the D435i is mounted behind the paddle,
        pointing ~45 degrees upward. If the mount is rigid and measured,
        this avoids the checkerboard routine.

        Args:
            mount_position_body: (3,) camera origin in body frame (metres).
            mount_orientation_rpy: (3,) roll, pitch, yaw in radians
                (camera-to-body rotation as Euler angles).

        Returns:
            CameraExtrinsics with the computed transform.
        """
        raise NotImplementedError(
            "CameraCalibrator.from_known_mount() is a stub."
        )
