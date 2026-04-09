"""Camera-to-body extrinsic calibration.

Provides the rigid transform from camera optical frame to robot body (IMU)
frame. This transform is used to convert ball detections from camera frame
to world frame (via body-to-world from IMU).

Three loading modes:
1. from_yaml(): load pre-calibrated extrinsics (production)
2. from_checkerboard(): run interactive calibration routine (setup)
3. from_known_mount(): compute from measured mount geometry (no hardware needed)

See docs/hardware_pipeline_architecture.md §3.3 for specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


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
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        R = np.array(data["R_cam_body"], dtype=np.float64)
        t = np.array(data["t_cam_body"], dtype=np.float64)

        if R.shape != (3, 3):
            raise ValueError(f"R_cam_body must be 3x3, got {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"t_cam_body must be (3,), got {t.shape}")

        return CameraExtrinsics(R_cam_body=R, t_cam_body=t)

    @staticmethod
    def from_checkerboard(
        camera,  # D435iCamera instance
        board_size: tuple[int, int] = (7, 5),
        square_size_m: float = 0.025,
        num_frames: int = 20,
        gravity_body: np.ndarray | None = None,
    ) -> CameraExtrinsics:
        """Checkerboard calibration: camera-to-body via PnP + gravity.

        The robot stands still on flat ground. A checkerboard is held above
        the camera at various poses. For each detected frame, solvePnP gives
        the camera-to-board rotation. The gravity vector (from IMU projected
        gravity, or assumed [0, 0, -1] in body frame if robot is upright)
        resolves the camera-to-body alignment.

        Algorithm:
        1. Capture ``num_frames`` depth frames with detected checkerboard corners.
        2. For each: solvePnP → R_board_cam (board-to-camera rotation).
        3. Camera gravity = average board Z axis direction across frames
           (board lies flat → its Z ≈ world Z ≈ -body Z for upward camera).
        4. Align camera gravity with body gravity via Wahba's single-vector
           solution (cross product + skew-symmetric).
        5. Translation = camera offset in body frame from mean board distance.

        Requires OpenCV and a started D435iCamera instance.

        Args:
            camera: Started D435iCamera with valid intrinsics.
            board_size: Inner corners (cols, rows) of the checkerboard.
            square_size_m: Physical side length of each square in metres.
            num_frames: Minimum number of valid frames to collect.
            gravity_body: (3,) gravity direction in body frame. Defaults to
                [0, 0, -1] (robot upright on flat ground, body Z = up).

        Returns:
            CameraExtrinsics from the calibration.

        Raises:
            ImportError: If OpenCV is not available.
            RuntimeError: If fewer than 3 valid frames are captured.
        """
        if cv2 is None:
            raise ImportError("OpenCV (cv2) required for checkerboard calibration.")

        if gravity_body is None:
            gravity_body = np.array([0.0, 0.0, -1.0])
        gravity_body = gravity_body / np.linalg.norm(gravity_body)

        intrinsics = camera.get_intrinsics()
        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros(5)  # D435i depth has no distortion model

        # Build 3D object points for the checkerboard (Z=0 plane)
        obj_points = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
        obj_points[:, :2] = np.mgrid[
            0:board_size[0], 0:board_size[1]
        ].T.reshape(-1, 2) * square_size_m

        # Collect frames with detected corners
        rvecs: list[np.ndarray] = []
        tvecs: list[np.ndarray] = []
        attempts = 0
        max_attempts = num_frames * 10  # don't loop forever

        while len(rvecs) < num_frames and attempts < max_attempts:
            attempts += 1
            result = camera.get_frame()
            if result is None:
                continue

            depth_u16, _timestamp = result

            # Convert depth to 8-bit grayscale for corner detection
            # Map valid range [168mm, 2000mm] to [0, 255]
            depth_f = depth_u16.astype(np.float32)
            valid = (depth_u16 >= 168) & (depth_u16 <= 2000)
            gray = np.zeros(depth_u16.shape, dtype=np.uint8)
            if np.any(valid):
                gray[valid] = np.clip(
                    (depth_f[valid] - 168.0) * 255.0 / (2000.0 - 168.0),
                    0, 255,
                ).astype(np.uint8)

            found, corners = cv2.findChessboardCorners(
                gray, board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if not found:
                continue

            # Refine corner locations
            corners = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )

            # solvePnP: board frame → camera frame
            success, rvec, tvec = cv2.solvePnP(
                obj_points, corners, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if success:
                rvecs.append(rvec.flatten())
                tvecs.append(tvec.flatten())
                logger.info(
                    "Checkerboard frame %d/%d captured (attempt %d)",
                    len(rvecs), num_frames, attempts,
                )

        if len(rvecs) < 3:
            raise RuntimeError(
                f"Only {len(rvecs)} valid checkerboard frames captured "
                f"(need at least 3). Check lighting and board visibility."
            )

        # Compute average gravity direction in camera frame.
        # Each PnP gives R_board_cam: transforms board frame to camera frame.
        # Board Z axis in camera frame = R_board_cam[:, 2].
        # If board is held roughly horizontal, board Z ≈ world Z direction.
        gravity_cam_samples = []
        for rvec in rvecs:
            R_board_cam, _ = cv2.Rodrigues(rvec)
            # Board Z axis in camera frame (points away from board surface)
            board_z_cam = R_board_cam[:, 2]
            # Board is above camera pointing down → board Z points away from
            # camera. Gravity in camera frame is opposite to board Z.
            gravity_cam_samples.append(-board_z_cam)

        gravity_cam = np.mean(gravity_cam_samples, axis=0)
        gravity_cam = gravity_cam / np.linalg.norm(gravity_cam)

        # Compute R_cam_body: rotation that maps camera frame to body frame.
        # We need R_cam_body such that R_cam_body @ gravity_cam = gravity_body.
        # This is the single-vector Wahba problem — solved via axis-angle.
        R_cam_body = _rotation_between_vectors(gravity_cam, gravity_body)

        # Translation: mean board position in camera frame, transformed to body.
        # The camera-to-body translation is the camera origin expressed in body frame.
        # From known mount geometry this is more reliable, but we estimate from
        # the average tvec (board origin in camera frame) as a sanity check.
        # For production use, prefer from_known_mount() or manual measurement.
        mean_tvec_cam = np.mean(tvecs, axis=0)
        # This gives board-centre in camera frame; not directly useful for
        # camera-in-body translation. Use zero (user should measure and override).
        t_cam_body = np.zeros(3, dtype=np.float64)
        logger.info(
            "Checkerboard calibration complete: %d frames, mean board distance %.3fm. "
            "Translation set to zero — measure camera mount offset for production use.",
            len(rvecs), np.linalg.norm(mean_tvec_cam),
        )

        return CameraExtrinsics(R_cam_body=R_cam_body, t_cam_body=t_cam_body)

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
                (camera-to-body rotation as Euler angles, intrinsic XYZ).

        Returns:
            CameraExtrinsics with the computed transform.
        """
        roll, pitch, yaw = mount_orientation_rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Intrinsic XYZ rotation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr],
        ])

        return CameraExtrinsics(
            R_cam_body=R,
            t_cam_body=np.asarray(mount_position_body, dtype=np.float64),
        )


def _rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute rotation matrix R such that R @ a = b.

    Uses Rodrigues' rotation formula via the cross product.
    Handles the degenerate case where a and b are (anti-)parallel.

    Args:
        a: (3,) unit vector (source direction).
        b: (3,) unit vector (target direction).

    Returns:
        (3, 3) rotation matrix.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-8:
        # Vectors are (anti-)parallel
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # 180-degree rotation: find an orthogonal axis
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis = axis / np.linalg.norm(axis)
        # R = 2 * outer(axis, axis) - I  (180-degree rotation about axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    # Skew-symmetric cross-product matrix of v
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ], dtype=np.float64)

    # Rodrigues' formula: R = I + vx + vx² * (1 - c) / s²
    R = np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s))
    return R
