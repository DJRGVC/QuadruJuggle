"""Camera-to-world coordinate transforms for the perception pipeline.

Used by:
  - demo_camera_ekf.py (camera -> detect -> EKF demo)
  - Any future real-hardware pipeline that needs cam-frame -> world-frame

All quaternions are (w, x, y, z) convention.
"""

from __future__ import annotations

import numpy as np


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def cam_detection_to_world(
    pos_cam: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w_ros: np.ndarray,
) -> np.ndarray:
    """Transform a camera-frame detection to world frame.

    Args:
        pos_cam: (3,) position in ROS camera optical frame [x_right, y_down, z_fwd].
        cam_pos_w: (3,) camera world position from ``cam.data.pos_w``.
        cam_quat_w_ros: (4,) camera world quaternion (w,x,y,z) in ROS convention
            from ``cam.data.quat_w_ros``. Rotates from ROS camera frame to world.

    Returns:
        (3,) position in world frame.
    """
    R_cam_w = quat_to_rotmat(cam_quat_w_ros)
    return R_cam_w @ pos_cam + cam_pos_w
