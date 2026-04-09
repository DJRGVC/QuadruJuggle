"""Unit tests for perception/real/ utility methods.

Tests implemented (non-stub) methods:
  - CameraIntrinsics.deproject()
  - CameraExtrinsics.transform_to_body()
  - BallDetector._median_depth_in_bbox()
  - CameraCalibrator.from_known_mount()

No hardware required — pure numpy math.
"""

import math
import numpy as np
import sys
import os

# Import perception/real modules directly (bypassing Isaac Lab __init__ chain)
_PERCEPTION_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../source/go1_ball_balance/go1_ball_balance/perception",
    )
)

import importlib.util


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load camera first (detector depends on it)
_camera_mod = _load_module(
    "go1_ball_balance.perception.real.camera",
    os.path.join(_PERCEPTION_DIR, "real", "camera.py"),
)
CameraIntrinsics = _camera_mod.CameraIntrinsics

_calibration_mod = _load_module(
    "go1_ball_balance.perception.real.calibration",
    os.path.join(_PERCEPTION_DIR, "real", "calibration.py"),
)
CameraCalibrator = _calibration_mod.CameraCalibrator
CameraExtrinsics = _calibration_mod.CameraExtrinsics

_detector_mod = _load_module(
    "go1_ball_balance.perception.real.detector",
    os.path.join(_PERCEPTION_DIR, "real", "detector.py"),
)
BallDetector = _detector_mod.BallDetector


# ── CameraIntrinsics.deproject ──────────────────────────────────────────

def test_deproject_principal_point():
    """Point at principal point projects to (0, 0, depth)."""
    intr = CameraIntrinsics(fx=420.0, fy=420.0, cx=424.0, cy=240.0,
                            width=848, height=480)
    pt = intr.deproject(424.0, 240.0, 0.5)
    np.testing.assert_allclose(pt, [0.0, 0.0, 0.5], atol=1e-7)


def test_deproject_off_centre():
    """Known offset from principal point."""
    intr = CameraIntrinsics(fx=400.0, fy=400.0, cx=400.0, cy=200.0,
                            width=800, height=400)
    # u=500 → (500-400)*0.3/400 = 0.075
    # v=300 → (300-200)*0.3/400 = 0.075
    pt = intr.deproject(500.0, 300.0, 0.3)
    np.testing.assert_allclose(pt, [0.075, 0.075, 0.3], atol=1e-7)


def test_deproject_zero_depth():
    """Depth=0 should give origin."""
    intr = CameraIntrinsics(fx=420.0, fy=420.0, cx=424.0, cy=240.0,
                            width=848, height=480)
    pt = intr.deproject(500.0, 300.0, 0.0)
    np.testing.assert_allclose(pt, [0.0, 0.0, 0.0], atol=1e-7)


# ── CameraExtrinsics.transform_to_body ─────────────────────────────────

def test_transform_identity():
    """Identity rotation + zero translation = passthrough."""
    ext = CameraExtrinsics(
        R_cam_body=np.eye(3),
        t_cam_body=np.zeros(3),
    )
    pt_cam = np.array([1.0, 2.0, 3.0])
    pt_body = ext.transform_to_body(pt_cam)
    np.testing.assert_allclose(pt_body, pt_cam, atol=1e-7)


def test_transform_translation_only():
    """Pure translation, no rotation."""
    ext = CameraExtrinsics(
        R_cam_body=np.eye(3),
        t_cam_body=np.array([0.1, -0.05, 0.07]),
    )
    pt_cam = np.array([0.0, 0.0, 0.5])
    pt_body = ext.transform_to_body(pt_cam)
    np.testing.assert_allclose(pt_body, [0.1, -0.05, 0.57], atol=1e-7)


def test_transform_90deg_rotation():
    """90-degree rotation about X axis (Y->Z, Z->-Y)."""
    # Rx(90°): [[1,0,0],[0,0,-1],[0,1,0]]
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    ext = CameraExtrinsics(R_cam_body=R, t_cam_body=np.zeros(3))
    pt_cam = np.array([0.0, 1.0, 0.0])
    pt_body = ext.transform_to_body(pt_cam)
    np.testing.assert_allclose(pt_body, [0.0, 0.0, 1.0], atol=1e-7)


def test_transform_batch():
    """Batch mode (N, 3) input."""
    ext = CameraExtrinsics(
        R_cam_body=np.eye(3),
        t_cam_body=np.array([1.0, 0.0, 0.0]),
    )
    pts_cam = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    pts_body = ext.transform_to_body(pts_cam)
    expected = np.array([[1.0, 0.0, 0.0], [2.0, 2.0, 3.0]])
    np.testing.assert_allclose(pts_body, expected, atol=1e-7)


def test_transform_roundtrip():
    """Applying R then R^T should recover original point (zero translation)."""
    # Random rotation via Rodrigues
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.7
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

    ext_fwd = CameraExtrinsics(R_cam_body=R, t_cam_body=np.zeros(3))
    ext_inv = CameraExtrinsics(R_cam_body=R.T, t_cam_body=np.zeros(3))

    pt = np.array([0.3, -0.1, 0.5])
    recovered = ext_inv.transform_to_body(ext_fwd.transform_to_body(pt))
    np.testing.assert_allclose(recovered, pt, atol=1e-10)


# ── BallDetector._median_depth_in_bbox ──────────────────────────────────

def test_median_depth_simple():
    """Median of uniform valid depth region."""
    depth = np.zeros((480, 848), dtype=np.uint16)
    # Fill a 10x10 box with depths 200-209 mm
    depth[100:110, 200:210] = np.arange(200, 300).reshape(10, 10).astype(np.uint16)
    result = BallDetector._median_depth_in_bbox(depth, (200, 100, 210, 110))
    assert result is not None
    # Median of 200..299 = 249.5 mm -> 0.2495 m
    np.testing.assert_allclose(result, 0.2495, atol=1e-4)


def test_median_depth_rejects_zeros():
    """Zero-depth pixels are filtered out."""
    depth = np.zeros((100, 100), dtype=np.uint16)
    depth[10:20, 10:20] = 0  # all zeros in bbox
    depth[10, 10] = 500  # one valid pixel
    result = BallDetector._median_depth_in_bbox(depth, (10, 10, 20, 20))
    assert result is not None
    np.testing.assert_allclose(result, 0.5, atol=1e-6)


def test_median_depth_all_invalid():
    """All zeros/out-of-range → None."""
    depth = np.zeros((100, 100), dtype=np.uint16)
    result = BallDetector._median_depth_in_bbox(depth, (10, 10, 20, 20))
    assert result is None


def test_median_depth_rejects_out_of_range():
    """Pixels beyond max_depth_mm are filtered."""
    depth = np.full((100, 100), 3000, dtype=np.uint16)  # 3m > default 2m max
    result = BallDetector._median_depth_in_bbox(depth, (10, 10, 20, 20))
    assert result is None


def test_median_depth_mixed():
    """Mix of valid, zero, and out-of-range pixels — only valid contribute."""
    depth = np.zeros((100, 100), dtype=np.uint16)
    # 5 pixels: 0(invalid), 100(below min), 300(valid), 400(valid), 5000(above max)
    depth[50, 50] = 0
    depth[50, 51] = 100  # below default min (168mm)
    depth[50, 52] = 300
    depth[50, 53] = 400
    depth[50, 54] = 5000  # above max
    result = BallDetector._median_depth_in_bbox(depth, (50, 50, 55, 51))
    assert result is not None
    # Valid: 300, 400 → median = 350mm = 0.35m
    np.testing.assert_allclose(result, 0.35, atol=1e-6)


# ── CameraCalibrator.from_known_mount ───────────────────────────────────

def test_from_known_mount_identity():
    """Zero RPY + zero position = identity transform."""
    ext = CameraCalibrator.from_known_mount(
        mount_position_body=np.zeros(3),
        mount_orientation_rpy=np.zeros(3),
    )
    np.testing.assert_allclose(ext.R_cam_body, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(ext.t_cam_body, np.zeros(3), atol=1e-10)


def test_from_known_mount_translation():
    """Non-zero mount position appears in translation."""
    pos = np.array([0.0, -0.05, 0.07])
    ext = CameraCalibrator.from_known_mount(
        mount_position_body=pos,
        mount_orientation_rpy=np.zeros(3),
    )
    np.testing.assert_allclose(ext.t_cam_body, pos, atol=1e-10)


def test_from_known_mount_pitch_45():
    """45-degree pitch (camera tilted 45° up). Check that a Z=1 camera
    point maps to both +X and +Z in body frame."""
    ext = CameraCalibrator.from_known_mount(
        mount_position_body=np.zeros(3),
        mount_orientation_rpy=np.array([0.0, np.pi / 4, 0.0]),
    )
    pt_cam = np.array([0.0, 0.0, 1.0])
    pt_body = ext.transform_to_body(pt_cam)
    # With pitch=45°, camera Z maps to body X*cos45 + Z*sin45
    np.testing.assert_allclose(
        pt_body,
        [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4)],
        atol=1e-10,
    )


def test_from_known_mount_rotation_is_orthogonal():
    """Rotation matrix from arbitrary RPY must be orthogonal."""
    ext = CameraCalibrator.from_known_mount(
        mount_position_body=np.array([0.1, -0.03, 0.05]),
        mount_orientation_rpy=np.array([0.3, -0.7, 1.2]),
    )
    R = ext.R_cam_body
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


# ── _rotation_between_vectors ─────────────────────────────────────────

_rotation_between_vectors = _calibration_mod._rotation_between_vectors


def test_rotation_between_vectors_identity():
    """Rotating a vector to itself gives identity."""
    v = np.array([0.0, 0.0, 1.0])
    R = _rotation_between_vectors(v, v)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_rotation_between_vectors_orthogonal():
    """Rotating X to Y gives 90-degree rotation about Z."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    R = _rotation_between_vectors(a, b)
    result = R @ a
    np.testing.assert_allclose(result, b, atol=1e-10)
    # Must be proper rotation
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_rotation_between_vectors_antiparallel():
    """Rotating a vector to its opposite gives 180-degree rotation."""
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([0.0, 0.0, -1.0])
    R = _rotation_between_vectors(a, b)
    result = R @ a
    np.testing.assert_allclose(result, b, atol=1e-10)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_rotation_between_vectors_arbitrary():
    """Rotating between arbitrary unit vectors maps correctly."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        a = rng.standard_normal(3)
        a /= np.linalg.norm(a)
        b = rng.standard_normal(3)
        b /= np.linalg.norm(b)
        R = _rotation_between_vectors(a, b)
        np.testing.assert_allclose(R @ a, b, atol=1e-10)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


# ── CameraCalibrator.from_checkerboard ─────────────────────────────────


class _MockCameraForCheckerboard:
    """Mock camera that returns depth frames for checkerboard calibration."""

    def __init__(self, intrinsics, frames):
        self._intrinsics = intrinsics
        self._frames = list(frames)
        self._idx = 0

    def get_intrinsics(self):
        return self._intrinsics

    def get_frame(self):
        if self._idx >= len(self._frames):
            return None
        frame = self._frames[self._idx]
        self._idx += 1
        return frame


def test_from_checkerboard_gravity_aligned():
    """Mock cv2 functions to test the gravity-alignment logic.

    Setup: board Z in camera frame = [0, 0, 1] (identity PnP).
    gravity_cam = -board_z = [0, 0, -1].
    gravity_body = [0, 0, -1] (default).
    Expected: R_cam_body ≈ identity.
    """
    import unittest.mock as mock
    import cv2 as _cv2

    intrinsics = CameraIntrinsics(
        fx=425.0, fy=425.0, cx=424.0, cy=240.0, width=848, height=480
    )

    # Create depth frames with valid depth (500mm uniform)
    depth = np.full((480, 848), 500, dtype=np.uint16)
    frames = [(depth.copy(), float(i)) for i in range(5)]
    camera = _MockCameraForCheckerboard(intrinsics, frames)

    board_size = (7, 5)
    n_corners = board_size[0] * board_size[1]

    # Fake corners at plausible pixel locations
    fake_corners = np.zeros((n_corners, 1, 2), dtype=np.float32)
    for i in range(n_corners):
        row, col = divmod(i, board_size[0])
        fake_corners[i, 0] = [200 + col * 20, 100 + row * 20]

    # Identity rvec (no rotation)
    rvec_identity = np.zeros(3, dtype=np.float64)
    tvec_half_m = np.array([0.0, 0.0, 0.5], dtype=np.float64)

    with mock.patch.object(_calibration_mod.cv2, 'findChessboardCorners',
                           return_value=(True, fake_corners)):
        with mock.patch.object(_calibration_mod.cv2, 'cornerSubPix',
                               return_value=fake_corners):
            with mock.patch.object(_calibration_mod.cv2, 'solvePnP',
                                   return_value=(True, rvec_identity.reshape(3, 1),
                                                 tvec_half_m.reshape(3, 1))):
                with mock.patch.object(_calibration_mod.cv2, 'CALIB_CB_ADAPTIVE_THRESH', 1):
                    with mock.patch.object(_calibration_mod.cv2, 'CALIB_CB_NORMALIZE_IMAGE', 2):
                        with mock.patch.object(_calibration_mod.cv2, 'SOLVEPNP_IPPE', 6):
                            ext = CameraCalibrator.from_checkerboard(
                                camera,
                                board_size=board_size,
                                square_size_m=0.025,
                                num_frames=3,
                            )

    # R_cam_body should be identity (gravity already aligned)
    np.testing.assert_allclose(ext.R_cam_body, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(ext.t_cam_body, np.zeros(3), atol=1e-6)


def test_from_checkerboard_rotated_board():
    """Board rotated 90° around X: board Z in cam = [0, -1, 0].

    gravity_cam = -board_z = [0, 1, 0].
    gravity_body = [0, 0, -1].
    R_cam_body should map [0, 1, 0] → [0, 0, -1].
    """
    import unittest.mock as mock
    import cv2 as _cv2

    intrinsics = CameraIntrinsics(
        fx=425.0, fy=425.0, cx=424.0, cy=240.0, width=848, height=480
    )
    depth = np.full((480, 848), 500, dtype=np.uint16)
    frames = [(depth.copy(), float(i)) for i in range(5)]
    camera = _MockCameraForCheckerboard(intrinsics, frames)

    board_size = (7, 5)
    n_corners = board_size[0] * board_size[1]
    fake_corners = np.zeros((n_corners, 1, 2), dtype=np.float32)
    for i in range(n_corners):
        row, col = divmod(i, board_size[0])
        fake_corners[i, 0] = [200 + col * 20, 100 + row * 20]

    # 90° rotation around X axis: R_x(pi/2) → board Z maps to [0, -1, 0] in camera
    rvec_90x = np.array([np.pi / 2, 0, 0], dtype=np.float64)
    tvec = np.array([0.0, 0.0, 0.5], dtype=np.float64)

    with mock.patch.object(_calibration_mod.cv2, 'findChessboardCorners',
                           return_value=(True, fake_corners)):
        with mock.patch.object(_calibration_mod.cv2, 'cornerSubPix',
                               return_value=fake_corners):
            with mock.patch.object(_calibration_mod.cv2, 'solvePnP',
                                   return_value=(True, rvec_90x.reshape(3, 1),
                                                 tvec.reshape(3, 1))):
                with mock.patch.object(_calibration_mod.cv2, 'CALIB_CB_ADAPTIVE_THRESH', 1):
                    with mock.patch.object(_calibration_mod.cv2, 'CALIB_CB_NORMALIZE_IMAGE', 2):
                        with mock.patch.object(_calibration_mod.cv2, 'SOLVEPNP_IPPE', 6):
                            ext = CameraCalibrator.from_checkerboard(
                                camera,
                                board_size=board_size,
                                square_size_m=0.025,
                                num_frames=3,
                            )

    # Verify: R_cam_body maps gravity_cam to gravity_body
    R_board_cam, _ = _cv2.Rodrigues(rvec_90x)
    board_z_cam = R_board_cam[:, 2]
    gravity_cam = -board_z_cam
    gravity_body = np.array([0.0, 0.0, -1.0])

    result = ext.R_cam_body @ gravity_cam
    np.testing.assert_allclose(result, gravity_body, atol=1e-6)
    np.testing.assert_allclose(ext.R_cam_body @ ext.R_cam_body.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(ext.R_cam_body), 1.0, atol=1e-6)


def test_from_checkerboard_too_few_frames_raises():
    """If camera produces no valid frames, should raise RuntimeError."""
    import unittest.mock as mock

    intrinsics = CameraIntrinsics(
        fx=425.0, fy=425.0, cx=424.0, cy=240.0, width=848, height=480
    )
    depth = np.full((480, 848), 500, dtype=np.uint16)
    frames = [(depth.copy(), 0.0)] * 5
    camera = _MockCameraForCheckerboard(intrinsics, frames)

    # findChessboardCorners always fails
    with mock.patch.object(_calibration_mod.cv2, 'findChessboardCorners',
                           return_value=(False, None)):
        with mock.patch.object(_calibration_mod.cv2, 'CALIB_CB_ADAPTIVE_THRESH', 1):
            with mock.patch.object(_calibration_mod.cv2, 'CALIB_CB_NORMALIZE_IMAGE', 2):
                try:
                    CameraCalibrator.from_checkerboard(camera, num_frames=3)
                    assert False, "Expected RuntimeError"
                except RuntimeError as e:
                    assert "valid checkerboard frames" in str(e)


def test_from_checkerboard_no_opencv_raises():
    """If cv2 is not available, should raise ImportError."""
    original_cv2 = _calibration_mod.cv2
    _calibration_mod.cv2 = None
    try:
        intrinsics = CameraIntrinsics(
            fx=425.0, fy=425.0, cx=424.0, cy=240.0, width=848, height=480
        )
        camera = _MockCameraForCheckerboard(intrinsics, [])
        try:
            CameraCalibrator.from_checkerboard(camera, num_frames=3)
            assert False, "Expected ImportError"
        except ImportError as e:
            assert "OpenCV" in str(e)
    finally:
        _calibration_mod.cv2 = original_cv2


def test_from_checkerboard_custom_gravity():
    """Custom gravity_body vector is respected."""
    import unittest.mock as mock

    intrinsics = CameraIntrinsics(
        fx=425.0, fy=425.0, cx=424.0, cy=240.0, width=848, height=480
    )
    depth = np.full((480, 848), 500, dtype=np.uint16)
    frames = [(depth.copy(), float(i)) for i in range(5)]
    camera = _MockCameraForCheckerboard(intrinsics, frames)

    board_size = (7, 5)
    n_corners = board_size[0] * board_size[1]
    fake_corners = np.zeros((n_corners, 1, 2), dtype=np.float32)
    for i in range(n_corners):
        row, col = divmod(i, board_size[0])
        fake_corners[i, 0] = [200 + col * 20, 100 + row * 20]

    rvec_identity = np.zeros(3, dtype=np.float64)
    tvec = np.array([0.0, 0.0, 0.5], dtype=np.float64)

    # Custom gravity: tilted robot
    gravity_body = np.array([0.1, 0.0, -0.995])
    gravity_body /= np.linalg.norm(gravity_body)

    with mock.patch.object(_calibration_mod.cv2, 'findChessboardCorners',
                           return_value=(True, fake_corners)):
        with mock.patch.object(_calibration_mod.cv2, 'cornerSubPix',
                               return_value=fake_corners):
            with mock.patch.object(_calibration_mod.cv2, 'solvePnP',
                                   return_value=(True, rvec_identity.reshape(3, 1),
                                                 tvec.reshape(3, 1))):
                with mock.patch.object(_calibration_mod.cv2, 'CALIB_CB_ADAPTIVE_THRESH', 1):
                    with mock.patch.object(_calibration_mod.cv2, 'CALIB_CB_NORMALIZE_IMAGE', 2):
                        with mock.patch.object(_calibration_mod.cv2, 'SOLVEPNP_IPPE', 6):
                            ext = CameraCalibrator.from_checkerboard(
                                camera,
                                board_size=board_size,
                                square_size_m=0.025,
                                num_frames=3,
                                gravity_body=gravity_body,
                            )

    # Verify gravity alignment
    gravity_cam = np.array([0.0, 0.0, -1.0])  # identity rvec → board_z=[0,0,1] → grav=-[0,0,1]
    result = ext.R_cam_body @ gravity_cam
    np.testing.assert_allclose(result, gravity_body, atol=1e-6)
    np.testing.assert_allclose(ext.R_cam_body @ ext.R_cam_body.T, np.eye(3), atol=1e-6)


# ── Runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
