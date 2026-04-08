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
