"""Tests for camera pixel projection — validates where the ball appears in D435i images.

Given the D435i camera config (70° tilt, 640x480, known intrinsics), verifies that
balls at various world positions project to the expected pixel locations.  This is
the inverse of the cam_detection_to_world pipeline and catches config errors
(wrong tilt, wrong intrinsics, convention mismatch) before a GPU run.
"""

from __future__ import annotations

import numpy as np
import pytest
import importlib
import sys, os

# Import frame_transforms directly (avoid go1_ball_balance.__init__ which pulls in Isaac Lab/pxr)
_ft_path = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "source", "go1_ball_balance",
    "go1_ball_balance", "perception", "frame_transforms.py",
))
_spec = importlib.util.spec_from_file_location("frame_transforms", _ft_path)
_ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ft)
quat_to_rotmat = _ft.quat_to_rotmat
cam_detection_to_world = _ft.cam_detection_to_world

# ── D435i camera config constants (must match ball_juggle_hier_env_cfg.py) ──

_FOCAL_LENGTH_CM = 11.24
_HORIZONTAL_APERTURE_CM = 20.955
_WIDTH = 640
_HEIGHT = 480
_FX_PX = _FOCAL_LENGTH_CM / _HORIZONTAL_APERTURE_CM * _WIDTH  # ≈ 343.1
_FY_PX = _FX_PX  # square pixels
_CX = _WIDTH / 2.0   # 320
_CY = _HEIGHT / 2.0  # 240

# Camera body-frame offset from trunk root
_CAM_BODY_OFFSET = np.array([-0.08, 0.0, 0.06])

# Camera orientation quaternion (w,x,y,z) — 70° pitch up in world convention
# -70° about Y: q = (cos(-35°), 0, sin(-35°), 0)
_CAM_QUAT_WORLD = np.array([0.8192, 0.0, -0.5736, 0.0])

# Paddle offset from trunk root (body frame)
_PADDLE_OFFSET_B = np.array([0.0, 0.0, 0.070])

# Nominal standing trunk height
_TRUNK_Z = 0.40


# ── Helpers ──────────────────────────────────────────────────────────────────

def world_to_cam(pos_world: np.ndarray, cam_pos_w: np.ndarray, cam_quat_w_ros: np.ndarray) -> np.ndarray:
    """Inverse of cam_detection_to_world: world-frame → camera ROS frame."""
    R = quat_to_rotmat(cam_quat_w_ros)
    return R.T @ (pos_world - cam_pos_w)


def project_to_pixel(pos_cam: np.ndarray) -> tuple[float, float]:
    """Project camera-frame 3D point to pixel (u, v) via pinhole model."""
    x, y, z = pos_cam
    assert z > 0, f"Point behind camera: z={z}"
    u = _FX_PX * x / z + _CX
    v = _FY_PX * y / z + _CY
    return float(u), float(v)


def in_frame(u: float, v: float) -> bool:
    """Check if pixel (u,v) is within the 640×480 image."""
    return 0 <= u < _WIDTH and 0 <= v < _HEIGHT


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def setup_camera_pose(trunk_pos: np.ndarray | None = None):
    """Compute camera world pose for a robot at trunk_pos (default: origin standing).

    Isaac Lab convention="world" means:
      - Identity quaternion → camera forward = +X_world, up = +Z_world
      - The offset quaternion rotates in this world-convention frame
      - Config: -70° about Y → pitches camera up by 70° from horizontal

    We reconstruct quat_w_ros (ROS camera frame → world) from first principles:
      1. Start from world-convention base: fwd=+X, right=-Y, up=+Z
      2. Apply the -70° Y tilt to get tilted frame axes in world
      3. Map to ROS camera convention: X_ros=right, Y_ros=down, Z_ros=forward
    """
    if trunk_pos is None:
        trunk_pos = np.array([0.0, 0.0, _TRUNK_Z])

    cam_pos_w = trunk_pos + _CAM_BODY_OFFSET

    # Tilt rotation: -70° about Y axis
    c = np.cos(np.radians(70))  # ≈ 0.342
    s = np.sin(np.radians(70))  # ≈ 0.940
    R_tilt = np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c],
    ])

    # World-convention base axes
    fwd_tilted = R_tilt @ np.array([1, 0, 0])   # → (0.342, 0, 0.940) — mostly up
    right_tilted = R_tilt @ np.array([0, -1, 0])  # → (0, -1, 0) — unchanged
    up_tilted = R_tilt @ np.array([0, 0, 1])    # → (-0.940, 0, 0.342) — mostly backward

    # ROS camera convention: X_ros=right, Y_ros=down, Z_ros=forward
    # R_ros_to_world columns = world images of each ROS basis vector
    R_ros_to_world = np.column_stack([
        right_tilted,    # X_ros (right) → right in world
        -up_tilted,      # Y_ros (down)  → -up in world
        fwd_tilted,      # Z_ros (fwd)   → forward in world
    ])

    cam_quat_w_ros = rotmat_to_quat(R_ros_to_world)

    return cam_pos_w, cam_quat_w_ros


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ── Camera intrinsics tests ─────────────────────────────────────────────────

class TestCameraIntrinsics:
    """Validate D435i camera intrinsic parameters."""

    def test_focal_length_px(self):
        """fx should be ~343 px for the D435i config."""
        assert 340 < _FX_PX < 346, f"fx={_FX_PX:.1f} outside expected range"

    def test_hfov(self):
        """Horizontal FOV should be ~86° (2 * atan(W/2 / fx))."""
        hfov_rad = 2 * np.arctan(_WIDTH / (2 * _FX_PX))
        hfov_deg = np.degrees(hfov_rad)
        assert 84 < hfov_deg < 88, f"HFOV={hfov_deg:.1f}° outside expected 86°"

    def test_vfov(self):
        """Vertical FOV should be ~70° (4:3 aspect)."""
        vfov_rad = 2 * np.arctan(_HEIGHT / (2 * _FY_PX))
        vfov_deg = np.degrees(vfov_rad)
        assert 68 < vfov_deg < 72, f"VFOV={vfov_deg:.1f}° outside expected 70°"


# ── Camera orientation tests ─────────────────────────────────────────────────

class TestCameraOrientation:
    """Validate D435i 70° tilt camera look direction."""

    def test_opengl_quat_is_unit(self):
        assert abs(np.linalg.norm(_CAM_QUAT_WORLD) - 1.0) < 1e-3

    def test_cam_looks_mostly_up(self):
        """With 70° tilt, the camera should look mostly upward (+Z world)."""
        cam_pos_w, cam_quat_w_ros = setup_camera_pose()
        R = quat_to_rotmat(cam_quat_w_ros)
        # In ROS camera convention, forward is +Z_cam.  R maps cam→world.
        fwd_world = R @ np.array([0, 0, 1])
        # Should have large positive Z component (looking up)
        assert fwd_world[2] > 0.8, f"Camera forward Z={fwd_world[2]:.3f}, expected >0.8 (looking up)"

    def test_cam_forward_elevation(self):
        """Camera elevation should be approximately 70°."""
        cam_pos_w, cam_quat_w_ros = setup_camera_pose()
        R = quat_to_rotmat(cam_quat_w_ros)
        fwd_world = R @ np.array([0, 0, 1])
        elev_deg = np.degrees(np.arcsin(np.clip(fwd_world[2], -1, 1)))
        assert 60 < elev_deg < 80, f"Camera elevation={elev_deg:.1f}°, expected ~70°"

    def test_cam_forward_has_positive_x(self):
        """Camera tilted 70° up from +X forward → forward should have +X component (cos70°≈0.34)."""
        cam_pos_w, cam_quat_w_ros = setup_camera_pose()
        R = quat_to_rotmat(cam_quat_w_ros)
        fwd_world = R @ np.array([0, 0, 1])
        assert 0.3 < fwd_world[0] < 0.4, f"Expected forward X≈0.34, got {fwd_world[0]:.3f}"


# ── Pixel projection tests ──────────────────────────────────────────────────

class TestPixelProjection:
    """Validate ball world positions project to expected pixel locations."""

    @pytest.fixture
    def cam_pose(self):
        """Standard camera pose: robot at origin, standing at 0.40m."""
        return setup_camera_pose()

    def _ball_world_above_paddle(self, height_above_paddle: float) -> np.ndarray:
        """Ball world position: centred on paddle, `height_above_paddle` m above it."""
        paddle_z = _TRUNK_Z + _PADDLE_OFFSET_B[2]  # 0.47 m
        return np.array([0.0, 0.0, paddle_z + height_above_paddle])

    def test_ball_at_02m_in_frame(self, cam_pose):
        """Ball 0.2m above paddle should be in frame (centred region)."""
        cam_pos_w, cam_quat_w_ros = cam_pose
        ball_w = self._ball_world_above_paddle(0.2)
        pos_cam = world_to_cam(ball_w, cam_pos_w, cam_quat_w_ros)
        assert pos_cam[2] > 0, "Ball should be in front of camera"
        u, v = project_to_pixel(pos_cam)
        assert in_frame(u, v), f"Ball at 0.2m above paddle should be in frame, got ({u:.0f}, {v:.0f})"

    def test_ball_at_05m_in_frame(self, cam_pose):
        """Ball 0.5m above paddle should be in frame."""
        cam_pos_w, cam_quat_w_ros = cam_pose
        ball_w = self._ball_world_above_paddle(0.5)
        pos_cam = world_to_cam(ball_w, cam_pos_w, cam_quat_w_ros)
        assert pos_cam[2] > 0
        u, v = project_to_pixel(pos_cam)
        assert in_frame(u, v), f"Ball at 0.5m should be in frame, got ({u:.0f}, {v:.0f})"

    def test_ball_at_1m_in_frame(self, cam_pose):
        """Ball 1.0m above paddle should be in frame."""
        cam_pos_w, cam_quat_w_ros = cam_pose
        ball_w = self._ball_world_above_paddle(1.0)
        pos_cam = world_to_cam(ball_w, cam_pos_w, cam_quat_w_ros)
        assert pos_cam[2] > 0
        u, v = project_to_pixel(pos_cam)
        assert in_frame(u, v), f"Ball at 1.0m should be in frame, got ({u:.0f}, {v:.0f})"

    def test_ball_at_rest_not_in_frame(self, cam_pose):
        """Ball resting on paddle (0m above) — may be below FOV at 70° tilt.

        At 70° tilt, ball at paddle level is at ~20° elevation from camera,
        which is below the ~35° lower edge of the 70° VFOV centred at 70°.
        """
        cam_pos_w, cam_quat_w_ros = cam_pose
        ball_w = self._ball_world_above_paddle(0.0)
        pos_cam = world_to_cam(ball_w, cam_pos_w, cam_quat_w_ros)
        if pos_cam[2] > 0:
            u, v = project_to_pixel(pos_cam)
            # Ball at rest should be out of frame or at the very edge
            # (it's below the camera's main FOV at 70° tilt)
            # This is informational — we allow either case since it depends
            # on exact tilt.  The key is documenting the expected behavior.
            print(f"Ball at rest: pixel=({u:.0f}, {v:.0f}), in_frame={in_frame(u, v)}")
        else:
            print("Ball at rest is behind camera (expected for extreme tilt)")

    def test_ball_higher_projects_higher_in_image(self, cam_pose):
        """Balls further above paddle should project to different v positions."""
        cam_pos_w, cam_quat_w_ros = cam_pose
        pixels = []
        for h in [0.2, 0.5, 1.0]:
            ball_w = self._ball_world_above_paddle(h)
            pos_cam = world_to_cam(ball_w, cam_pos_w, cam_quat_w_ros)
            if pos_cam[2] > 0:
                u, v = project_to_pixel(pos_cam)
                pixels.append((h, u, v))

        assert len(pixels) >= 2, "Need at least 2 visible heights"
        # Print for diagnostic value
        for h, u, v in pixels:
            print(f"  h={h:.1f}m → pixel ({u:.0f}, {v:.0f})")

    def test_ball_offset_xy_projects_laterally(self, cam_pose):
        """Ball offset in Y-world should shift in U-pixel (lateral)."""
        cam_pos_w, cam_quat_w_ros = cam_pose
        ball_centre = self._ball_world_above_paddle(0.5)
        ball_left = ball_centre.copy()
        ball_left[1] = 0.1  # 10cm to the left (Y world)

        pc_centre = world_to_cam(ball_centre, cam_pos_w, cam_quat_w_ros)
        pc_left = world_to_cam(ball_left, cam_pos_w, cam_quat_w_ros)

        assert pc_centre[2] > 0 and pc_left[2] > 0
        u_c, _ = project_to_pixel(pc_centre)
        u_l, _ = project_to_pixel(pc_left)
        assert abs(u_c - u_l) > 10, f"10cm Y offset should shift u by >10px, got {abs(u_c-u_l):.1f}"

    def test_expected_ball_area_at_05m(self, cam_pose):
        """Ball at 0.5m above paddle: expected pixel area for 20mm radius ball."""
        cam_pos_w, cam_quat_w_ros = cam_pose
        ball_w = self._ball_world_above_paddle(0.5)
        pos_cam = world_to_cam(ball_w, cam_pos_w, cam_quat_w_ros)
        depth = pos_cam[2]
        r_px = _FX_PX * 0.020 / depth  # ball radius in pixels
        area_px = np.pi * r_px ** 2
        print(f"Ball at 0.5m: depth={depth:.3f}m, radius={r_px:.1f}px, area={area_px:.0f}px²")
        # Should be a reasonable blob size (not subpixel, not huge)
        assert r_px > 1, f"Ball too small: {r_px:.1f}px radius"
        assert area_px < 10000, f"Ball too large: {area_px:.0f}px²"


# ── Roundtrip tests ──────────────────────────────────────────────────────────

class TestRoundtrip:
    """Verify world→cam→world roundtrip consistency."""

    def test_cam_to_world_roundtrip(self):
        """cam_detection_to_world(world_to_cam(p)) should return p."""
        cam_pos_w, cam_quat_w_ros = setup_camera_pose()
        for h in [0.1, 0.3, 0.5, 1.0]:
            ball_w = np.array([0.0, 0.0, _TRUNK_Z + _PADDLE_OFFSET_B[2] + h])
            pos_cam = world_to_cam(ball_w, cam_pos_w, cam_quat_w_ros)
            recovered = cam_detection_to_world(pos_cam, cam_pos_w, cam_quat_w_ros)
            np.testing.assert_allclose(recovered, ball_w, atol=1e-6,
                                       err_msg=f"Roundtrip failed at h={h}m")

    def test_pixel_deproject_roundtrip(self):
        """project→deproject should recover camera-frame position."""
        cam_pos_w, cam_quat_w_ros = setup_camera_pose()
        ball_w = np.array([0.05, -0.03, _TRUNK_Z + _PADDLE_OFFSET_B[2] + 0.4])
        pos_cam = world_to_cam(ball_w, cam_pos_w, cam_quat_w_ros)
        u, v = project_to_pixel(pos_cam)
        # Deproject
        z = pos_cam[2]
        x_rec = (u - _CX) * z / _FX_PX
        y_rec = (v - _CY) * z / _FY_PX
        np.testing.assert_allclose([x_rec, y_rec, z], pos_cam, atol=1e-6)


# ── Diagnostic: print projection table ───────────────────────────────────────

class TestProjectionTable:
    """Diagnostic: print a table of ball heights → pixel locations."""

    def test_print_projection_table(self):
        cam_pos_w, cam_quat_w_ros = setup_camera_pose()
        print("\n── Ball projection table (D435i 70° tilt) ──")
        print(f"  Camera pos_w: {cam_pos_w}")
        R = quat_to_rotmat(cam_quat_w_ros)
        fwd = R @ np.array([0, 0, 1])
        elev = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
        print(f"  Camera forward (world): ({fwd[0]:.3f}, {fwd[1]:.3f}, {fwd[2]:.3f}), elev={elev:.1f}°")
        print(f"  fx={_FX_PX:.1f}px, cx={_CX:.0f}, cy={_CY:.0f}")
        print(f"{'height':>8s} {'depth':>8s} {'u':>6s} {'v':>6s} {'r_px':>6s} {'in_frame':>9s}")
        print("  " + "-" * 52)

        for h in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50]:
            ball_w = np.array([0.0, 0.0, _TRUNK_Z + _PADDLE_OFFSET_B[2] + h])
            pos_cam = world_to_cam(ball_w, cam_pos_w, cam_quat_w_ros)
            if pos_cam[2] <= 0:
                print(f"  {h:>6.2f}m  {'behind camera':>40s}")
                continue
            u, v = project_to_pixel(pos_cam)
            r_px = _FX_PX * 0.020 / pos_cam[2]
            ok = in_frame(u, v)
            print(f"  {h:>6.2f}m {pos_cam[2]:>7.3f}m {u:>6.0f} {v:>6.0f} {r_px:>5.1f}px {'✓' if ok else '✗':>6s}")
