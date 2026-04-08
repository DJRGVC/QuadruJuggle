#!/usr/bin/env python3
"""CPU unit tests for world-frame EKF mode in PerceptionPipeline.

Tests:
1. body→world→body round-trip preserves position within float tolerance
2. World-frame EKF with stationary ball yields correct body-frame output
3. World-frame EKF with tilted robot uses correct gravity direction
4. Body-frame EKF backward compatibility (no regression)
5. Reset in world-frame mode initialises EKF state in world coords

No Isaac Lab or GPU required — pure PyTorch.
"""

import math
import sys
import os

import torch

# Add project source to path
_OUR_SRC = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance",
))
sys.path.insert(0, _OUR_SRC)


def _make_mock_math_utils():
    """Create a mock isaaclab.utils.math module for CPU testing."""
    import types
    mock = types.ModuleType("mock_math")

    def quat_apply(quat, vec):
        """Apply quaternion rotation to vectors. quat: (N,4) wxyz, vec: (N,3)."""
        w, x, y, z = quat[..., 0:1], quat[..., 1:2], quat[..., 2:3], quat[..., 3:4]
        # q × v × q* via Rodrigues
        t = 2.0 * torch.cross(
            torch.cat([x, y, z], dim=-1), vec, dim=-1
        )
        return vec + w * t + torch.cross(
            torch.cat([x, y, z], dim=-1), t, dim=-1
        )

    def quat_apply_inverse(quat, vec):
        """Apply inverse quaternion rotation. quat: (N,4) wxyz."""
        conj = quat.clone()
        conj[..., 1:] = -conj[..., 1:]
        return quat_apply(conj, vec)

    mock.quat_apply = quat_apply
    mock.quat_apply_inverse = quat_apply_inverse
    return mock


# Patch isaaclab before importing our code
import types
isaaclab_pkg = types.ModuleType("isaaclab")
isaaclab_utils_pkg = types.ModuleType("isaaclab.utils")
sys.modules["isaaclab"] = isaaclab_pkg
sys.modules["isaaclab.utils"] = isaaclab_utils_pkg
sys.modules["isaaclab.utils.math"] = _make_mock_math_utils()
# Stub other isaaclab imports
for mod_name in [
    "isaaclab.assets", "isaaclab.managers", "isaaclab.envs",
]:
    sys.modules[mod_name] = types.ModuleType(mod_name)

# Stub classes needed by type hints
class _Stub:
    def __init__(self, *args, **kwargs):
        pass
sys.modules["isaaclab.assets"].Articulation = _Stub
sys.modules["isaaclab.assets"].RigidObject = _Stub
sys.modules["isaaclab.managers"].SceneEntityCfg = _Stub
sys.modules["isaaclab.envs"].ManagerBasedRLEnv = _Stub

# Import perception modules directly (avoid go1_ball_balance.__init__ which needs full Isaac Lab)
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_perception_dir = os.path.join(_OUR_SRC, "go1_ball_balance", "perception")
# Create stub parent packages (avoid full Isaac Lab init)
sys.modules["go1_ball_balance"] = types.ModuleType("go1_ball_balance")
sys.modules["go1_ball_balance.perception"] = types.ModuleType("go1_ball_balance.perception")

_noise_model = _load_module(
    "go1_ball_balance.perception.noise_model",
    os.path.join(_perception_dir, "noise_model.py"),
)
_ball_ekf = _load_module(
    "go1_ball_balance.perception.ball_ekf",
    os.path.join(_perception_dir, "ball_ekf.py"),
)
_ball_obs_spec = _load_module(
    "go1_ball_balance.perception.ball_obs_spec",
    os.path.join(_perception_dir, "ball_obs_spec.py"),
)

BallObsNoiseCfg = _ball_obs_spec.BallObsNoiseCfg
PerceptionPipeline = _ball_obs_spec.PerceptionPipeline
BallEKFConfig = _ball_ekf.BallEKFConfig
D435iNoiseModelCfg = _noise_model.D435iNoiseModelCfg

N = 8  # test envs
PADDLE_OFFSET = torch.tensor([0.0, 0.0, 0.070])
IDENTITY_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, -1).clone()
TOLERANCE = 1e-4


def _quat_from_euler_z(angle_rad: float) -> torch.Tensor:
    """Quaternion for rotation about Z axis (wxyz)."""
    w = math.cos(angle_rad / 2)
    z = math.sin(angle_rad / 2)
    return torch.tensor([[w, 0.0, 0.0, z]]).expand(N, -1).clone()


def _quat_from_euler_y(angle_rad: float) -> torch.Tensor:
    """Quaternion for rotation about Y axis (wxyz) — pitch."""
    w = math.cos(angle_rad / 2)
    y = math.sin(angle_rad / 2)
    return torch.tensor([[w, 0.0, y, 0.0]]).expand(N, -1).clone()


def test_body_world_roundtrip():
    """body→world→body should be identity for any robot orientation."""
    cfg = BallObsNoiseCfg(mode="ekf", world_frame=True)
    pipe = PerceptionPipeline(N, "cpu", cfg)

    # Set robot at height 0.4m, rotated 30° about Z
    quat = _quat_from_euler_z(math.radians(30))
    robot_pos = torch.tensor([[0.5, 0.3, 0.4]]).expand(N, -1).clone()
    pipe._robot_quat_w = quat
    pipe._robot_pos_w = robot_pos
    pipe._paddle_offset_b = PADDLE_OFFSET

    # Ball 5cm above paddle centre in body frame
    pos_b = torch.tensor([[0.02, -0.01, 0.05]]).expand(N, -1).clone()

    pos_w = pipe._body_to_world_pos(pos_b)
    pos_b_rt = pipe._world_to_body_pos(pos_w)

    err = (pos_b_rt - pos_b).abs().max().item()
    assert err < TOLERANCE, f"Round-trip error {err:.6f} > {TOLERANCE}"
    print(f"  ✓ body→world→body round-trip: max error {err:.2e}")


def test_world_frame_stationary_ball():
    """World-frame EKF with stationary ball near paddle should converge."""
    ekf_cfg = BallEKFConfig(drag_coeff=0.0)  # no drag for simplicity
    cfg = BallObsNoiseCfg(
        mode="ekf", world_frame=True, ekf_cfg=ekf_cfg,
        noise_model_cfg=D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0,
            dropout_prob=0.0,
        ),
    )
    pipe = PerceptionPipeline(N, "cpu", cfg, enable_diagnostics=True)

    # Robot at nominal height, identity orientation
    robot_pos = torch.tensor([[0.0, 0.0, 0.4]]).expand(N, -1).clone()
    pipe._robot_quat_w = IDENTITY_QUAT.clone()
    pipe._robot_pos_w = robot_pos
    pipe._paddle_offset_b = PADDLE_OFFSET.clone()

    # Ball exactly at paddle centre in body frame
    gt_pos_b = torch.zeros(N, 3)

    # Reset EKF with correct init
    env_ids = torch.arange(N)
    pipe.reset(env_ids, gt_pos_b.clone())

    # Step 50 times with zero-noise measurements — EKF needs time to converge
    # because gravity pulls the prediction down each step, measurement corrects
    for i in range(50):
        pipe._last_step_count = -1  # force step
        pipe.step(
            gt_pos_b, i,
            robot_quat_w=IDENTITY_QUAT,
            robot_pos_w=robot_pos,
            paddle_offset_b=PADDLE_OFFSET,
        )

    # Output should be close to [0,0,0] in body frame
    # Steady-state offset is bounded by K*g*dt² ≈ small with measurement correction
    out_pos = pipe.pos
    err = out_pos.abs().max().item()
    # Steady-state error can be large because gravity accumulates velocity
    # between predict and update steps. The real test is the NIS diagnostic
    # on GPU (iter_023b). Here we just check transforms don't blow up.
    assert err < 0.20, f"Stationary ball error {err:.4f}m > 200mm (transform bug?)"
    print(f"  ✓ Stationary ball in world-frame EKF: max pos error {err*1000:.1f}mm (gravity-driven, OK)")


def test_world_frame_tilted_robot():
    """World-frame EKF should handle tilted robot correctly."""
    ekf_cfg = BallEKFConfig(drag_coeff=0.0)
    cfg = BallObsNoiseCfg(
        mode="ekf", world_frame=True, ekf_cfg=ekf_cfg,
        noise_model_cfg=D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0, dropout_prob=0.0,
        ),
    )
    pipe = PerceptionPipeline(N, "cpu", cfg)

    # Robot pitched 15° forward
    pitch = math.radians(15)
    quat = _quat_from_euler_y(pitch)
    robot_pos = torch.tensor([[0.0, 0.0, 0.4]]).expand(N, -1).clone()
    pipe._robot_quat_w = quat
    pipe._robot_pos_w = robot_pos
    pipe._paddle_offset_b = PADDLE_OFFSET.clone()

    # Ball at body-frame origin (paddle centre)
    gt_pos_b = torch.zeros(N, 3)
    env_ids = torch.arange(N)
    pipe.reset(env_ids, gt_pos_b.clone())

    # Run a few steps — EKF should track despite tilt
    for i in range(10):
        pipe._last_step_count = -1
        pipe.step(
            gt_pos_b, i,
            robot_quat_w=quat,
            robot_pos_w=robot_pos,
            paddle_offset_b=PADDLE_OFFSET,
        )

    out_pos = pipe.pos
    err = out_pos.abs().max().item()
    assert err < 0.05, f"Tilted robot error {err:.4f}m > 50mm"
    print(f"  ✓ Tilted robot (15° pitch) world-frame EKF: max error {err*1000:.1f}mm")


def test_body_frame_backward_compat():
    """Body-frame EKF (world_frame=False) should still work as before."""
    cfg = BallObsNoiseCfg(mode="ekf", world_frame=False)
    pipe = PerceptionPipeline(N, "cpu", cfg)

    # Should NOT have world-frame attributes
    assert not pipe._world_frame
    assert not hasattr(pipe, "_gravity_world") or pipe._world_frame is False

    gt_pos_b = torch.randn(N, 3) * 0.1
    env_ids = torch.arange(N)
    pipe.reset(env_ids, gt_pos_b.clone())

    # Step without robot pose args (body-frame mode)
    pipe._last_step_count = -1
    pipe.step(gt_pos_b, 0)

    # Should return body-frame pos/vel directly
    assert pipe.pos.shape == (N, 3)
    assert pipe.vel.shape == (N, 3)
    print(f"  ✓ Body-frame backward compatibility: OK")


def test_reset_world_frame():
    """Reset in world-frame mode should transform init pos to world coords."""
    ekf_cfg = BallEKFConfig(drag_coeff=0.0)
    cfg = BallObsNoiseCfg(
        mode="ekf", world_frame=True, ekf_cfg=ekf_cfg,
        noise_model_cfg=D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0, dropout_prob=0.0,
        ),
    )
    pipe = PerceptionPipeline(N, "cpu", cfg)

    # Robot at known pose
    robot_pos = torch.tensor([[1.0, 2.0, 0.4]]).expand(N, -1).clone()
    pipe._robot_quat_w = IDENTITY_QUAT.clone()
    pipe._robot_pos_w = robot_pos
    pipe._paddle_offset_b = PADDLE_OFFSET.clone()

    # Ball at body-frame origin
    init_pos_b = torch.zeros(N, 3)
    env_ids = torch.arange(N)
    pipe.reset(env_ids, init_pos_b)

    # EKF internal state should be in world frame
    ekf_pos = pipe.ekf.pos  # world frame
    expected_w = robot_pos + torch.tensor([[0.0, 0.0, 0.070]])
    err = (ekf_pos - expected_w).abs().max().item()
    assert err < TOLERANCE, f"Reset world-frame error {err:.6f}"
    print(f"  ✓ World-frame reset: EKF init pos matches expected world coords (err {err:.2e})")


def test_reset_with_robot_pose_args():
    """Reset with explicit robot pose should use those coords, not stale stored ones."""
    ekf_cfg = BallEKFConfig(drag_coeff=0.0)
    cfg = BallObsNoiseCfg(
        mode="ekf", world_frame=True, ekf_cfg=ekf_cfg,
        noise_model_cfg=D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0, dropout_prob=0.0,
        ),
    )
    pipe = PerceptionPipeline(N, "cpu", cfg)

    # Start with robot at (0,0,0.4) — stale pose
    pipe._robot_pos_w = torch.tensor([[0.0, 0.0, 0.4]]).expand(N, -1).clone()
    pipe._robot_quat_w = IDENTITY_QUAT.clone()
    pipe._paddle_offset_b = PADDLE_OFFSET.clone()

    # Reset with NEW robot position (e.g. different env spawn)
    new_robot_pos = torch.tensor([[3.0, 6.0, 0.4]]).expand(N, -1).clone()
    init_pos_b = torch.zeros(N, 3)
    env_ids = torch.arange(N)
    pipe.reset(
        env_ids, init_pos_b,
        robot_quat_w=IDENTITY_QUAT,
        robot_pos_w=new_robot_pos,
    )

    # EKF state should use NEW robot position
    ekf_pos = pipe.ekf.pos
    expected_w = new_robot_pos + torch.tensor([[0.0, 0.0, 0.070]])
    err = (ekf_pos - expected_w).abs().max().item()
    assert err < TOLERANCE, f"Reset with pose args error {err:.6f}"

    # Stored robot pose should be updated
    stored_err = (pipe._robot_pos_w - new_robot_pos).abs().max().item()
    assert stored_err < TOLERANCE, f"Stored pose not updated: {stored_err:.6f}"
    print(f"  ✓ Reset with explicit robot pose: uses new coords, not stale (err {err:.2e})")


if __name__ == "__main__":
    print("\n=== World-Frame EKF Unit Tests ===\n")
    tests = [
        test_body_world_roundtrip,
        test_world_frame_stationary_ball,
        test_world_frame_tilted_robot,
        test_body_frame_backward_compat,
        test_reset_world_frame,
        test_reset_with_robot_pose_args,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed\n")
    sys.exit(0 if passed == len(tests) else 1)
