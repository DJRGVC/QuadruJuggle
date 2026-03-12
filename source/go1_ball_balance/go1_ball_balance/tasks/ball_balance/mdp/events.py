"""Custom event terms for the ball-balance task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_paddle_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    paddle_cfg: SceneEntityCfg = SceneEntityCfg("paddle"),
    offset_b: tuple[float, float, float] = (0.0, 0.0, 0.13),
) -> None:
    """Reset the kinematic paddle to the default standing position for the given envs.

    Called on episode reset.  We approximate the trunk world position from the
    env origin + standing height (0.40 m) + body-frame offset, with an identity
    quaternion (robot resets upright).  The interval event will correct any
    deviation from the first physics step onward.
    """
    paddle: RigidObject = env.scene[paddle_cfg.name]

    env_origins = env.scene.env_origins[env_ids]          # (N, 3)
    # Robot resets to ~0.40 m above ground with trunk upright
    trunk_pos = env_origins + torch.tensor([0.0, 0.0, 0.40], device=env.device)

    off = torch.tensor(offset_b, device=env.device, dtype=torch.float32)
    paddle_pos = trunk_pos + off.unsqueeze(0)              # offset in world frame (upright)

    quat_identity = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32
    ).unsqueeze(0).expand(len(env_ids), -1)

    pose = torch.cat([paddle_pos, quat_identity], dim=-1)  # (N, 7)
    paddle.write_root_pose_to_sim(pose, env_ids=env_ids)


def reset_ball_on_paddle(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.13),
    ball_radius: float = 0.020,
    xy_std: float = 0.05,
    drop_height_mean: float = 0.20,
    drop_height_std: float = 0.05,
    vel_xy_std: float = 0.0,
) -> None:
    """Reset the ball above the paddle with Gaussian-randomised height and XY position.

    The ball is spawned drop_height_mean ± drop_height_std metres above the paddle
    surface, and xy_std metres (Gaussian) away from the paddle centre in XY.
    This forces the policy to catch/recover the ball from varied positions each
    episode rather than starting in a trivially balanced state.

    Args:
        xy_std:           Gaussian std (metres) for XY offset from paddle centre.
        drop_height_mean: Mean height above paddle surface to spawn ball (metres).
        drop_height_std:  Gaussian std of drop height (metres); set 0 for fixed height.
        vel_xy_std:       Gaussian std (m/s) for random lateral spawn velocity.
                          0 for purely vertical bounce (early curriculum stages).
                          Curriculum increases this in later stages to require the
                          policy to track a drifting ball.
    """
    ball: RigidObject = env.scene[ball_cfg.name]

    env_origins = env.scene.env_origins[env_ids]          # (N, 3)
    n = len(env_ids)

    # Approximate paddle centre in world frame (trunk upright, standing height)
    trunk_pos = env_origins + torch.tensor([0.0, 0.0, 0.40], device=env.device)
    off = torch.tensor(paddle_offset_b, device=env.device, dtype=torch.float32)
    paddle_pos = trunk_pos + off.unsqueeze(0)              # (N, 3)

    # Gaussian XY offset from paddle centre
    xy_noise = torch.randn(n, 2, device=env.device) * xy_std
    ball_pos = paddle_pos.clone()
    ball_pos[:, :2] += xy_noise

    # Gaussian drop height above paddle surface
    drop_height = drop_height_mean + torch.randn(n, device=env.device) * drop_height_std
    drop_height = drop_height.clamp(min=ball_radius)      # never below paddle surface
    ball_pos[:, 2] += drop_height

    # Identity quaternion
    quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32
    ).unsqueeze(0).expand(n, -1)
    pose = torch.cat([ball_pos, quat], dim=-1)             # (N, 7)

    # Random lateral spawn velocity (curriculum adds drift in later stages)
    vel = torch.zeros(n, 6, device=env.device)
    if vel_xy_std > 0.0:
        vel[:, :2] = torch.randn(n, 2, device=env.device) * vel_xy_std

    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(vel, env_ids=env_ids)


def randomize_target_apex(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    target_min: float = 0.30,
    target_max: float = 1.00,
    sigma_ratio: float = 2.5,
) -> None:
    """Randomize per-env target apex height at episode reset.

    Samples target_height uniformly from [target_min, target_max] for each
    reset env.  Sigma is derived as target / sigma_ratio to maintain the
    target/σ ≥ 2.0 invariant.

    Creates ``env._target_apex_heights`` and ``env._target_apex_sigmas``
    buffers on first call.  The ball_apex_height_reward and
    target_apex_height_obs functions read from these buffers when present.
    """
    if not hasattr(env, "_target_apex_heights"):
        mid = (target_min + target_max) / 2.0
        env._target_apex_heights = torch.full(
            (env.num_envs,), mid, device=env.device, dtype=torch.float32,
        )
        env._target_apex_sigmas = torch.full(
            (env.num_envs,), mid / sigma_ratio, device=env.device, dtype=torch.float32,
        )

    targets = torch.empty(len(env_ids), device=env.device).uniform_(target_min, target_max)
    env._target_apex_heights[env_ids] = targets
    env._target_apex_sigmas[env_ids] = targets / sigma_ratio


def update_paddle_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    paddle_cfg: SceneEntityCfg = SceneEntityCfg("paddle"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset_b: tuple[float, float, float] = (0.0, 0.0, 0.13),
) -> None:
    """Continuously track the robot trunk with the kinematic paddle (every policy step).

    Runs as an interval event with interval_range=(1,1) so it fires every step for
    all envs.  We update ALL envs (not just env_ids) so the paddle never lags behind
    a moving trunk.
    """
    paddle: RigidObject = env.scene[paddle_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w     # (num_envs, 3)
    trunk_quat_w = robot.data.root_quat_w   # (num_envs, 4) wxyz

    off = torch.tensor(offset_b, device=env.device, dtype=torch.float32)
    off_w = math_utils.quat_apply(trunk_quat_w, off.unsqueeze(0).expand(env.num_envs, -1))
    paddle_pos_w = trunk_pos_w + off_w

    pose = torch.cat([paddle_pos_w, trunk_quat_w], dim=-1)  # (num_envs, 7)
    paddle.write_root_pose_to_sim(pose)
