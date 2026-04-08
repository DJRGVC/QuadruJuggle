"""Custom reward terms for the ball-juggle task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_apex_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.10,
    std: float = 0.10,
    ball_radius: float = 0.020,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.070),
    min_height: float = 0.34,
    nominal_height: float = 0.40,
) -> torch.Tensor:
    """Gaussian reward for launching the ball to a target apex height above the paddle.

    Returns exp(-|h - target_height|^2 / 2σ^2) where h is the ball's height
    above the paddle surface (0 when resting on the paddle).  Peak reward of 1.0
    when the ball is exactly at target_height; decays symmetrically above and below.

    Supports per-env targets: if ``env._target_apex_heights`` exists (set by the
    ``randomize_target_apex`` reset event), those per-env values are used instead
    of the scalar ``target_height`` / ``std`` params.

    Height-gated so a collapsed robot earns no ball reward.

    Args:
        target_height: Fallback scalar target (used when per-env targets are not set).
        std: Fallback scalar sigma.
        ball_radius: Physical radius of the ball (metres).
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.
        paddle_offset_b: Paddle-centre offset from trunk origin (body frame).
        min_height: Trunk Z below which gate is 0 (metres).
        nominal_height: Trunk Z at which gate reaches 1 (metres).

    Returns:
        Tensor of shape (num_envs,).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

    # Height of ball above the paddle surface (0 when resting, >0 when airborne)
    ball_z = ball.data.root_pos_w[:, 2]
    paddle_z = paddle_pos_w[:, 2]
    h = ball_z - paddle_z - ball_radius   # (N,)

    # Per-env targets override scalar params when available
    if hasattr(env, "_target_apex_heights"):
        tgt = env._target_apex_heights   # (N,)
        sig = env._target_apex_sigmas    # (N,)
    else:
        tgt = target_height
        sig = std

    # Gaussian kernel centred at target_height
    reward = torch.exp(-(h - tgt).pow(2) / (2.0 * sig ** 2))

    # Height gate: suppress reward when robot is collapsed
    trunk_z = trunk_pos_w[:, 2]
    height_gate = torch.clamp(
        (trunk_z - min_height) / (nominal_height - min_height), 0.0, 1.0
    )

    return height_gate * reward


def ball_low_penalty(
    env: ManagerBasedRLEnv,
    low_threshold: float = 0.03,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.070),
    ball_radius: float = 0.020,
    min_height: float = 0.20,
) -> torch.Tensor:
    """Penalty when ball stays too close to the paddle surface (anti-balance).

    Returns 1.0 when the ball is at or below ``low_threshold`` above the paddle
    surface (ball is resting / barely moving). Returns 0.0 when ball is above
    ``low_threshold``. This breaks the balance-not-bounce local optimum by making
    passive balancing costly.

    Use with a negative weight (e.g., -1.0) in RewardsCfg.

    Args:
        low_threshold: Height above paddle surface below which penalty is active (metres).
            Default 0.03m = ball resting + small bounce range.
        ball_cfg: Scene entity for the ball.
        robot_cfg: Scene entity for the robot.
        paddle_offset_b: Paddle offset from trunk origin (body frame).
        ball_radius: Physical radius of the ball (metres).
        min_height: Trunk Z gate — no penalty if robot is collapsed.

    Returns:
        Tensor of shape (num_envs,) in [0, 1].
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

    ball_z = ball.data.root_pos_w[:, 2]
    paddle_z = paddle_pos_w[:, 2]
    h = ball_z - paddle_z - ball_radius   # (N,) — 0 when resting

    # Gate: only active when robot is upright
    trunk_z = trunk_pos_w[:, 2]
    active = (trunk_z > min_height).float()

    # Penalty: 1.0 when h <= low_threshold, 0.0 otherwise
    penalty = (h <= low_threshold).float()
    return active * penalty


def ball_release_velocity_reward(
    env: ManagerBasedRLEnv,
    max_vel: float = 3.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.070),
    ball_radius: float = 0.020,
    min_height: float = 0.20,
) -> torch.Tensor:
    """Reward upward ball velocity when the ball is airborne (DribbleBot/JuggleRL pattern).

    Returns vz / max_vel (clamped to [0, 1]) when the ball is above the paddle surface
    and moving upward. Zero when ball is resting on the paddle or moving downward.

    This rewards the ACT of throwing — proportional to launch velocity — without
    penalizing the fall phase, providing safe exploration signal that doesn't
    create death spirals when combined with ball_low_penalty.

    Args:
        max_vel: Upward velocity that saturates the reward at 1.0 (metres/s).
        ball_cfg: Scene entity for the ball.
        robot_cfg: Scene entity for the robot.
        paddle_offset_b: Paddle offset from trunk origin (body frame).
        ball_radius: Physical radius of the ball (metres).
        min_height: Trunk Z gate — no reward if robot is collapsed.

    Returns:
        Tensor of shape (num_envs,) in [0, 1].
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

    ball_z = ball.data.root_pos_w[:, 2]
    paddle_z = paddle_pos_w[:, 2]
    h = ball_z - paddle_z - ball_radius   # 0 when resting

    ball_vel_z = ball.data.root_lin_vel_w[:, 2]  # vertical velocity in world frame

    # Only reward when ball is airborne AND moving upward
    airborne = (h > 0.001).float()
    upward_vel = torch.clamp(ball_vel_z, 0.0, max_vel) / max_vel

    # Height gate: no reward when robot is collapsed
    trunk_z = trunk_pos_w[:, 2]
    active = (trunk_z > min_height).float()

    return active * airborne * upward_vel
