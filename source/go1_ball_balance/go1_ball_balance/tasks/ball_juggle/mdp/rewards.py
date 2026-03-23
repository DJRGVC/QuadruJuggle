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
    paddle_radius: float = 0.153,
) -> torch.Tensor:
    """Gaussian reward for launching the ball to a target apex height above the paddle.

    Returns exp(-|h - target_height|^2 / 2σ^2) where h is the ball's height
    above the paddle surface (0 when resting on the paddle).  Peak reward of 1.0
    when the ball is exactly at target_height; decays symmetrically above and below.

    This fires every policy step — when the ball is ascending through target_height
    AND when descending through it — so repeated bouncing earns reward continuously.

    Height-gated so a collapsed robot earns no ball reward.

    Args:
        target_height: Target ball height above paddle surface (metres).
            Curriculum stages: 0.10 → 0.20 → 0.30 → 0.45 → 0.60 m.
        std: Gaussian half-width (metres).  Tighter = more precise apex required.
            Curriculum tightens in lock-step with target_height.
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

    # Per-env target height (set by randomize_apex_height event) or fixed param
    if hasattr(env, "_apex_target_h"):
        t_h = env._apex_target_h  # (N,)
        t_std = env._apex_target_std if hasattr(env, "_apex_target_std") else torch.full_like(t_h, std)
    else:
        t_h = target_height
        t_std = std

    # Gaussian kernel centred at target_height
    reward = torch.exp(-(h - t_h).pow(2) / (2.0 * t_std ** 2))

    # Height gate: suppress reward when robot is collapsed
    trunk_z = trunk_pos_w[:, 2]
    height_gate = torch.clamp(
        (trunk_z - min_height) / (nominal_height - min_height), 0.0, 1.0
    )

    # Paddle XY gate: zero reward if ball is outside the paddle disc area.
    # Prevents the robot from cheating by juggling with its body instead of paddle.
    ball_xy = ball.data.root_pos_w[:, :2]
    paddle_xy = paddle_pos_w[:, :2]
    xy_dist = torch.norm(ball_xy - paddle_xy, dim=-1)           # (N,)
    paddle_gate = (xy_dist <= paddle_radius).float()            # 1 inside, 0 outside

    return height_gate * paddle_gate * reward


def ball_bouncing_reward(
    env: "ManagerBasedRLEnv",
    target_vz: float = 1.5,
    std: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.070),
    min_height: float = 0.34,
    nominal_height: float = 0.40,
) -> torch.Tensor:
    """Reward for high ball vertical speed — penalises cradling (ball_vz ≈ 0).

    Uses a half-Gaussian reward: peak 1.0 when |ball_vz| >= target_vz,
    decaying when the ball is slow (resting on paddle).

    Height-gated like ball_apex_height_reward.

    Args:
        target_vz: Target |ball_vz| in m/s. Default 1.5 m/s ≈ 0.11m apex.
        std: Gaussian width in m/s. Wider = more forgiving of low speed.
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

    ball_vz_abs = ball.data.root_lin_vel_w[:, 2].abs()  # (N,)

    # Half-Gaussian: full reward above target_vz, decays below
    speed_err = (ball_vz_abs - target_vz).clamp(max=0.0)  # 0 when fast, negative when slow
    reward = torch.exp(-speed_err.pow(2) / (2.0 * std ** 2))

    # Height gate: suppress reward when robot is collapsed
    trunk_z = robot.data.root_pos_w[:, 2]
    height_gate = torch.clamp(
        (trunk_z - min_height) / (nominal_height - min_height), 0.0, 1.0
    )

    return height_gate * reward
