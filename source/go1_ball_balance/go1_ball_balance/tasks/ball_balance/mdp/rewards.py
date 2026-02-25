"""Custom reward terms for the ball-balance task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_on_paddle_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.18),
) -> torch.Tensor:
    """Exponential reward for keeping the ball centred on the paddle.

    r = exp(-||ball_xy - paddle_center_xy||^2 / (2 * std^2))

    This gives r=1.0 when the ball is perfectly centred and falls off with a
    Gaussian envelope of width ``std`` (metres).

    Args:
        env: The RL environment.
        std: Gaussian half-width in metres. Smaller = tighter.
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.
        paddle_offset_b: Paddle-centre offset from trunk origin (body frame).

    Returns:
        Tensor of shape (num_envs,).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_rotate(trunk_quat_w, offset_b)

    ball_pos_w = ball.data.root_pos_w

    # XY distance only — we don't penalise height directly
    dist_xy = torch.norm(ball_pos_w[:, :2] - paddle_pos_w[:, :2], dim=-1)

    return torch.exp(-dist_xy.pow(2) / (2.0 * std**2))


def body_lin_vel_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise large linear velocity of the robot base (encourages standing still).

    Returns:
        Tensor of shape (num_envs,) — negative values suitable for a negative weight.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    lin_vel = robot.data.root_lin_vel_b          # body-frame linear velocity (N, 3)
    return torch.norm(lin_vel, dim=-1)


def body_ang_vel_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise large angular velocity of the robot base.

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ang_vel = robot.data.root_ang_vel_b          # body-frame angular velocity (N, 3)
    return torch.norm(ang_vel, dim=-1)
