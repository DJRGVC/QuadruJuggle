"""Custom termination terms for the ball-balance task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_off_paddle(
    env: ManagerBasedRLEnv,
    radius: float = 0.15,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.18),
) -> torch.Tensor:
    """Terminate when the ball drifts further than ``radius`` metres from the
    paddle centre (XY distance in world frame).

    Args:
        env: The RL environment.
        radius: Maximum allowed XY distance (metres) before termination.
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.
        paddle_offset_b: Paddle-centre offset from trunk origin (body frame).

    Returns:
        Bool tensor of shape (num_envs,).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_rotate(trunk_quat_w, offset_b)

    ball_pos_w = ball.data.root_pos_w

    dist_xy = torch.norm(ball_pos_w[:, :2] - paddle_pos_w[:, :2], dim=-1)

    return dist_xy > radius


def ball_below_paddle(
    env: ManagerBasedRLEnv,
    min_height_offset: float = -0.05,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.18),
) -> torch.Tensor:
    """Terminate when the ball falls below the paddle surface (ball has dropped off).

    Args:
        env: The RL environment.
        min_height_offset: How far below the paddle centre the ball can be (metres).
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.
        paddle_offset_b: Paddle-centre offset from trunk origin (body frame).

    Returns:
        Bool tensor of shape (num_envs,).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_rotate(trunk_quat_w, offset_b)

    ball_z = ball.data.root_pos_w[:, 2]
    paddle_z = paddle_pos_w[:, 2]

    return ball_z < (paddle_z + min_height_offset)
