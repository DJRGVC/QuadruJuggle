"""Custom termination terms for the ball-balance task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def trunk_height_collapsed(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.12,
    grace_steps: int = 50,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when trunk height is below minimum_height, but only after grace_steps.

    The grace period lets physics settle after reset before the termination is armed.
    Without it, joint-angle randomisation at spawn can cause a brief free-fall that
    dips the trunk below the threshold before the policy has taken a single action.

    Args:
        minimum_height: Trunk Z (world frame) below which the episode ends.
        grace_steps:    Number of policy steps to skip at episode start.
        asset_cfg:      Scene entity config for the robot.

    Returns:
        Bool tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    below = asset.data.root_pos_w[:, 2] < minimum_height
    past_grace = env.episode_length_buf >= grace_steps
    return below & past_grace


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
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

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
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

    ball_z = ball.data.root_pos_w[:, 2]
    paddle_z = paddle_pos_w[:, 2]

    return ball_z < (paddle_z + min_height_offset)


def robot_tilt(
    env: ManagerBasedRLEnv,
    max_tilt: float = 0.5,
    grace_steps: int = 50,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when trunk tilts beyond max_tilt radians from upright.

    Uses the projected gravity XY magnitude — 0 when level, 1 when sideways.
    sin(max_tilt) converts the angle threshold to this space.

    Args:
        max_tilt:    Maximum allowed tilt angle in radians.
        grace_steps: Steps to skip after reset before arming.
        asset_cfg:   Scene entity config for the robot.

    Returns:
        Bool tensor of shape (num_envs,).
    """
    import math
    robot: Articulation = env.scene[asset_cfg.name]
    gravity_b = robot.data.projected_gravity_b          # (N, 3)
    tilt_sin = torch.norm(gravity_b[:, :2], dim=-1)     # 0=level, 1=sideways
    too_tilted = tilt_sin > math.sin(max_tilt)
    past_grace = env.episode_length_buf >= grace_steps
    return too_tilted & past_grace
