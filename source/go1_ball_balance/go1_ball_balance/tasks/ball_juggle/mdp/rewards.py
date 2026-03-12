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
