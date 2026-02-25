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
    min_height: float = 0.28,
    nominal_height: float = 0.40,
) -> torch.Tensor:
    """Exponential reward for keeping the ball centred on the paddle,
    gated by trunk height so a collapsed robot earns no ball reward.

    r = height_gate * exp(-||ball_xy - paddle_center_xy||^2 / (2 * std^2))

    height_gate ramps linearly from 0 at ``min_height`` to 1 at
    ``nominal_height``, ensuring the robot must stand upright to collect
    any ball reward at all.

    Args:
        env: The RL environment.
        std: Gaussian half-width in metres. Smaller = tighter.
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.
        paddle_offset_b: Paddle-centre offset from trunk origin (body frame).
        min_height: Trunk Z below which the gate is 0 (metres).
        nominal_height: Trunk Z at which the gate reaches 1 (metres).

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

    # XY distance only
    dist_xy = torch.norm(ball_pos_w[:, :2] - paddle_pos_w[:, :2], dim=-1)

    # Height gate: 0 when trunk is at/below min_height, 1 at nominal standing height
    trunk_z = trunk_pos_w[:, 2]
    height_gate = torch.clamp(
        (trunk_z - min_height) / (nominal_height - min_height), 0.0, 1.0
    )

    return height_gate * torch.exp(-dist_xy.pow(2) / (2.0 * std**2))


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


def trunk_tilt_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise trunk roll and pitch away from level.

    Uses the projected gravity vector: when the robot is perfectly upright,
    the gravity vector in body frame is (0, 0, -1), so the XY components are
    both 0.  Any tilt increases the XY magnitude, which we penalise.

    Returns:
        Tensor of shape (num_envs,) — positive values, apply with negative weight.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # gravity projected into body frame — built-in data field
    gravity_b = robot.data.projected_gravity_b          # (N, 3)
    # XY magnitude = sin(tilt_angle) — 0 when level, 1 when fully sideways
    return torch.norm(gravity_b[:, :2], dim=-1)


def ball_lateral_vel_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.28,
    nominal_height: float = 0.40,
) -> torch.Tensor:
    """Penalise lateral (XY) speed of the ball, gated by trunk height.

    Fires as soon as the ball begins sliding, even while it is still within
    the centering reward radius.  This gives the policy a gradient to actively
    damp ball motion rather than passively watching it roll off.

    Returns:
        Tensor of shape (num_envs,) — positive values, apply with negative weight.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ball_speed_xy = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)

    trunk_z = robot.data.root_pos_w[:, 2]
    height_gate = torch.clamp(
        (trunk_z - min_height) / (nominal_height - min_height), 0.0, 1.0
    )

    return height_gate * ball_speed_xy


def ball_height_penalty(
    env: ManagerBasedRLEnv,
    ball_radius: float = 0.020,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.18),
    min_height: float = 0.28,
    nominal_height: float = 0.40,
) -> torch.Tensor:
    """Penalise the ball being above the paddle surface, gated by trunk height.

    Returns 0 when the ball rests on the paddle surface.  Once airborne the
    penalty grows linearly with bounce height.  The gate suppresses this term
    when the robot is collapsed, so it does not compete with base_height_penalty.

    Args:
        env: The RL environment.
        ball_radius: Physical radius of the ball (metres).
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.
        paddle_offset_b: Paddle-centre offset from trunk origin (body frame).
        min_height: Trunk Z below which the gate is 0 (metres).
        nominal_height: Trunk Z at which the gate reaches 1 (metres).

    Returns:
        Tensor of shape (num_envs,) — positive values, apply with negative weight.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_rotate(trunk_quat_w, offset_b)

    ball_z = ball.data.root_pos_w[:, 2]
    paddle_z = paddle_pos_w[:, 2]

    airborne_height = torch.clamp(ball_z - paddle_z - ball_radius, min=0.0)

    trunk_z = trunk_pos_w[:, 2]
    height_gate = torch.clamp(
        (trunk_z - min_height) / (nominal_height - min_height), 0.0, 1.0
    )

    return height_gate * airborne_height


def base_height_penalty(
    env: ManagerBasedRLEnv,
    min_height: float = 0.28,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise the robot for letting its trunk drop below a minimum height.

    Returns 0 when the trunk is at or above ``min_height`` (normal standing is
    ~0.38 m).  Once the trunk drops below that floor the penalty grows linearly
    with the deficit, making it costly to collapse or squat all the way down.

    Args:
        env: The RL environment.
        min_height: Z threshold in world frame (metres).  Trunk below this
            value incurs a penalty proportional to the shortfall.
        robot_cfg: Scene entity config for the robot.

    Returns:
        Tensor of shape (num_envs,) — positive values, apply with negative weight.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    trunk_z = robot.data.root_pos_w[:, 2]            # world-frame trunk height
    return torch.clamp(min_height - trunk_z, min=0.0)