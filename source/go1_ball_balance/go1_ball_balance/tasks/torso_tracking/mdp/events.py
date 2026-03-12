"""Custom event terms for the torso-tracking task."""

from __future__ import annotations

import math

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def update_target_marker_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    marker_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    normal_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.070),
) -> None:
    """Move the target disc and normal arrow to show the commanded torso pose.

    The disc is placed at the commanded height + paddle offset, tilted by
    the commanded roll/pitch.  The normal arrow sticks up from the disc
    centre along its local Z axis so you can see tilt direction at a glance.

    Runs at 200 Hz (every physics step) for all envs.
    """
    marker: RigidObject = env.scene[marker_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    if not hasattr(env, "_torso_cmd"):
        return

    cmd = env._torso_cmd  # (N, 8): [h, h_dot, roll, pitch, omega_roll, omega_pitch, vx, vy]

    # Disc position: robot XY + commanded height + paddle offset Z
    robot_pos = robot.data.root_pos_w  # (N, 3)
    disc_pos = robot_pos.clone()
    disc_pos[:, 2] = cmd[:, 0] + paddle_offset_b[2] + 0.3048  # h_target + paddle offset + 1ft visual lift

    # Disc orientation from commanded roll and pitch (yaw = robot yaw)
    _, _, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    roll_cmd = cmd[:, 2]
    pitch_cmd = cmd[:, 3]
    disc_quat = math_utils.quat_from_euler_xyz(roll_cmd, pitch_cmd, yaw)

    disc_pose = torch.cat([disc_pos, disc_quat], dim=-1)  # (N, 7)
    marker.write_root_pose_to_sim(disc_pose)

    # Normal arrow: offset 0.075m along disc local Z (half the arrow height)
    if normal_cfg is not None:
        normal_obj: RigidObject = env.scene[normal_cfg.name]
        up_local = torch.zeros(env.num_envs, 3, device=env.device)
        up_local[:, 2] = 0.075  # half of 0.15m arrow height
        up_world = math_utils.quat_apply(disc_quat, up_local)
        normal_pos = disc_pos + up_world
        normal_pose = torch.cat([normal_pos, disc_quat], dim=-1)
        normal_obj.write_root_pose_to_sim(normal_pose)


def _place_velocity_arrow(
    arrow: RigidObject,
    vx_b: torch.Tensor,
    vy_b: torch.Tensor,
    yaw: torch.Tensor,
    paddle_pos_w: torch.Tensor,
    paddle_radius: float,
    margin: float,
    device: torch.device,
) -> None:
    """Place a single velocity arrow adjacent to the paddle.

    The arrow base starts at paddle edge + margin in the velocity direction.
    Arrow cylinder (local Z = long axis) is tilted horizontal and rotated
    to point in the world-frame velocity direction.

    Hidden underground when speed ≈ 0.
    """
    speed = torch.sqrt(vx_b**2 + vy_b**2)
    vel_angle_b = torch.atan2(vy_b, vx_b)
    vel_angle_w = vel_angle_b + yaw

    # Quaternion: tilt cylinder horizontal, then rotate to velocity direction
    half_pi = torch.full_like(yaw, -math.pi / 2.0)
    zeros = torch.zeros_like(yaw)
    arrow_quat = math_utils.quat_from_euler_xyz(zeros, half_pi, vel_angle_w)

    # Arrow base = paddle centre + (paddle_radius + margin) in velocity dir
    # Then offset another half-arrow-length (0.125m) so base is at the edge
    total_offset = paddle_radius + margin + 0.125  # edge + margin + half arrow
    pos = paddle_pos_w.clone()
    pos[:, 0] += total_offset * torch.cos(vel_angle_w)
    pos[:, 1] += total_offset * torch.sin(vel_angle_w)

    # Hide underground when speed near zero
    pos[speed < 0.01, 2] = -1.0

    arrow_pose = torch.cat([pos, arrow_quat], dim=-1)
    arrow.write_root_pose_to_sim(arrow_pose)


def update_velocity_arrows_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    cmd_arrow_cfg: SceneEntityCfg = SceneEntityCfg("cmd_velocity_arrow"),
    actual_arrow_cfg: SceneEntityCfg = SceneEntityCfg("actual_velocity_arrow"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.070),
    paddle_radius: float = 0.085,
    margin: float = 0.20,
) -> None:
    """Place commanded (cyan) and actual (orange) velocity arrows next to the paddle.

    Each arrow starts at paddle_edge + margin in its respective velocity
    direction.  When tracking is perfect the arrows overlap; divergence
    is immediately visible.
    """
    cmd_arrow: RigidObject = env.scene[cmd_arrow_cfg.name]
    actual_arrow: RigidObject = env.scene[actual_arrow_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    if not hasattr(env, "_torso_cmd"):
        return

    cmd = env._torso_cmd  # (N, 8)
    _, _, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)

    # Paddle world position (robot root + body-frame offset)
    root_pos = robot.data.root_pos_w  # (N, 3)
    offset_b = torch.tensor(paddle_offset_b, device=env.device).expand(env.num_envs, -1)
    offset_w = math_utils.quat_apply(robot.data.root_quat_w, offset_b)
    paddle_pos_w = root_pos + offset_w  # (N, 3)

    # Commanded velocity (body frame)
    _place_velocity_arrow(
        cmd_arrow, cmd[:, 6], cmd[:, 7], yaw,
        paddle_pos_w, paddle_radius, margin, env.device,
    )

    # Actual velocity (body frame)
    vx_actual = robot.data.root_lin_vel_b[:, 0]
    vy_actual = robot.data.root_lin_vel_b[:, 1]
    _place_velocity_arrow(
        actual_arrow, vx_actual, vy_actual, yaw,
        paddle_pos_w, paddle_radius, margin, env.device,
    )
