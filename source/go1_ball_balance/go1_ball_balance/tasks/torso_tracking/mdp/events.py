"""Custom event terms for the torso-tracking task."""

from __future__ import annotations

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

    cmd = env._torso_cmd  # (N, 6): [h, h_dot, roll, pitch, omega_roll, omega_pitch]

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
