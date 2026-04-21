"""Reward terms for the torso-tracking task.

Each tracking reward is a Gaussian kernel centred on the commanded target
value, giving peak reward 1.0 at perfect tracking and smooth decay for
deviations.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_rpy(quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract roll, pitch, yaw from wxyz quaternion batch."""
    return math_utils.euler_xyz_from_quat(quat_w)


def height_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.03,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk height matching h_target.

    reward = exp(-|z - h_target|^2 / 2*std^2)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    trunk_z = robot.data.root_pos_w[:, 2]
    h_target = env._torso_cmd[:, 0]
    return torch.exp(-(trunk_z - h_target).pow(2) / (2.0 * std**2))


def height_vel_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk vertical velocity matching h_dot_target."""
    robot: Articulation = env.scene[robot_cfg.name]
    z_dot = robot.data.root_lin_vel_w[:, 2]
    h_dot_target = env._torso_cmd[:, 1]
    return torch.exp(-(z_dot - h_dot_target).pow(2) / (2.0 * std**2))


def roll_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk roll matching roll_target."""
    robot: Articulation = env.scene[robot_cfg.name]
    roll, _, _ = _get_rpy(robot.data.root_quat_w)
    roll_target = env._torso_cmd[:, 2]
    return torch.exp(-(roll - roll_target).pow(2) / (2.0 * std**2))


def pitch_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk pitch matching pitch_target."""
    robot: Articulation = env.scene[robot_cfg.name]
    _, pitch, _ = _get_rpy(robot.data.root_quat_w)
    pitch_target = env._torso_cmd[:, 3]
    return torch.exp(-(pitch - pitch_target).pow(2) / (2.0 * std**2))


def roll_rate_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk roll rate matching omega_roll target."""
    robot: Articulation = env.scene[robot_cfg.name]
    omega_roll = robot.data.root_ang_vel_b[:, 0]
    omega_roll_target = env._torso_cmd[:, 4]
    return torch.exp(-(omega_roll - omega_roll_target).pow(2) / (2.0 * std**2))


def pitch_rate_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk pitch rate matching omega_pitch target."""
    robot: Articulation = env.scene[robot_cfg.name]
    omega_pitch = robot.data.root_ang_vel_b[:, 1]
    omega_pitch_target = env._torso_cmd[:, 5]
    return torch.exp(-(omega_pitch - omega_pitch_target).pow(2) / (2.0 * std**2))


# ---------------------------------------------------------------------------
# 9D-extension rewards: vx, vy, omega_yaw tracking (body frame)
# ---------------------------------------------------------------------------


def vx_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.30,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for body-frame forward velocity matching vx target.

    Measured in body frame so the command is heading-invariant.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    vx = robot.data.root_lin_vel_b[:, 0]
    vx_target = env._torso_cmd[:, 6]
    return torch.exp(-(vx - vx_target).pow(2) / (2.0 * std**2))


def vy_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.30,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for body-frame lateral velocity matching vy target."""
    robot: Articulation = env.scene[robot_cfg.name]
    vy = robot.data.root_lin_vel_b[:, 1]
    vy_target = env._torso_cmd[:, 7]
    return torch.exp(-(vy - vy_target).pow(2) / (2.0 * std**2))


def yaw_rate_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.60,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for body-frame yaw rate matching omega_yaw target."""
    robot: Articulation = env.scene[robot_cfg.name]
    omega_yaw = robot.data.root_ang_vel_b[:, 2]
    omega_yaw_target = env._torso_cmd[:, 8]
    return torch.exp(-(omega_yaw - omega_yaw_target).pow(2) / (2.0 * std**2))


def vxy_error_l2(
    env: ManagerBasedRLEnv,
    min_cmd_magnitude: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear L2 penalty on XY-velocity tracking error.

    Provides a constant gradient even at large errors (fills the Gaussian's
    flat tail). Only active when commanded |v_xy| > min_cmd_magnitude so it
    does not fight stationary poses.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    v_xy_cmd = env._torso_cmd[:, 6:8]
    v_xy_meas = robot.data.root_lin_vel_b[:, :2]
    err = torch.norm(v_xy_meas - v_xy_cmd, dim=-1)
    mask = (torch.norm(v_xy_cmd, dim=-1) > min_cmd_magnitude).float()
    return err * mask


def yaw_rate_error_l2(
    env: ManagerBasedRLEnv,
    min_cmd_magnitude: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear L1 penalty on yaw-rate tracking error (dead-zoned)."""
    robot: Articulation = env.scene[robot_cfg.name]
    yaw_cmd = env._torso_cmd[:, 8]
    yaw_meas = robot.data.root_ang_vel_b[:, 2]
    err = (yaw_meas - yaw_cmd).abs()
    mask = (yaw_cmd.abs() > min_cmd_magnitude).float()
    return err * mask
