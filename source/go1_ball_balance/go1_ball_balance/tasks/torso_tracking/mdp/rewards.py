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
