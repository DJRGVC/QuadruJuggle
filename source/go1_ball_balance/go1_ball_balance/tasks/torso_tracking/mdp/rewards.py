"""Reward terms for the torso-tracking task.

Architecture follows Isaac Lab / Walk-These-Ways convention:
  - Velocity tracking = ONLY positive rewards (Gaussian + feet_air_time)
  - Pose tracking = negative penalties (squared error)
  - Standing still earns 0 positive reward; the only path to reward is moving

Pose penalties use squared error: (actual - target)^2.  Perfect tracking = 0
penalty.  No height gate needed on penalties (they're already 0 at perfect
tracking).

Velocity rewards are gated on height tracking quality (Hoeller et al. 2024)
to prevent the optimizer from collapsing height to chase velocity.
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


def _height_gate(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, gate_std: float = 0.10) -> torch.Tensor:
    """Soft gate based on height tracking quality.

    Returns a value in [0, 1] per env.  Near 1.0 when height is well-tracked,
    decays smoothly as height error grows.  Uses same Gaussian form as
    height_tracking_reward but with a configurable std (gate_std).

    This is called internally by every non-height reward function.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    trunk_z = robot.data.root_pos_w[:, 2]
    h_target = env._torso_cmd[:, 0]
    return torch.exp(-(trunk_z - h_target).pow(2) / (2.0 * gate_std**2))


# ── Pose tracking PENALTIES (squared error, 0 at perfect tracking) ────────────

def height_error_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared error on trunk height vs h_target."""
    robot: Articulation = env.scene[robot_cfg.name]
    return (robot.data.root_pos_w[:, 2] - env._torso_cmd[:, 0]).pow(2)


def height_vel_error_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared error on trunk vertical velocity vs h_dot_target."""
    robot: Articulation = env.scene[robot_cfg.name]
    return (robot.data.root_lin_vel_w[:, 2] - env._torso_cmd[:, 1]).pow(2)


def roll_error_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared error on trunk roll vs roll_target."""
    robot: Articulation = env.scene[robot_cfg.name]
    roll, _, _ = _get_rpy(robot.data.root_quat_w)
    return (roll - env._torso_cmd[:, 2]).pow(2)


def pitch_error_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared error on trunk pitch vs pitch_target."""
    robot: Articulation = env.scene[robot_cfg.name]
    _, pitch, _ = _get_rpy(robot.data.root_quat_w)
    return (pitch - env._torso_cmd[:, 3]).pow(2)


def roll_rate_error_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared error on trunk roll rate vs omega_roll_target."""
    robot: Articulation = env.scene[robot_cfg.name]
    return (robot.data.root_ang_vel_b[:, 0] - env._torso_cmd[:, 4]).pow(2)


def pitch_rate_error_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared error on trunk pitch rate vs omega_pitch_target."""
    robot: Articulation = env.scene[robot_cfg.name]
    return (robot.data.root_ang_vel_b[:, 1] - env._torso_cmd[:, 5]).pow(2)


# ── Legacy Gaussian tracking rewards (kept for reference / vx/vy) ────────────

def height_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.03,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk height matching h_target.

    reward = exp(-|z - h_target|^2 / 2*std^2)

    This is the only UNGATED tracking reward — all others depend on this
    being high before they pay out.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    trunk_z = robot.data.root_pos_w[:, 2]
    h_target = env._torso_cmd[:, 0]
    return torch.exp(-(trunk_z - h_target).pow(2) / (2.0 * std**2))


def height_vel_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    gate_std: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk vertical velocity matching h_dot_target.

    Gated on height tracking quality.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    z_dot = robot.data.root_lin_vel_w[:, 2]
    h_dot_target = env._torso_cmd[:, 1]
    raw = torch.exp(-(z_dot - h_dot_target).pow(2) / (2.0 * std**2))
    gate = _height_gate(env, robot_cfg, gate_std)
    return raw * gate


def roll_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    gate_std: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk roll matching roll_target.

    Gated on height tracking quality.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    roll, _, _ = _get_rpy(robot.data.root_quat_w)
    roll_target = env._torso_cmd[:, 2]
    raw = torch.exp(-(roll - roll_target).pow(2) / (2.0 * std**2))
    gate = _height_gate(env, robot_cfg, gate_std)
    return raw * gate


def pitch_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    gate_std: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk pitch matching pitch_target.

    Gated on height tracking quality.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    _, pitch, _ = _get_rpy(robot.data.root_quat_w)
    pitch_target = env._torso_cmd[:, 3]
    raw = torch.exp(-(pitch - pitch_target).pow(2) / (2.0 * std**2))
    gate = _height_gate(env, robot_cfg, gate_std)
    return raw * gate


def roll_rate_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    gate_std: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk roll rate matching omega_roll target.

    Gated on height tracking quality.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    omega_roll = robot.data.root_ang_vel_b[:, 0]
    omega_roll_target = env._torso_cmd[:, 4]
    raw = torch.exp(-(omega_roll - omega_roll_target).pow(2) / (2.0 * std**2))
    gate = _height_gate(env, robot_cfg, gate_std)
    return raw * gate


def pitch_rate_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    gate_std: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for trunk pitch rate matching omega_pitch target.

    Gated on height tracking quality.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    omega_pitch = robot.data.root_ang_vel_b[:, 1]
    omega_pitch_target = env._torso_cmd[:, 5]
    raw = torch.exp(-(omega_pitch - omega_pitch_target).pow(2) / (2.0 * std**2))
    gate = _height_gate(env, robot_cfg, gate_std)
    return raw * gate


def vx_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.15,
    min_cmd: float = 0.01,
    gate_std: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for body-frame forward velocity matching vx_target.

    Masked to zero when |vx_target| < min_cmd (no free reward for standing still).
    Gated on height tracking quality.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    vx = robot.data.root_lin_vel_b[:, 0]
    vx_target = env._torso_cmd[:, 6]
    raw = torch.exp(-(vx - vx_target).pow(2) / (2.0 * std**2))
    mask = (vx_target.abs() >= min_cmd).float()
    gate = _height_gate(env, robot_cfg, gate_std)
    return raw * mask * gate


def vy_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.15,
    min_cmd: float = 0.01,
    gate_std: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for body-frame lateral velocity matching vy_target.

    Masked to zero when |vy_target| < min_cmd (no free reward for standing still).
    Gated on height tracking quality.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    vy = robot.data.root_lin_vel_b[:, 1]
    vy_target = env._torso_cmd[:, 7]
    raw = torch.exp(-(vy - vy_target).pow(2) / (2.0 * std**2))
    mask = (vy_target.abs() >= min_cmd).float()
    gate = _height_gate(env, robot_cfg, gate_std)
    return raw * mask * gate


def vxy_error_penalty(
    env: ManagerBasedRLEnv,
    min_cmd: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear penalty on velocity tracking error when vxy commands are active.

    Returns L2 norm of (v_actual - v_cmd) in XY, masked to zero when commands
    are near-zero.  Provides constant gradient at ALL error magnitudes — unlike
    the Gaussian tracking reward which has flat tails and lets the policy earn
    high reward by standing still.

    Same principle as ball_xy_dist_penalty vs ball_on_paddle_exp.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    vx = robot.data.root_lin_vel_b[:, 0]
    vy = robot.data.root_lin_vel_b[:, 1]
    vx_cmd = env._torso_cmd[:, 6]
    vy_cmd = env._torso_cmd[:, 7]
    error = ((vx - vx_cmd).pow(2) + (vy - vy_cmd).pow(2)).sqrt()
    mask = (vx_cmd.abs() + vy_cmd.abs() > min_cmd).float()
    return error * mask


def feet_air_time_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    min_cmd: float = 0.01,
    foot_contact_cfg: SceneEntityCfg = SceneEntityCfg("foot_contact_forces"),
) -> torch.Tensor:
    """Reward long steps taken by the feet (Isaac Lab standard, adapted for torso commands).

    On each foot's first ground contact after an airborne phase, rewards
    (air_time - threshold) — encouraging swing durations >= threshold seconds.
    Masked to zero when vxy commands are near-zero (no gait reward for standing).

    Ref: Isaac Lab velocity tracking config; Margolis & Agrawal (RSS 2022).
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[foot_contact_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, foot_contact_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, foot_contact_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # Only reward stepping when velocity commands are active
    vx_cmd = env._torso_cmd[:, 6]
    vy_cmd = env._torso_cmd[:, 7]
    mask = (vx_cmd.abs() + vy_cmd.abs() > min_cmd).float()
    return reward * mask
