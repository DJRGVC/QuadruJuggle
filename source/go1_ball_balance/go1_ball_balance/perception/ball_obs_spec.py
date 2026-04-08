"""Ball observation provider — drop-in replacement for oracle obs terms.

ETH-style architecture: GT ball state → noise model → EKF → policy.
This module provides ObsTerm-compatible functions that the env config can
swap in place of the raw ``ball_pos_in_paddle_frame`` / ``ball_vel_in_paddle_frame``.

Noise modes:
    "oracle"  — pass-through ground truth (default, identical to current pipeline)
    "d435i"   — D435i-style structured noise (position + depth-dependent, dropout)
    "ekf"     — noise model → EKF filter (full pipeline, added later)

Usage in env_cfg.py::

    from go1_ball_balance.perception.ball_obs_spec import (
        ball_pos_perceived,
        ball_vel_perceived,
        BallObsNoiseCfg,
    )

    ball_pos = ObsTerm(
        func=ball_pos_perceived,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "noise_cfg": BallObsNoiseCfg(mode="oracle"),
        },
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Noise configuration
# ---------------------------------------------------------------------------

@dataclass
class D435iNoiseParams:
    """D435i depth-camera noise model parameters.

    Defaults from Intel D435i datasheet + empirical measurements:
    - Depth accuracy ~2% of distance at 1m (structured IR stereo)
    - XY accuracy ~1-2mm at 1m (640×480, ~60° HFOV)
    - Dropout probability increases at close range / specular surfaces
    """

    sigma_xy_base: float = 0.002       # 2mm base XY noise std (metres)
    sigma_z_base: float = 0.003        # 3mm base depth noise std
    sigma_z_per_metre: float = 0.002   # +2mm per metre of distance
    dropout_prob: float = 0.02         # 2% chance of missed detection per step
    latency_steps: int = 1             # observation delay in policy steps


@dataclass
class BallObsNoiseCfg:
    """Configuration for the ball observation noise model.

    Pass this as a ``noise_cfg`` param to ``ball_pos_perceived`` and
    ``ball_vel_perceived`` ObsTerms.
    """

    mode: str = "oracle"
    """Noise mode: "oracle", "d435i", or "ekf"."""

    d435i: D435iNoiseParams = field(default_factory=D435iNoiseParams)
    """D435i noise parameters (only used when mode="d435i")."""

    noise_scale: float = 1.0
    """Multiplicative scale applied to all D435i noise std and dropout parameters.

    0.0 = silent (oracle-equivalent even in d435i mode).
    1.0 = full noise as specified by d435i params.
    Values in between allow gradual noise curriculum scheduling.
    The scale applies to: sigma_xy_base, sigma_z_base, sigma_z_per_metre, dropout_prob.
    """


# ---------------------------------------------------------------------------
# Internal: oracle ball state (same math as observations.py)
# ---------------------------------------------------------------------------

def _ball_pos_paddle_frame_gt(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    paddle_offset_b: tuple[float, float, float],
) -> torch.Tensor:
    """Ground-truth ball position in paddle frame. Shape (N, 3)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(
        paddle_offset_b, device=env.device
    ).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

    ball_pos_w = ball.data.root_pos_w
    diff_w = ball_pos_w - paddle_pos_w
    return math_utils.quat_apply_inverse(trunk_quat_w, diff_w)


def _ball_vel_paddle_frame_gt(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ground-truth ball velocity in trunk frame. Shape (N, 3)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_quat_w = robot.data.root_quat_w
    ball_vel_w = ball.data.root_lin_vel_w
    return math_utils.quat_apply_inverse(trunk_quat_w, ball_vel_w)


# ---------------------------------------------------------------------------
# Public ObsTerm-compatible functions
# ---------------------------------------------------------------------------

def ball_pos_perceived(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.070),
    noise_cfg: BallObsNoiseCfg | None = None,
) -> torch.Tensor:
    """Ball position in paddle frame, optionally with perception noise.

    Drop-in replacement for ``mdp.ball_pos_in_paddle_frame``.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    if noise_cfg is None:
        noise_cfg = BallObsNoiseCfg()

    pos_b = _ball_pos_paddle_frame_gt(env, ball_cfg, robot_cfg, paddle_offset_b)

    if noise_cfg.mode == "oracle":
        return pos_b

    if noise_cfg.mode == "d435i":
        return _apply_d435i_pos_noise(pos_b, noise_cfg.d435i, noise_cfg.noise_scale)

    if noise_cfg.mode == "ekf":
        raise NotImplementedError(
            "EKF noise mode not yet implemented — coming in a future iteration."
        )

    raise ValueError(f"Unknown noise mode: {noise_cfg.mode!r}")


def ball_vel_perceived(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noise_cfg: BallObsNoiseCfg | None = None,
) -> torch.Tensor:
    """Ball velocity in trunk frame, optionally with perception noise.

    Drop-in replacement for ``mdp.ball_vel_in_paddle_frame``.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    if noise_cfg is None:
        noise_cfg = BallObsNoiseCfg()

    vel_b = _ball_vel_paddle_frame_gt(env, ball_cfg, robot_cfg)

    if noise_cfg.mode == "oracle":
        return vel_b

    if noise_cfg.mode == "d435i":
        return _apply_d435i_vel_noise(vel_b, noise_cfg.d435i, noise_cfg.noise_scale)

    if noise_cfg.mode == "ekf":
        raise NotImplementedError(
            "EKF noise mode not yet implemented — coming in a future iteration."
        )

    raise ValueError(f"Unknown noise mode: {noise_cfg.mode!r}")


# ---------------------------------------------------------------------------
# D435i noise model (structured, depth-dependent)
# ---------------------------------------------------------------------------

def _apply_d435i_pos_noise(
    pos_b: torch.Tensor,
    params: D435iNoiseParams,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply D435i-style structured noise to ball position in paddle frame.

    Noise model:
    - XY noise: Gaussian with std = sigma_xy_base (lateral pixel noise)
    - Z noise: Gaussian with std = sigma_z_base + sigma_z_per_metre * |z|
      (depth accuracy degrades with distance — D435i stereo baseline effect)
    - Dropout: with probability dropout_prob, return last known position
      (simulated as zeroing the update — full dropout buffer added with EKF)
    """
    device = pos_b.device
    N = pos_b.shape[0]

    if scale <= 0.0:
        return pos_b

    # XY noise (lateral, from pixel quantisation + IR pattern matching)
    xy_noise = torch.randn(N, 2, device=device) * (params.sigma_xy_base * scale)

    # Z noise (depth, distance-dependent)
    z_dist = pos_b[:, 2].abs()  # distance along z in paddle frame
    sigma_z = (params.sigma_z_base + params.sigma_z_per_metre * z_dist) * scale
    z_noise = torch.randn(N, device=device) * sigma_z

    noise = torch.stack([xy_noise[:, 0], xy_noise[:, 1], z_noise], dim=-1)

    # Dropout: zero the noise for dropped frames (position freezes at GT
    # for now; proper hold-last-value requires EKF state buffer)
    dropout_prob = params.dropout_prob * scale
    if dropout_prob > 0:
        dropout_mask = (
            torch.rand(N, device=device) < dropout_prob
        ).unsqueeze(-1)
        # On dropout, return GT (no noise) — conservative stub.
        # Full pipeline will hold last EKF estimate instead.
        noise = noise * (~dropout_mask).float()

    return pos_b + noise


def _apply_d435i_vel_noise(
    vel_b: torch.Tensor,
    params: D435iNoiseParams,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply noise to ball velocity estimate.

    In the real pipeline, velocity comes from the EKF (finite-differenced
    noisy positions). For the d435i stub, we approximate this as additive
    Gaussian noise scaled by the position noise parameters and a typical
    frame-to-frame dt (~33ms at 30Hz).

    The velocity noise std is approximately sigma_pos / dt, which at
    30Hz gives ~3x amplification of position noise.
    """
    device = vel_b.device
    N = vel_b.shape[0]

    # Approximate velocity noise from finite-differenced position noise
    # at 30Hz camera rate: sigma_vel ≈ sqrt(2) * sigma_pos / dt
    dt_camera = 1.0 / 30.0  # 30Hz D435i frame rate
    sigma_vel_xy = (2 ** 0.5) * params.sigma_xy_base / dt_camera
    sigma_vel_z = (2 ** 0.5) * params.sigma_z_base / dt_camera

    if scale <= 0.0:
        return vel_b

    noise = torch.zeros_like(vel_b)
    noise[:, :2] = torch.randn(N, 2, device=device) * (sigma_vel_xy * scale)
    noise[:, 2] = torch.randn(N, device=device) * (sigma_vel_z * scale)

    dropout_prob = params.dropout_prob * scale
    if dropout_prob > 0:
        dropout_mask = (
            torch.rand(N, device=device) < dropout_prob
        ).unsqueeze(-1)
        noise = noise * (~dropout_mask).float()

    return vel_b + noise
