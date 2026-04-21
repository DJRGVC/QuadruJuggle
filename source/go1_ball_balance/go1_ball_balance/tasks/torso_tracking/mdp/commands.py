"""9D torso command buffer and resampling event for the torso-tracking task.

The command vector (num_envs, 9) stores the target torso pose/velocity:
  [h, h_dot, roll, pitch, omega_roll, omega_pitch, vx, vy, omega_yaw]

Commands are resampled uniformly within per-dimension ranges that widen
with curriculum stage (narrow → full).

Smooth mode (play only): resample sets a *goal*; update_torso_commands_smooth()
blends the active command toward the goal each physics step (exp smoothing).
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from .observations import CMD_DIM

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Command key order — must match observations._NORM / action_term._CMD_SCALES.
_CMD_KEYS = [
    "h", "h_dot", "roll", "pitch",
    "omega_roll", "omega_pitch",
    "vx", "vy", "omega_yaw",
]

# Default full ranges (Stage D — widest).  Curriculum narrows these at start.
_CMD_RANGES_DEFAULT = {
    "h":     (0.25, 0.50),    # m  (wider for high bounces)
    "h_dot": (-1.0, 1.0),     # m/s  (wider for active juggling)
    "roll":  (-0.4, 0.4),     # rad
    "pitch": (-0.4, 0.4),     # rad
    "omega_roll":  (0.0, 0.0),   # rad/s (curriculum may widen)
    "omega_pitch": (0.0, 0.0),   # rad/s
    "vx":    (-0.5, 0.5),     # m/s (body-frame forward)
    "vy":    (-0.5, 0.5),     # m/s (body-frame lateral)
    "omega_yaw":  (-1.5, 1.5),   # rad/s (body-frame yaw rate)
}

# Per-dimension smoothing time constants — one per CMD_KEY.
# Larger tau = heavier filter = slower transitions.
# h/h_dot use heavy filtering (leg compliance is slow to respond).
# roll/pitch/rates use light filtering (ball juggling needs fast tilt response).
# vx/vy/yaw: moderate filtering so velocity commands feel responsive but stable.
_SMOOTH_TAUS = [
    0.40,   # h
    0.30,   # h_dot
    0.02,   # roll
    0.02,   # pitch
    0.01,   # omega_roll
    0.01,   # omega_pitch
    0.25,   # vx
    0.25,   # vy
    0.20,   # omega_yaw
]


def _ensure_cmd_buffer(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Lazily create env._torso_cmd if it doesn't exist yet."""
    if not hasattr(env, "_torso_cmd"):
        env._torso_cmd = torch.zeros(env.num_envs, CMD_DIM, device=env.device)
        env._torso_cmd_goal = torch.zeros(env.num_envs, CMD_DIM, device=env.device)
        env._torso_cmd_ranges = {k: list(v) for k, v in _CMD_RANGES_DEFAULT.items()}
    return env._torso_cmd


def _sample_goals(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Sample new random goals into env._torso_cmd_goal."""
    _ensure_cmd_buffer(env)
    n = len(env_ids)
    ranges = env._torso_cmd_ranges
    for i, key in enumerate(_CMD_KEYS):
        lo, hi = ranges[key]
        env._torso_cmd_goal[env_ids, i] = torch.empty(n, device=env.device).uniform_(lo, hi)


def resample_torso_commands(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Resample 9D torso commands for the given envs.

    In training mode (no smooth): snaps active command to new sample.
    In play smooth mode: only updates the goal; the smooth blender
    interpolates the active command toward it.
    """
    _sample_goals(env, env_ids)

    # In training, snap immediately (no smooth interpolation)
    if not getattr(env, "_torso_smooth_enabled", False):
        env._torso_cmd[env_ids] = env._torso_cmd_goal[env_ids]


def resample_torso_commands_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Resample on reset — always snaps to goal (even in smooth mode)."""
    _sample_goals(env, env_ids)
    env._torso_cmd[env_ids] = env._torso_cmd_goal[env_ids]


def update_torso_commands_smooth(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Exponentially blend active commands toward goals (every physics step).

    alpha = 1 - exp(-dt / tau), so ~63% of the way in tau seconds.
    Only active when env._torso_smooth_enabled is True (set by play config).
    """
    if not getattr(env, "_torso_smooth_enabled", False):
        return
    if not hasattr(env, "_torso_cmd"):
        return

    dt = env.cfg.sim.dt  # 1/200 = 0.005s
    alphas = torch.tensor(
        [1.0 - math.exp(-dt / tau) for tau in _SMOOTH_TAUS],
        device=env.device,
    )  # (CMD_DIM,) — one blend rate per dimension
    env._torso_cmd[:] = env._torso_cmd + alphas * (env._torso_cmd_goal - env._torso_cmd)
