"""6D torso command buffer and resampling event for the torso-tracking task.

The command vector (num_envs, 6) stores the target torso pose/velocity:
  [h_target, h_dot_target, roll_target, pitch_target, omega_roll, omega_pitch]

Commands are resampled uniformly within per-dimension ranges that widen
with curriculum stage (narrow → full).

Smooth mode (play only): resample sets a *goal*; update_torso_commands_smooth()
blends the active command toward the goal each physics step (exp smoothing).
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Default full ranges (Stage C — widest).  Curriculum narrows these at start.
_CMD_RANGES_DEFAULT = {
    "h":     (0.25, 0.50),    # m  (wider for high bounces)
    "h_dot": (-1.0, 1.0),     # m/s  (wider for active juggling)
    "roll":  (-0.4, 0.4),     # rad
    "pitch": (-0.4, 0.4),     # rad
    "omega_roll":  (-3.0, 3.0),   # rad/s
    "omega_pitch": (-3.0, 3.0),   # rad/s
}

# Smoothing time constant (seconds).  Larger = slower transitions.
_SMOOTH_TAU = 0.15


def _ensure_cmd_buffer(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Lazily create env._torso_cmd if it doesn't exist yet."""
    if not hasattr(env, "_torso_cmd"):
        env._torso_cmd = torch.zeros(env.num_envs, 6, device=env.device)
        env._torso_cmd_goal = torch.zeros(env.num_envs, 6, device=env.device)
        env._torso_cmd_ranges = {k: list(v) for k, v in _CMD_RANGES_DEFAULT.items()}
    return env._torso_cmd


def _sample_goals(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Sample new random goals into env._torso_cmd_goal."""
    _ensure_cmd_buffer(env)
    n = len(env_ids)
    ranges = env._torso_cmd_ranges
    keys = ["h", "h_dot", "roll", "pitch", "omega_roll", "omega_pitch"]
    for i, key in enumerate(keys):
        lo, hi = ranges[key]
        env._torso_cmd_goal[env_ids, i] = torch.empty(n, device=env.device).uniform_(lo, hi)


def resample_torso_commands(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Resample 6D torso commands for the given envs.

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
    alpha = 1.0 - math.exp(-dt / _SMOOTH_TAU)
    env._torso_cmd[:] = env._torso_cmd + alpha * (env._torso_cmd_goal - env._torso_cmd)
