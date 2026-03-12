"""Custom observation terms for the torso-tracking task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Normalization constants (map physical ranges to ~[-1, 1])
_NORM = torch.tensor([
    1.0 / 0.15,    # h:     centre 0.35, half-range 0.15  → /0.15
    1.0 / 1.5,     # h_dot: half-range 1.5
    1.0 / 0.5,     # roll
    1.0 / 0.5,     # pitch
    1.0 / 4.0,     # omega_roll
    1.0 / 4.0,     # omega_pitch
    1.0 / 0.5,     # vx: half-range 0.5
    1.0 / 0.5,     # vy: half-range 0.5
])

_OFFSET = torch.tensor([
    -0.35,   # h centre = (0.20+0.50)/2
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,     # vx symmetric
    0.0,     # vy symmetric
])


def torso_command_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the 8D torso command, normalized to approximately [-1, 1].

    Reads env._torso_cmd (populated by resample_torso_commands event).

    Returns:
        Tensor of shape (num_envs, 8).
    """
    if not hasattr(env, "_torso_cmd"):
        return torch.zeros(env.num_envs, 8, device=env.device)

    norm = _NORM.to(env.device)
    offset = _OFFSET.to(env.device)
    return (env._torso_cmd + offset) * norm
