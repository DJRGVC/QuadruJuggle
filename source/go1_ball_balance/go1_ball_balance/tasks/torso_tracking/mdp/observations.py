"""Custom observation terms for the torso-tracking task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Normalization constants (map physical ranges to ~[-1, 1])
_NORM = torch.tensor([
    1.0 / 0.125,   # h:     centre 0.375, half-range 0.125  → /0.125
    1.0 / 1.0,     # h_dot: half-range 1.0
    1.0 / 0.4,     # roll
    1.0 / 0.4,     # pitch
    1.0 / 3.0,     # omega_roll
    1.0 / 3.0,     # omega_pitch
])

_OFFSET = torch.tensor([
    -0.375,  # h centre = (0.25+0.50)/2
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
])


def torso_command_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the 6D torso command, normalized to approximately [-1, 1].

    Reads env._torso_cmd (populated by resample_torso_commands event).

    Returns:
        Tensor of shape (num_envs, 6).
    """
    if not hasattr(env, "_torso_cmd"):
        return torch.zeros(env.num_envs, 6, device=env.device)

    norm = _NORM.to(env.device)
    offset = _OFFSET.to(env.device)
    return (env._torso_cmd + offset) * norm
