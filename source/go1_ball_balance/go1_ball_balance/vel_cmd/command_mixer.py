"""CommandMixer — overrides vx/vy channels of pi1's 8D output with user input.

Indices in the 8D command tensor (from action_term.py):
    6 = vx (normalized, +/-1 maps to +/-0.5 m/s)
    7 = vy (normalized, +/-1 maps to +/-0.5 m/s)

Three blend modes:
  - "override"   : fully replace pi1's vx/vy with user input
  - "blend"      : alpha * pi1 + (1-alpha) * user
  - "passthrough": no modification (for ablation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class CommandMixerCfg:
    mode: Literal["override", "blend", "passthrough"] = "override"
    blend_alpha: float = 0.0  # 0.0 = full user; 1.0 = full pi1
    vx_idx: int = 6
    vy_idx: int = 7


class CommandMixer:
    """Mixes pi1's 8D output with user velocity commands."""

    def __init__(self, cfg: CommandMixerCfg):
        self.cfg = cfg

    def mix(
        self,
        pi1_cmd: torch.Tensor,   # (N, 8) in [-1, +1]
        vel_user: torch.Tensor,  # (N, 2) [vx_norm, vy_norm] in [-1, +1]
    ) -> torch.Tensor:
        """Return mixed command tensor (N, 8)."""
        if self.cfg.mode == "passthrough":
            return pi1_cmd

        out = pi1_cmd.clone()
        alpha = self.cfg.blend_alpha

        if self.cfg.mode == "override":
            alpha = 0.0

        out[:, self.cfg.vx_idx] = alpha * pi1_cmd[:, self.cfg.vx_idx] + (1.0 - alpha) * vel_user[:, 0]
        out[:, self.cfg.vy_idx] = alpha * pi1_cmd[:, self.cfg.vy_idx] + (1.0 - alpha) * vel_user[:, 1]

        return out
