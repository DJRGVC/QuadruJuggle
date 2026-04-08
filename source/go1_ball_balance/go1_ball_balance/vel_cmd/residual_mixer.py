"""ResidualMixer — adds user velocity to pi1's residual vx/vy output.

Method 2 (Residual Velocity) from vel-cmd-survey final proposal.
Based on HiLMa-Res (Shi et al., CoRL 2024) and Walk These Ways
(Margolis & Agrawal, CoRL 2023).

pi1 outputs an 8D command where indices 6,7 are RESIDUAL velocity corrections.
Final velocity = user_base + pi1_residual, clamped to [-max_total_norm, +max_total_norm].

Unlike CommandMixer (Method 1) which discards pi1's vx/vy entirely,
ResidualMixer lets pi1 compensate for ball-induced disturbances relative
to the user's base velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ResidualMixerCfg:
    vx_idx: int = 6
    vy_idx: int = 7
    # Max total normalized velocity magnitude per axis (1.0 = ±0.5 m/s at pi2 scale).
    max_total_norm: float = 1.0


class ResidualMixer:
    """Adds user base velocity to pi1's residual vx/vy output."""

    def __init__(self, cfg: ResidualMixerCfg | None = None):
        self.cfg = cfg or ResidualMixerCfg()

    def mix(
        self,
        pi1_cmd: torch.Tensor,   # (N, 8) pi1 output; indices 6,7 = residual vx/vy
        vel_user: torch.Tensor,  # (N, 2) [vx_u_norm, vy_u_norm] in [-1, +1]
    ) -> torch.Tensor:
        """Return command with vx/vy = clamp(pi1_residual + vel_user, -max, +max).

        All other command channels (0-5) are passed through unchanged.
        """
        out = pi1_cmd.clone()
        cap = self.cfg.max_total_norm
        out[:, self.cfg.vx_idx] = torch.clamp(
            pi1_cmd[:, self.cfg.vx_idx] + vel_user[:, 0], -cap, cap
        )
        out[:, self.cfg.vy_idx] = torch.clamp(
            pi1_cmd[:, self.cfg.vy_idx] + vel_user[:, 1], -cap, cap
        )
        return out
