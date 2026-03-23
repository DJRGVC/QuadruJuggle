"""Hier-specific event terms — apex height randomisation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_apex_height(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    height_range: tuple[float, float] = (0.10, 0.60),
    std_range: tuple[float, float] = (0.08, 0.12),
) -> None:
    """Randomise the target apex height per environment on reset.

    Sets ``env._apex_target_h`` (N,) and ``env._apex_target_std`` (N,) which are
    read by ball_apex_height_reward and target_apex_height_obs.

    Args:
        height_range: (min, max) target apex height above paddle [m].
        std_range:    (min, max) Gaussian std for the reward kernel [m].
                      Wider std is easier; tighten as height increases.
    """
    n = env.num_envs
    device = env.device

    if not hasattr(env, "_apex_target_h"):
        env._apex_target_h   = torch.full((n,), height_range[0], device=device)
        env._apex_target_std = torch.full((n,), std_range[1],    device=device)

    lo, hi = height_range
    env._apex_target_h[env_ids] = (
        torch.rand(len(env_ids), device=device) * (hi - lo) + lo
    )

    # Scale std: tighter at higher targets (harder), looser at lower (easier)
    t = (env._apex_target_h[env_ids] - lo) / max(hi - lo, 1e-6)  # 0→1
    std_lo, std_hi = std_range
    env._apex_target_std[env_ids] = std_hi - t * (std_hi - std_lo)
