"""Custom observation terms for the ball-juggle task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_apex_height_obs(
    env: ManagerBasedRLEnv,
    reward_term_name: str = "ball_apex_height",
    max_target_height: float = 1.00,
) -> torch.Tensor:
    """Returns the current target apex height, normalised to [0, 1].

    Reads the live ``target_height`` parameter from the ball_apex_height reward
    term so the policy can condition on the curriculum stage.  Without this, a
    policy trained at Stage A (target=0.10 m) has no signal to change behaviour
    when the curriculum advances to Stage B (target=0.20 m).

    Args:
        reward_term_name: Name of the apex-height reward term in RewardsCfg.
        max_target_height: Value at which the observation saturates to 1.0.
            Should match the highest curriculum stage target height.

    Returns:
        Tensor of shape (num_envs, 1) — normalised target height.
    """
    target_h = 0.10  # default (Stage A)
    if hasattr(env, "reward_manager"):
        for i, name in enumerate(env.reward_manager._term_names):
            if name == reward_term_name:
                target_h = float(
                    env.reward_manager._term_cfgs[i].params.get("target_height", target_h)
                )
                break
    norm = target_h / max(max_target_height, 1e-6)
    return torch.full((env.num_envs, 1), norm, device=env.device)
