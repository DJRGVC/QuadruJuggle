"""8D torso command buffer and resampling event for the torso-tracking task.

The command vector (num_envs, 8) stores the target torso pose/velocity:
  [h_target, h_dot_target, roll_target, pitch_target,
   omega_roll, omega_pitch, vx_target, vy_target]

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

# Default full ranges (final stage — widest).  Curriculum narrows these at start.
_CMD_RANGES_DEFAULT = {
    "h":     (0.20, 0.50),    # m  (near-ground to fully extended)
    "h_dot": (-1.5, 1.5),     # m/s  (fast vertical for juggling)
    "roll":  (-0.5, 0.5),     # rad  (~29°)
    "pitch": (-0.5, 0.5),     # rad
    "omega_roll":  (-4.0, 4.0),   # rad/s
    "omega_pitch": (-4.0, 4.0),   # rad/s
    "vx":    (-0.5, 0.5),     # m/s  (body-frame forward velocity)
    "vy":    (-0.5, 0.5),     # m/s  (body-frame lateral velocity)
}

# Smoothing time constant (seconds).  Larger = slower transitions.
_SMOOTH_TAU = 0.4


def _ensure_cmd_buffer(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Lazily create env._torso_cmd if it doesn't exist yet."""
    if not hasattr(env, "_torso_cmd"):
        env._torso_cmd = torch.zeros(env.num_envs, 8, device=env.device)
        env._torso_cmd_goal = torch.zeros(env.num_envs, 8, device=env.device)
        env._torso_cmd_ranges = {k: list(v) for k, v in _CMD_RANGES_DEFAULT.items()}
        # Apply deferred vxy override from play.py --max-vxy flag
        if hasattr(env, "_max_vxy_override"):
            v = env._max_vxy_override
            env._torso_cmd_ranges["vx"] = [-v, v]
            env._torso_cmd_ranges["vy"] = [-v, v]
    return env._torso_cmd


def _sample_goals(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Sample new random goals into env._torso_cmd_goal."""
    _ensure_cmd_buffer(env)
    n = len(env_ids)
    ranges = env._torso_cmd_ranges
    keys = ["h", "h_dot", "roll", "pitch", "omega_roll", "omega_pitch", "vx", "vy"]
    for i, key in enumerate(keys):
        lo, hi = ranges[key]
        env._torso_cmd_goal[env_ids, i] = torch.empty(n, device=env.device).uniform_(lo, hi)


def resample_torso_commands(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Resample 8D torso commands for the given envs.

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


# ── Deterministic circle pattern for play mode ───────────────────────────────
# 20 waypoints, counter-clockwise, all envs get the same command.
# Each waypoint is held for 0.5 seconds (25 policy steps).
# Full circle = 20 × 0.5 = 10 s.
_CIRCLE_N_WAYPOINTS = 20
_CIRCLE_SPEED = 0.3          # m/s magnitude
_CIRCLE_HOLD_STEPS = 25      # policy steps per waypoint (0.5 s at 50 Hz)
_CIRCLE_DEFAULT_H = 0.38     # comfortable standing height
_CIRCLE_DEFAULT_HDOT = 0.0
_CIRCLE_DEFAULT_RP = 0.0
_CIRCLE_DEFAULT_OMEGA = 0.0


def update_circle_commands(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Deterministic circle velocity pattern for play visualization.

    All envs receive the same vx/vy command that rotates counter-clockwise
    through 20 evenly-spaced directions.  Height/tilt/rates are held constant.

    Only active when env._torso_circle_enabled is True (set by play config).
    Runs at 200 Hz but only updates the command every _CIRCLE_HOLD_STEPS
    policy steps.
    """
    if not getattr(env, "_torso_circle_enabled", False):
        return
    _ensure_cmd_buffer(env)

    # Count total policy steps (approximation: physics_steps / decimation)
    if not hasattr(env, "_circle_physics_count"):
        env._circle_physics_count = 0
    env._circle_physics_count += 1

    # Only update once per policy step (every decimation physics steps)
    decimation = getattr(env.cfg, "decimation", 4)
    if env._circle_physics_count % decimation != 0:
        return

    policy_step = env._circle_physics_count // decimation
    waypoint_idx = (policy_step // _CIRCLE_HOLD_STEPS) % _CIRCLE_N_WAYPOINTS
    angle = 2.0 * math.pi * waypoint_idx / _CIRCLE_N_WAYPOINTS

    vx = _CIRCLE_SPEED * math.cos(angle)
    vy = _CIRCLE_SPEED * math.sin(angle)

    # Set ALL envs to the same command
    env._torso_cmd[:, 0] = _CIRCLE_DEFAULT_H
    env._torso_cmd[:, 1] = _CIRCLE_DEFAULT_HDOT
    env._torso_cmd[:, 2] = _CIRCLE_DEFAULT_RP
    env._torso_cmd[:, 3] = _CIRCLE_DEFAULT_RP
    env._torso_cmd[:, 4] = _CIRCLE_DEFAULT_OMEGA
    env._torso_cmd[:, 5] = _CIRCLE_DEFAULT_OMEGA
    env._torso_cmd[:, 6] = vx
    env._torso_cmd[:, 7] = vy
    env._torso_cmd_goal[:] = env._torso_cmd

    # Log occasionally
    if policy_step % _CIRCLE_HOLD_STEPS == 0:
        print(
            f"[CIRCLE] waypoint {waypoint_idx}/{_CIRCLE_N_WAYPOINTS}  "
            f"angle={math.degrees(angle):.0f}°  "
            f"vx={vx:.3f} vy={vy:.3f} m/s"
        )


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
