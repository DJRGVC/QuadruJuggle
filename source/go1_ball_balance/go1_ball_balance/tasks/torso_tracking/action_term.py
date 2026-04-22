"""Custom ActionTerm that wraps a frozen pi2 (torso-tracking) policy.

Architecture:
  - pi1 (or mirror law) outputs a 6D juggling command:
        [h, h_dot, roll, pitch, omega_roll, omega_pitch]
  - The user (or a scripted controller) supplies a 3D locomotion command:
        [vx, vy, omega_yaw]  (body frame, read from env._user_cmd_vel)
  - This term concatenates them → 9D → runs frozen pi2 → 12D joint targets.

The 6D input is the ActionTerm interface used by the RL manager (so pi1
still trains / runs with 6D output). The user 3D is an external override
buffered on the env; during pi2 training the torso-tracking task samples
a full 9D command directly (no TorsoCommandAction used there).

Usage in a hierarchical env config:
    actions = TorsoCommandActionCfg(
        asset_name="robot",
        pi2_checkpoint="/path/to/model_best.pt",
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .mdp.observations import _NORM, _OFFSET, CMD_DIM

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Command dimension ranges for scaling [-1, 1] → physical units (9D).
_CMD_SCALES = torch.tensor([
    0.125,   # h: half-range (centre 0.375)
    1.0,     # h_dot
    0.4,     # roll
    0.4,     # pitch
    3.0,     # omega_roll
    3.0,     # omega_pitch
    0.5,     # vx
    0.5,     # vy
    1.5,     # omega_yaw
])

_CMD_OFFSETS = torch.tensor([
    0.375,   # h centre
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,     # vx symmetric
    0.0,     # vy symmetric
    0.0,     # omega_yaw symmetric
])

# pi1 drives only the first 6 dims (juggling channel).
# The remaining 3 dims (vx, vy, omega_yaw) are user-driven at play time.
_PI1_ACTION_DIM = 6
_USER_CMD_DIM = 3


def ensure_user_cmd_buffer(env) -> torch.Tensor:
    """Ensure env._user_cmd_vel exists (3D: vx, vy, omega_yaw, normalized [-1,1]).

    Called lazily by TorsoCommandAction on first process_actions. External
    controllers (keyboard, CLI args, policies) can write into this tensor
    directly.
    """
    if not hasattr(env, "_user_cmd_vel"):
        env._user_cmd_vel = torch.zeros(env.num_envs, _USER_CMD_DIM, device=env.device)
    return env._user_cmd_vel


class TorsoCommandAction(ActionTerm):
    """Action term that takes 6D juggling commands from pi1 and 3D user
    locomotion commands, concatenates to 9D, runs frozen pi2, and applies
    12D joint position targets.

    The 6D raw actions (from pi1) and the 3D user commands are each in
    [-1, 1] and are scaled to physical units before being fed as part of
    pi2's 42D observation vector (9 command + 33 proprio).
    """

    cfg: "TorsoCommandActionCfg"

    def __init__(self, cfg: "TorsoCommandActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        self._robot: Articulation = env.scene[cfg.asset_name]
        self._num_envs = env.num_envs
        self._device = env.device

        # Load frozen pi2 actor
        self._load_pi2(cfg.pi2_checkpoint)

        # Joint IDs (all 12)
        self._joint_ids, _ = self._robot.find_joints(".*")
        self._num_joints = len(self._joint_ids)

        # Action buffers — RL-facing is 6D (pi1 output dim).
        self._raw_actions = torch.zeros(self._num_envs, _PI1_ACTION_DIM, device=self._device)
        self._joint_targets = torch.zeros(self._num_envs, self._num_joints, device=self._device)

        # Default joint positions (for offset)
        self._default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()

        # Scaling tensors (9D)
        self._cmd_scales = _CMD_SCALES.to(self._device)
        self._cmd_offsets = _CMD_OFFSETS.to(self._device)
        self._obs_norm = _NORM.to(self._device)
        self._obs_offset = _OFFSET.to(self._device)

        # Ensure user command buffer exists (initialised to zero motion).
        ensure_user_cmd_buffer(env)

    def _load_pi2(self, checkpoint_path: str) -> None:
        """Load pi2 actor weights from checkpoint and freeze."""
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=True)

        # RSL-RL saves model state dict under 'model_state_dict'
        model_state = checkpoint.get("model_state_dict", checkpoint)

        # Extract actor weights — RSL-RL actor keys are 'actor.0.weight', 'actor.0.bias', etc.
        actor_keys = sorted([k for k in model_state if k.startswith("actor.")])

        # Determine layer dims from weights
        layers = []
        for key in actor_keys:
            if "weight" in key:
                out_dim, in_dim = model_state[key].shape
                layers.append((in_dim, out_dim))

        # Build actor MLP: linear → ELU → linear → ELU → ... → linear (no final activation)
        import torch.nn as nn
        modules = []
        for i, (in_dim, out_dim) in enumerate(layers):
            modules.append(nn.Linear(in_dim, out_dim))
            if i < len(layers) - 1:  # no activation after final layer
                modules.append(nn.ELU())

        self._pi2_actor = nn.Sequential(*modules).to(self._device)

        # Load weights
        actor_state = {}
        for key in actor_keys:
            # Map 'actor.X.weight' → sequential module index
            parts = key.split(".")
            original_idx = int(parts[1])
            param_type = parts[2]

            # In RSL-RL, actor layers are numbered 0,2,4... (skipping activation indices)
            # In our Sequential, linear layers are at 0,2,4... (alternating with ELU)
            seq_idx = original_idx  # RSL-RL already uses skip-2 indexing for Linear layers
            actor_state[f"{seq_idx}.{param_type}"] = model_state[key]

        self._pi2_actor.load_state_dict(actor_state)

        # Freeze all parameters
        for param in self._pi2_actor.parameters():
            param.requires_grad = False
        self._pi2_actor.eval()

        # Check if normalizer exists
        self._obs_normalizer = None
        if "actor_obs_normalizer.running_mean" in model_state:
            running_mean = model_state["actor_obs_normalizer.running_mean"]
            running_var = model_state["actor_obs_normalizer.running_var"]
            count = model_state.get("actor_obs_normalizer.count", torch.tensor(1.0))
            self._obs_normalizer = {
                "mean": running_mean.to(self._device),
                "var": running_var.to(self._device),
                "count": count,
            }

        # Record pi2 input dim so we can tell 6D vs 9D checkpoints apart.
        self._pi2_input_dim = layers[0][0] if layers else None

        print(f"[TorsoCommandAction] Loaded pi2 from {checkpoint_path}")
        print(f"  Actor architecture: {[f'{in_d}→{out_d}' for in_d, out_d in layers]}")
        print(f"  Normalizer: {'yes' if self._obs_normalizer else 'no'}")

    @property
    def action_dim(self) -> int:
        # RL-facing action dim is pi1's output dim (6D juggling channel).
        return _PI1_ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._joint_targets

    def process_actions(self, actions: torch.Tensor) -> None:
        """Convert 6D juggling + 3D user commands → 12D joint targets via frozen pi2.

        Called once per policy step (50 Hz).
        """
        self._raw_actions[:] = actions

        # Fetch user-driven vx/vy/omega_yaw (normalised to [-1, 1]).
        user_cmd = ensure_user_cmd_buffer(self._env)
        # Concatenate pi1's 6D + user's 3D → 9D normalised command.
        cmd_norm = torch.cat([actions, user_cmd], dim=-1)  # (N, 9)

        # Scale [-1, 1] → physical units.
        torso_cmd = cmd_norm * self._cmd_scales + self._cmd_offsets

        # Normalise back the way the observation pipeline does (training parity).
        torso_cmd_norm = (torso_cmd + self._obs_offset) * self._obs_norm

        # Proprioception.
        base_lin_vel = self._robot.data.root_lin_vel_b          # (N, 3)
        base_ang_vel = self._robot.data.root_ang_vel_b          # (N, 3)
        projected_gravity = self._robot.data.projected_gravity_b # (N, 3)
        joint_pos_rel = (
            self._robot.data.joint_pos[:, self._joint_ids]
            - self._robot.data.default_joint_pos[:, self._joint_ids]
        )
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]

        # Build pi2's observation vector (must match training order exactly).
        # 9D command + 33D proprio = 42D total.
        pi2_obs = torch.cat([
            torso_cmd_norm,    # 9
            base_lin_vel,      # 3
            base_ang_vel,      # 3
            projected_gravity, # 3
            joint_pos_rel,     # 12
            joint_vel,         # 12
        ], dim=-1)  # (N, 42)

        # Apply normalizer if available.
        if self._obs_normalizer is not None:
            mean = self._obs_normalizer["mean"]
            var = self._obs_normalizer["var"]
            pi2_obs = (pi2_obs - mean) / (var.sqrt() + 1e-8)

        # Run frozen pi2 actor (no grad).
        with torch.no_grad():
            pi2_actions = self._pi2_actor(pi2_obs)  # (N, 12)

        # Same scaling as JointPositionAction: target = default + scale * action.
        self._joint_targets[:] = self._default_joint_pos + 0.25 * pi2_actions

        # Cache the full 9D command on the env so viz / diagnostics can see it
        # (same buffer the torso-tracking training task uses).
        if not hasattr(self._env, "_torso_cmd"):
            self._env._torso_cmd = torch.zeros(self._num_envs, CMD_DIM, device=self._device)
        self._env._torso_cmd[:] = torso_cmd

    def apply_actions(self) -> None:
        """Apply joint position targets to robot (called every physics step)."""
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        # Clear user command so a fresh episode starts stationary.
        if hasattr(self._env, "_user_cmd_vel"):
            self._env._user_cmd_vel[env_ids] = 0.0


@configclass
class TorsoCommandActionCfg(ActionTermCfg):
    """Configuration for the TorsoCommandAction term."""

    class_type: type = TorsoCommandAction

    pi2_checkpoint: str = MISSING
    """Path to the trained pi2 (torso-tracking) checkpoint."""
