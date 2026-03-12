"""Custom ActionTerm that wraps a frozen pi2 (torso-tracking) policy.

pi1 outputs 8D torso commands → this term constructs pi2's 41D observation
vector, runs the frozen pi2 actor MLP, and applies the resulting 12D joint
position targets to the robot.

Usage in an env config:
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

from .mdp.observations import _NORM, _OFFSET

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Command dimension ranges for scaling [-1, 1] → physical units
_CMD_SCALES = torch.tensor([
    0.15,    # h: half-range (centre 0.35)
    1.5,     # h_dot
    0.5,     # roll
    0.5,     # pitch
    4.0,     # omega_roll
    4.0,     # omega_pitch
    0.5,     # vx: half-range 0.5
    0.5,     # vy: half-range 0.5
])

_CMD_OFFSETS = torch.tensor([
    0.35,    # h centre = (0.20+0.50)/2
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,     # vx symmetric
    0.0,     # vy symmetric
])


class TorsoCommandAction(ActionTerm):
    """Action term that takes 8D torso commands from pi1, runs frozen pi2, and
    applies 12D joint position targets.

    The 8D input actions (from pi1) are in [-1, 1] and are scaled to physical
    units before being fed as part of pi2's 41D observation vector.
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

        # Action buffers
        self._raw_actions = torch.zeros(self._num_envs, 8, device=self._device)
        self._joint_targets = torch.zeros(self._num_envs, self._num_joints, device=self._device)

        # Default joint positions (for offset)
        self._default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()

        # Paddle reference for rigid attachment (if paddle exists in scene)
        self._paddle = None
        self._paddle_offset_b = None
        try:
            self._paddle = env.scene["paddle"]
            self._paddle_offset_b = torch.tensor(
                [0.0, 0.0, 0.070], device=self._device, dtype=torch.float32,
            )
        except KeyError:
            pass  # no paddle in scene (e.g., torso-tracking task)

        # Scaling tensors
        self._cmd_scales = _CMD_SCALES.to(self._device)
        self._cmd_offsets = _CMD_OFFSETS.to(self._device)
        self._obs_norm = _NORM.to(self._device)
        self._obs_offset = _OFFSET.to(self._device)

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
        layer_idx = 0
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

        print(f"[TorsoCommandAction] Loaded pi2 from {checkpoint_path}")
        print(f"  Actor architecture: {[f'{in_d}→{out_d}' for in_d, out_d in layers]}")
        print(f"  Normalizer: {'yes' if self._obs_normalizer else 'no'}")

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._joint_targets

    def process_actions(self, actions: torch.Tensor) -> None:
        """Convert 8D commands → 12D joint targets via frozen pi2.

        Called once per policy step (50 Hz).
        """
        self._raw_actions[:] = actions

        # Scale [-1, 1] → physical units
        torso_cmd = actions * self._cmd_scales + self._cmd_offsets

        # Build pi2's 41D observation vector (must match training order exactly):
        # [torso_command_normalized(8), base_lin_vel(3), base_ang_vel(3),
        #  projected_gravity(3), joint_pos_rel(12), joint_vel_rel(12)]

        # Normalize torso command (same as torso_command_obs)
        torso_cmd_norm = (torso_cmd + self._obs_offset) * self._obs_norm

        # Proprioception
        base_lin_vel = self._robot.data.root_lin_vel_b          # (N, 3)
        base_ang_vel = self._robot.data.root_ang_vel_b          # (N, 3)
        projected_gravity = self._robot.data.projected_gravity_b # (N, 3)
        joint_pos_rel = (
            self._robot.data.joint_pos[:, self._joint_ids]
            - self._robot.data.default_joint_pos[:, self._joint_ids]
        )
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]

        # Concatenate in training order
        pi2_obs = torch.cat([
            torso_cmd_norm,    # 8
            base_lin_vel,      # 3
            base_ang_vel,      # 3
            projected_gravity, # 3
            joint_pos_rel,     # 12
            joint_vel,         # 12
        ], dim=-1)  # (N, 41)

        # Apply normalizer if available
        if self._obs_normalizer is not None:
            mean = self._obs_normalizer["mean"]
            var = self._obs_normalizer["var"]
            pi2_obs = (pi2_obs - mean) / (var.sqrt() + 1e-8)

        # Run frozen pi2 actor (no grad)
        with torch.no_grad():
            pi2_actions = self._pi2_actor(pi2_obs)  # (N, 12)

        # Apply same scaling as JointPositionAction: target = default + scale * action
        self._joint_targets[:] = self._default_joint_pos + 0.25 * pi2_actions

    def apply_actions(self) -> None:
        """Apply joint position targets to robot (called every physics step).

        Also updates the kinematic paddle pose to ensure rigid attachment
        with zero visual lag (belt-and-suspenders with the interval event).
        """
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

        if self._paddle is not None:
            trunk_pos_w = self._robot.data.root_pos_w
            trunk_quat_w = self._robot.data.root_quat_w
            off_w = math_utils.quat_apply(
                trunk_quat_w,
                self._paddle_offset_b.unsqueeze(0).expand(self._num_envs, -1),
            )
            paddle_pos_w = trunk_pos_w + off_w
            pose = torch.cat([paddle_pos_w, trunk_quat_w], dim=-1)
            self._paddle.write_root_pose_to_sim(pose)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class TorsoCommandActionCfg(ActionTermCfg):
    """Configuration for the TorsoCommandAction term."""

    class_type: type = TorsoCommandAction

    pi2_checkpoint: str = MISSING
    """Path to the trained pi2 (torso-tracking) checkpoint."""
