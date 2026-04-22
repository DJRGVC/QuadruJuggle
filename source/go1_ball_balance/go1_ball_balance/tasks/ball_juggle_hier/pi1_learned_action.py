"""Learned pi1 action term: RL policy outputs 6D normalized torso command → frozen pi2.

The RL policy directly outputs the normalized 6D torso command in the same
space that TorsoCommandAction expects:
    [h_norm, h_dot_norm, roll_norm, pitch_norm, omega_roll_norm, omega_pitch_norm]
each in approximately [-1, 1].

Physical mapping (from TorsoCommandAction._CMD_SCALES / _OFFSETS):
    h         = h_norm * 0.125 + 0.375   → [0.25, 0.50] m
    h_dot     = h_dot_norm * 1.0         → [-1, 1]
    roll      = roll_norm * 0.4          → [-0.4, 0.4] rad
    pitch     = pitch_norm * 0.4         → [-0.4, 0.4] rad
    omega_roll  = omega_roll_norm * 3.0  → [-3, 3] rad/s
    omega_pitch = omega_pitch_norm * 3.0 → [-3, 3] rad/s

No mirror-law math. No hyperparameters (except pi2 checkpoint).
The RL agent learns to produce these commands directly.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from go1_ball_balance.tasks.torso_tracking.action_term import TorsoCommandAction, TorsoCommandActionCfg


class LearnedPi1TorsoAction(TorsoCommandAction):
    """Learned pi1: RL policy 6D normalized output → frozen pi2 → 12D joints.

    Inherits all pi2 loading and inference from TorsoCommandAction.
    The only difference: action_dim=6 (RL directly controls all 6 torso
    command dimensions instead of going through mirror-law geometry).
    """

    cfg: "LearnedPi1TorsoActionCfg"

    @property
    def action_dim(self) -> int:
        """6D: [h, h_dot, roll, pitch, omega_roll, omega_pitch] normalized."""
        return 6

    def process_actions(self, actions) -> None:
        """Pass 6D normalized RL output directly to frozen pi2.

        Args:
            actions: (N, 6) tensor, each dim nominally in [-1, 1].
        """
        if self.cfg.clip_actions:
            actions = actions.clamp(-1.0, 1.0)
        super().process_actions(actions)


@configclass
class LearnedPi1TorsoActionCfg(TorsoCommandActionCfg):
    """Configuration for LearnedPi1TorsoAction."""

    class_type: type = LearnedPi1TorsoAction

    clip_actions: bool = True
    """Clamp pi1's raw 6D output to [-1, 1] before scaling to physical torso commands.

    PPO's Gaussian action head is unbounded, and Frank's launcher pi1 converged to
    mean actions with magnitude 5-10× beyond [-1, 1] (reward optimum in sim treats
    saturated physical commands as equivalent). At play/eval time those extreme
    values scale to physically impossible torso targets (e.g. h=0.8m vs pi2's
    trained [0.25, 0.50] range), which crowds out user-command tracking.

    Clamping here maps every raw output to the same physical command it already
    produced in sim (post-saturation), and keeps the user's 3D locomotion slice
    in the distribution pi2 knows how to track. Set False to reproduce legacy
    launcher behaviour exactly."""
