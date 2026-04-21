"""Mirror-law high-level controller (pi1) → frozen pi2 → joint targets.

Mirror law math:
  Given ball incoming velocity v_in and desired outgoing velocity v_out,
  the paddle normal that achieves this reflection is:
      n = normalize(v_out - v_in)

  Desired outgoing velocity is computed from the target apex height:
      v_out_z = sqrt(2 * g * apex_height)   [upward]
      v_out_xy = -K * ball_xy_rel_paddle    [centering correction]

  The ball restitution coefficient scales the required v_out.

  Roll / pitch of the trunk are extracted from the desired paddle normal
  rotated into the robot body frame:
      pitch = atan2(-nx_body, nz_body)
      roll  = atan2( ny_body, nz_body)

Integration:
  Replace TorsoCommandActionCfg with MirrorLawTorsoActionCfg in any
  env that already uses the ball_juggle_hier scene.  The frozen pi2
  checkpoint path is still required.

  action_dim = 1: the single input scalar is the normalised target
  apex height [0, 1] → [apex_height_min, apex_height_max] metres.
  Feed torch.ones(N, 1) during play to use the configured default.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING, field

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .action_term import TorsoCommandAction, TorsoCommandActionCfg
from .ball_kalman_filter import BallKalmanFilter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_GRAVITY = 9.81  # m/s²

# Physical command ranges (must match TorsoCommandAction._CMD_SCALES / _OFFSETS)
_CMD_SCALES_PHYS = torch.tensor([0.125, 1.0, 0.4, 0.4, 3.0, 3.0])
_CMD_OFFSETS_PHYS = torch.tensor([0.375, 0.0, 0.0, 0.0, 0.0, 0.0])


class MirrorLawTorsoAction(TorsoCommandAction):
    """High-level mirror-law controller → frozen pi2 → 12D joint targets.

    Replaces the RL pi1 with closed-form mirror-law geometry.  The only
    learnable (or hand-tuned) input is the apex height target (1D).

    Useful for:
      - Play / evaluation without training pi1
      - Initialising hierarchical training with a strong prior
      - Debugging the torso-tracking (pi2) policy in a juggling context
    """

    cfg: "MirrorLawTorsoActionCfg"

    def __init__(self, cfg: "MirrorLawTorsoActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # Separate 1D buffer for the apex-height input (parent keeps its 6D _raw_actions)
        self._apex_input = torch.zeros(self._num_envs, 1, device=self._device)

        # Ball and paddle
        self._ball: RigidObject = env.scene[cfg.ball_cfg.name]
        self._paddle_offset_b = torch.tensor(
            cfg.paddle_offset_b, dtype=torch.float32, device=self._device
        )

        # Mirror-law hyper-parameters
        self._apex_min = cfg.apex_height_min
        self._apex_max = cfg.apex_height_max
        self._h_nominal = cfg.h_nominal
        self._centering_gain = cfg.centering_gain
        self._restitution = cfg.restitution

        # Perception noise (std dev in metres / m/s) — 0 = perfect state
        self._ball_pos_noise_std = cfg.ball_pos_noise_std
        self._ball_vel_noise_std = cfg.ball_vel_noise_std

        # Kalman filter for ball state estimation
        self._use_kalman = cfg.use_kalman and (cfg.ball_pos_noise_std > 0.0)
        if self._use_kalman:
            self._kf = BallKalmanFilter(
                num_envs=self._num_envs,
                dt=env.step_dt,
                device=self._device,
                pos_noise_std=cfg.ball_pos_noise_std,
                process_vel_std=cfg.kalman_process_vel_std,
            )

        # Impact tilt gain (>1 amplifies roll/pitch during impact for stronger bounce)
        self._impact_tilt_gain = cfg.impact_tilt_gain

        # Command EMA smoothing (reduces shaking from noisy velocity inputs).
        # alpha=1.0 → no smoothing (default); alpha<1 → exponential moving average.
        self._cmd_alpha = cfg.cmd_smooth_alpha
        N = self._num_envs
        self._ema_roll  = torch.zeros(N, device=self._device)
        self._ema_pitch = torch.zeros(N, device=self._device)
        self._ema_hdot  = torch.full((N,), 0.15, device=self._device)

        # Pre-computed inverse scaling (physical → normalised for pi2 input)
        self._phys_scales = _CMD_SCALES_PHYS.to(self._device)
        self._phys_offsets = _CMD_OFFSETS_PHYS.to(self._device)

    # ------------------------------------------------------------------
    # ActionTerm API
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        """1D: normalised apex-height target in [0, 1]."""
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        """Return the 1D apex-height input (not the 6D parent buffer)."""
        return self._apex_input

    def process_actions(self, actions: torch.Tensor) -> None:
        """Compute mirror-law 6D torso command and run frozen pi2.

        Args:
            actions: (N, 1) normalised apex height, typically in [0, 1].
        """
        self._apex_input[:] = actions

        # ── Apex height ────────────────────────────────────────────────
        apex_norm = actions[:, 0].clamp(0.0, 1.0)  # (N,)
        apex = self._apex_min + apex_norm * (self._apex_max - self._apex_min)

        # ── Ball state in world frame (with optional perception noise) ─
        ball_pos_w = self._ball.data.root_pos_w.clone()
        ball_vel_w = self._ball.data.root_lin_vel_w.clone()
        if self._ball_pos_noise_std > 0.0:
            ball_pos_noisy = ball_pos_w + torch.randn_like(ball_pos_w) * self._ball_pos_noise_std
        else:
            ball_pos_noisy = ball_pos_w
        if self._use_kalman:
            ball_pos_kf, ball_vel_kf = self._kf.step(ball_pos_noisy)
            ball_pos_w = ball_pos_kf
            if self._ball_vel_noise_std > 0.0:
                # No direct velocity sensor — use KF estimate (physics model
                # smooths the step-to-step jumps that raw noise causes).
                ball_vel_w = ball_vel_kf
            # else: ball_vel_noise_std == 0 → keep GT velocity; KF-derived
            # velocity (from differencing noisy positions) would be worse.
        else:
            ball_pos_w = ball_pos_noisy
            if self._ball_vel_noise_std > 0.0:
                ball_vel_w += torch.randn_like(ball_vel_w) * self._ball_vel_noise_std

        # ── Paddle centre in world frame ───────────────────────────────
        robot_pos_w  = self._robot.data.root_pos_w    # (N, 3)
        robot_quat_w = self._robot.data.root_quat_w   # (N, 4) wxyz
        offset_b = self._paddle_offset_b.unsqueeze(0).expand(self._num_envs, -1)
        paddle_pos_w = robot_pos_w + math_utils.quat_apply(robot_quat_w, offset_b)

        # Ball position relative to paddle (world frame)
        p_rel_w = ball_pos_w - paddle_pos_w  # (N, 3)

        # ── Desired outgoing velocity (world frame) ────────────────────
        # Vertical speed needed to reach apex above paddle
        v_out_z = (2.0 * _GRAVITY * apex).sqrt().clamp(min=0.5)  # (N,) ≥ 0.5 m/s

        # Lateral: steer ball back toward paddle centre
        v_out_x = -self._centering_gain * p_rel_w[:, 0]
        v_out_y = -self._centering_gain * p_rel_w[:, 1]

        v_out_w = torch.stack([v_out_x, v_out_y, v_out_z], dim=-1)  # (N, 3)

        # ── Mirror law: paddle normal ──────────────────────────────────
        # For inelastic impact with restitution e:
        #   v_out = e * n * (v_in · n) ... actually for paddle normal we need:
        #   effective v_out required from paddle = v_out_desired / restitution
        v_in_w = ball_vel_w
        v_out_eff = v_out_w / max(self._restitution, 0.1)

        # Paddle normal bisects incoming and outgoing directions
        n_raw = v_out_eff - v_in_w  # (N, 3)
        n_w = F.normalize(n_raw, dim=-1)  # unit normal in world frame

        # Ensure normal points upward
        flip = (n_w[:, 2] < 0).unsqueeze(-1).float()
        n_w = n_w * (1.0 - 2.0 * flip)

        # ── Rotate normal into robot body frame ────────────────────────
        quat_w2b = math_utils.quat_conjugate(robot_quat_w)
        n_b = math_utils.quat_apply(quat_w2b, n_w)  # (N, 3)

        nx_b, ny_b, nz_b = n_b[:, 0], n_b[:, 1], n_b[:, 2]

        # ── Extract roll and pitch from body-frame normal ──────────────
        # Clamp denominator to avoid atan2(0, 0) singularity near vertical
        nz_safe = nz_b.clamp(min=0.15)
        # Body +z axis = (cos r * sin p, -sin r, cos r * cos p), so:
        #   pitch = atan2( nx_b, nz_b)   (no negation on nx)
        #   roll  = atan2(-ny_b, nz_b)   (negation on ny)
        pitch_tgt = torch.atan2( nx_b, nz_safe)
        roll_tgt  = torch.atan2(-ny_b, nz_safe)

        # Clip to command range limits
        pitch_tgt = pitch_tgt.clamp(-0.4, 0.4)
        roll_tgt  = roll_tgt.clamp(-0.4, 0.4)

        # ── Impact tilt boost: amplify roll/pitch during impact phase ──
        # The mirror law computes the *correct direction* for the normal but
        # the resulting tilt angle is often small (ball nearly overhead).
        # Amplifying it during impact makes the paddle surface move diagonally,
        # injecting more energy into the ball — the same mechanism learned pi1
        # discovers through RL.  impact_tilt_gain=1.0 = no change (default).
        if self._impact_tilt_gain > 1.0:
            impact_phase = (ball_vel_w[:, 2] < 0.0).float() * (p_rel_w[:, 2] < 0.50).float()
            gain = 1.0 + (self._impact_tilt_gain - 1.0) * impact_phase
            pitch_tgt = (pitch_tgt * gain).clamp(-0.4, 0.4)
            roll_tgt  = (roll_tgt  * gain).clamp(-0.4, 0.4)

        # ── EMA smoothing on roll / pitch ──────────────────────────────
        if self._cmd_alpha < 1.0:
            self._ema_roll  = self._cmd_alpha * roll_tgt  + (1.0 - self._cmd_alpha) * self._ema_roll
            self._ema_pitch = self._cmd_alpha * pitch_tgt + (1.0 - self._cmd_alpha) * self._ema_pitch
            roll_tgt  = self._ema_roll
            pitch_tgt = self._ema_pitch

        # ── Trunk height and velocity ──────────────────────────────────
        h_cmd = torch.full((self._num_envs,), self._h_nominal, device=self._device)

        # h_dot: pulse upward when ball is near the paddle and descending.
        # 1D impact: v_out = e*|v_in| + v_paddle*(1+e)
        # → v_paddle = (v_out_target - e*|v_in_z|) / (1+e)
        # Negative when incoming speed is already sufficient (clamp to 0 → stationary paddle,
        # restitution alone reduces height). Positive when energy needs to be added.
        # Pi2's h_dot tracking is ~34-50%, so we command 2× the physical target.
        # Compliant legs need ~0.2s of lead time to build upward velocity before impact.
        # Wider 0.50m window (vs old 0.25m) gives that lead time at typical fall speeds.
        ball_rel_z = p_rel_w[:, 2]          # positive = ball above paddle
        ball_descending = (ball_vel_w[:, 2] < 0.0).float()
        near_impact     = (ball_rel_z < 0.50).float()   # within 50 cm above paddle
        v_in_z_abs = ball_vel_w[:, 2].abs()
        v_paddle_target = (v_out_z - self._restitution * v_in_z_abs) / (1.0 + self._restitution)
        # Clamp to physical max first, then apply 2× tracking-lag boost.
        # Clamping *before* the boost preserves proportionality across apex heights
        # (previously: 2× caused saturation at apex > ~0.05 m, making all heights identical).
        _H_DOT_PHYS_MAX = 1.0  # m/s — training range max
        _TRACKING_RATIO = 0.5  # pi2 achieves ~50% of commanded h_dot
        v_paddle_cmd = (v_paddle_target / _TRACKING_RATIO).clamp(0.0, _H_DOT_PHYS_MAX)
        h_dot_impulse = (v_paddle_cmd * ball_descending * near_impact).clamp(0.0, 1.0)
        # Small baseline when NOT in impact phase keeps body from drooping on springy legs.
        not_impacting = 1.0 - (ball_descending * near_impact).clamp(0.0, 1.0)
        h_dot_cmd = (h_dot_impulse + 0.15 * not_impacting).clamp(0.0, 1.0)

        # ── EMA smoothing on h_dot ─────────────────────────────────────
        if self._cmd_alpha < 1.0:
            self._ema_hdot = self._cmd_alpha * h_dot_cmd + (1.0 - self._cmd_alpha) * self._ema_hdot
            h_dot_cmd = self._ema_hdot

        # ── Angular rate commands ──────────────────────────────────────
        # Set to zero; pi2 handles rate from angle error tracking
        omega_roll  = torch.zeros(self._num_envs, device=self._device)
        omega_pitch = torch.zeros(self._num_envs, device=self._device)

        # ── Assemble 6D command in physical units ──────────────────────
        cmd_phys = torch.stack(
            [h_cmd, h_dot_cmd, roll_tgt, pitch_tgt, omega_roll, omega_pitch],
            dim=-1,
        )  # (N, 6)

        # ── Normalise to [-1, 1] (inverse of TorsoCommandAction scaling) ──
        cmd_norm = (cmd_phys - self._phys_offsets) / self._phys_scales

        # ── Run frozen pi2 via parent ──────────────────────────────────
        super().process_actions(cmd_norm)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._apex_input[:] = 0.0
            self._ema_roll[:]  = 0.0
            self._ema_pitch[:] = 0.0
            self._ema_hdot[:]  = 0.15
            ids = torch.arange(self._num_envs, device=self._device)
        else:
            self._apex_input[env_ids] = 0.0
            self._ema_roll[env_ids]   = 0.0
            self._ema_pitch[env_ids]  = 0.0
            self._ema_hdot[env_ids]   = 0.15
            ids = torch.tensor(env_ids, device=self._device) if not isinstance(env_ids, torch.Tensor) else env_ids

        if self._use_kalman:
            self._kf.reset(
                ids,
                self._ball.data.root_pos_w,
                self._ball.data.root_lin_vel_w,
            )


@configclass
class MirrorLawTorsoActionCfg(TorsoCommandActionCfg):
    """Configuration for MirrorLawTorsoAction."""

    class_type: type = MirrorLawTorsoAction

    # Ball scene entity
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")

    # Paddle offset in body frame (metres) — must match scene config
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.070)

    # Apex height range [m above paddle].  actions[:,0] ∈ [0,1] maps into this.
    apex_height_min: float = 0.05   # minimum target bounce height
    apex_height_max: float = 0.40   # maximum target bounce height

    # Nominal trunk height command [m] — must be within [0.25, 0.50]
    h_nominal: float = 0.38

    # Lateral centering gain [1/s]: v_out_xy = -K * ball_xy_rel_paddle
    centering_gain: float = 2.0

    # Ball–paddle restitution coefficient (0 < e ≤ 1)
    restitution: float = 0.85

    # Perception noise: Gaussian std added to ball pos/vel each step.
    # Simulates a camera/detector with finite accuracy.
    # 0 = perfect ground-truth state (default).
    ball_pos_noise_std: float = 0.0   # metres
    ball_vel_noise_std: float = 0.0   # m/s — only used when use_kalman=False

    # Kalman filter: smooths noisy ball position measurements.
    # When ball_vel_noise_std == 0: GT velocity is used (more accurate than KF).
    # When ball_vel_noise_std  > 0: KF velocity is used (physics smoothing beats
    #   raw noisy measurement; prevents shaking from jumpy paddle-normal commands).
    # Enabled automatically when ball_pos_noise_std > 0 and use_kalman=True.
    use_kalman: bool = True
    # Process noise std for velocity [m/s/step] — large value allows fast
    # recovery after contact events that discontinuously flip ball velocity.
    kalman_process_vel_std: float = 3.0

    # Command EMA smoothing: exponential moving average on roll, pitch, h_dot.
    # 1.0 = no smoothing (default, backward compatible).
    # 0.3 = strong smoothing (~3 step lag, ~2.4× noise reduction).
    # Use 0.3–0.5 when ball_vel_noise_std > 0 to prevent body shaking.
    cmd_smooth_alpha: float = 1.0

    # Impact tilt gain: multiply computed roll/pitch by this factor during impact.
    # 1.0 = mirror law only (default). 1.5–2.5 = diagonal energy injection.
    # Start at 1.5, increase until ball reaches target apex height.
    # Beyond ~2.5 the robot may become unstable.
    impact_tilt_gain: float = 1.0
