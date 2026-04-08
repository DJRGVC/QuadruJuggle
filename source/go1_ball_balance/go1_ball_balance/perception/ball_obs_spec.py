"""Ball observation provider — drop-in replacement for oracle obs terms.

ETH-style architecture: GT ball state → noise model → EKF → policy.
This module provides ObsTerm-compatible functions that the env config can
swap in place of the raw ``ball_pos_in_paddle_frame`` / ``ball_vel_in_paddle_frame``.

Noise modes:
    "oracle"  — pass-through ground truth (default, identical to current pipeline)
    "d435i"   — D435i-style structured noise (position + depth-dependent, dropout)
    "ekf"     — noise model → EKF filter (full pipeline, added later)

Usage in env_cfg.py::

    from go1_ball_balance.perception.ball_obs_spec import (
        ball_pos_perceived,
        ball_vel_perceived,
        BallObsNoiseCfg,
    )

    ball_pos = ObsTerm(
        func=ball_pos_perceived,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "noise_cfg": BallObsNoiseCfg(mode="oracle"),
        },
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .ball_ekf import BallEKF, BallEKFConfig
from .noise_model import D435iNoiseModel, D435iNoiseModelCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Noise configuration
# ---------------------------------------------------------------------------

@dataclass
class D435iNoiseParams:
    """D435i depth-camera noise model parameters.

    Defaults from Intel D435i datasheet + empirical measurements:
    - Depth accuracy ~2% of distance at 1m (structured IR stereo)
    - XY accuracy ~1-2mm at 1m (640×480, ~60° HFOV)
    - Dropout probability increases at close range / specular surfaces
    """

    sigma_xy_base: float = 0.002       # 2mm base XY noise std (metres)
    sigma_z_base: float = 0.003        # 3mm base depth noise std
    sigma_z_per_metre: float = 0.002   # +2mm per metre of distance
    dropout_prob: float = 0.02         # 2% chance of missed detection per step
    latency_steps: int = 1             # observation delay in policy steps


@dataclass
class BallObsNoiseCfg:
    """Configuration for the ball observation noise model.

    Pass this as a ``noise_cfg`` param to ``ball_pos_perceived`` and
    ``ball_vel_perceived`` ObsTerms.
    """

    mode: str = "oracle"
    """Noise mode: "oracle", "d435i", or "ekf"."""

    noise_scale: float = 1.0
    """Multiplier for noise amplitudes (0.0 = oracle-equivalent, 1.0 = full noise).

    Scales sigma_xy_base, sigma_z_base, sigma_z_per_metre, and dropout_prob
    in both d435i and ekf modes. Useful for gradual noise introduction across
    curriculum stages (e.g. 0.25 → 0.50 → 0.75 → 1.0).
    """

    d435i: D435iNoiseParams = field(default_factory=D435iNoiseParams)
    """D435i noise parameters (only used when mode="d435i")."""

    noise_model_cfg: D435iNoiseModelCfg = field(default_factory=D435iNoiseModelCfg)
    """Stateful noise model config (used when mode="ekf")."""

    ekf_cfg: BallEKFConfig = field(default_factory=BallEKFConfig)
    """EKF config (used when mode="ekf")."""

    policy_dt: float = 0.02
    """Policy step period in seconds (50Hz default). Used by EKF predict step."""


# ---------------------------------------------------------------------------
# Perception pipeline: stateful noise + EKF (lazy-initialized on env)
# ---------------------------------------------------------------------------

class PerceptionPipeline:
    """Stateful perception pipeline: D435i noise model → EKF → filtered state.

    Attached to the env object as ``env._perception_pipeline`` on first use.
    Manages noise model and EKF instances, handles resets, and provides
    filtered position and velocity estimates.

    Optionally tracks diagnostics (estimation error vs GT) when
    ``enable_diagnostics=True``. Access via :attr:`diagnostics`.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        noise_cfg: BallObsNoiseCfg,
        enable_diagnostics: bool = False,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = noise_cfg

        self.noise_model = D435iNoiseModel(
            num_envs=num_envs,
            device=device,
            cfg=noise_cfg.noise_model_cfg,
        )
        self.ekf = BallEKF(
            num_envs=num_envs,
            device=device,
            cfg=noise_cfg.ekf_cfg,
        )

        # Track which step we're on (for deduplication within a single step)
        self._last_step_count = -1

        # Robot acceleration estimation via finite differences
        self._prev_robot_vel_b: torch.Tensor | None = None

        # Diagnostics: running error accumulators
        self._diagnostics_enabled = enable_diagnostics
        if enable_diagnostics:
            self._diag = _PerceptionDiagnostics(num_envs, self.device)
        else:
            self._diag = None

    def step(
        self,
        gt_pos_b: torch.Tensor,
        env_step_count: int,
        gt_vel_b: torch.Tensor | None = None,
        gravity_b: torch.Tensor | None = None,
        robot_vel_b: torch.Tensor | None = None,
    ) -> None:
        """Run one noise→EKF cycle. Idempotent within a single env step.

        Called by both ball_pos_perceived and ball_vel_perceived; only
        runs the actual predict+update once per step.

        Args:
            gt_pos_b: GT ball position in paddle frame (N, 3).
            env_step_count: Current env step counter.
            gt_vel_b: GT ball velocity in trunk frame (N, 3). Only needed
                for diagnostics — pass None to skip velocity error tracking.
            gravity_b: Gravity vector in body frame (N, 3). If provided,
                the EKF uses the actual gravity direction accounting for
                trunk tilt. Otherwise defaults to [0, 0, -9.81].
            robot_vel_b: Robot body-frame linear velocity (N, 3). Used to
                compute body-frame acceleration via finite differences for
                pseudo-force compensation in EKF predict.
        """
        if env_step_count == self._last_step_count:
            return  # already ran this step (pos called before vel)

        self._last_step_count = env_step_count

        # Compute robot body-frame acceleration via finite differences
        robot_acc_b = None
        if robot_vel_b is not None:
            if self._prev_robot_vel_b is not None:
                dt = self.cfg.policy_dt
                robot_acc_b = (robot_vel_b - self._prev_robot_vel_b) / dt
                # Clamp to prevent wild accelerations from resets
                robot_acc_b = robot_acc_b.clamp(-50.0, 50.0)
            self._prev_robot_vel_b = robot_vel_b.clone()

        # Generate noisy measurement
        noisy_pos, detected = self.noise_model.sample(gt_pos_b)

        # EKF predict + update (with body-frame gravity + acceleration compensation)
        self.ekf.step(
            noisy_pos, detected, dt=self.cfg.policy_dt,
            gravity_b=gravity_b, robot_acc_b=robot_acc_b,
        )

        # Diagnostics
        if self._diag is not None:
            self._diag.record(
                gt_pos=gt_pos_b,
                gt_vel=gt_vel_b,
                noisy_pos=noisy_pos,
                ekf_pos=self.ekf.pos,
                ekf_vel=self.ekf.vel,
                detected=detected,
            )

    def reset(
        self,
        env_ids: torch.Tensor,
        init_pos: torch.Tensor,
        init_vel: torch.Tensor | None = None,
    ) -> None:
        """Reset noise model and EKF for specified environments."""
        self.noise_model.reset(env_ids, init_pos)
        self.ekf.reset(env_ids, init_pos, init_vel)
        # Clear robot velocity buffer for reset envs so we don't compute
        # a spurious acceleration spike on the first post-reset step
        if self._prev_robot_vel_b is not None:
            self._prev_robot_vel_b[env_ids] = 0.0

    @property
    def pos(self) -> torch.Tensor:
        """EKF-filtered ball position estimate (N, 3)."""
        return self.ekf.pos

    @property
    def vel(self) -> torch.Tensor:
        """EKF-filtered ball velocity estimate (N, 3)."""
        return self.ekf.vel

    def update_noise_scale(self, scale: float) -> None:
        """Update noise amplitudes in the live noise model (for curriculum).

        Modifies the D435iNoiseModel's config in-place so that subsequent
        calls to ``sample()`` use the scaled parameters. Does not affect EKF
        process/measurement noise (those are in BallEKFConfig).
        """
        base = self.cfg.noise_model_cfg  # already scaled at creation time
        # We need to work from the UNSCALED base. Store it on first call.
        if not hasattr(self, "_base_noise_model_cfg"):
            # On first pipeline creation, cfg.noise_model_cfg was already
            # scaled by noise_scale. Recover the unscaled version.
            old_scale = self.cfg.noise_scale
            if old_scale > 0:
                self._base_noise_model_cfg = D435iNoiseModelCfg(
                    sigma_xy_base=base.sigma_xy_base / old_scale,
                    sigma_z_base=base.sigma_z_base / old_scale,
                    sigma_z_per_metre=base.sigma_z_per_metre / old_scale,
                    dropout_prob=base.dropout_prob / old_scale,
                    latency_steps=base.latency_steps,
                    camera_hz=base.camera_hz,
                )
            else:
                self._base_noise_model_cfg = D435iNoiseModelCfg()

        bm = self._base_noise_model_cfg
        self.noise_model.cfg.sigma_xy_base = bm.sigma_xy_base * scale
        self.noise_model.cfg.sigma_z_base = bm.sigma_z_base * scale
        self.noise_model.cfg.sigma_z_per_metre = bm.sigma_z_per_metre * scale
        self.noise_model.cfg.dropout_prob = bm.dropout_prob * scale
        self.cfg.noise_scale = scale

    @property
    def diagnostics(self) -> dict[str, float] | None:
        """Return current diagnostic summary and reset accumulators.

        Returns None if diagnostics are disabled. Otherwise returns a dict
        with keys like 'pos_rmse_ekf', 'pos_rmse_raw', 'vel_rmse_ekf',
        'detection_rate', 'ekf_improvement_pct', 'mean_nis'.
        """
        if self._diag is None:
            return None
        result = self._diag.summary_and_reset()
        # Add EKF ANEES diagnostic (resets accumulator)
        result["mean_nis"] = round(self.ekf.reset_nis(), 3)
        return result


class _PerceptionDiagnostics:
    """Lightweight running-statistics tracker for perception pipeline errors."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._pos_sq_err_ekf = 0.0   # sum of squared EKF pos errors
        self._pos_sq_err_raw = 0.0   # sum of squared raw noisy pos errors
        self._vel_sq_err_ekf = 0.0   # sum of squared EKF vel errors
        self._detected_count = 0     # number of detected samples
        self._total_count = 0        # total (env × step) samples

    def record(
        self,
        gt_pos: torch.Tensor,
        gt_vel: torch.Tensor | None,
        noisy_pos: torch.Tensor,
        ekf_pos: torch.Tensor,
        ekf_vel: torch.Tensor,
        detected: torch.Tensor,
    ) -> None:
        """Record one step of errors across all envs."""
        N = gt_pos.shape[0]

        # Position errors (L2 per env, then sum)
        ekf_pos_err = (ekf_pos - gt_pos).pow(2).sum(dim=-1)  # (N,)
        raw_pos_err = (noisy_pos - gt_pos).pow(2).sum(dim=-1)  # (N,)

        self._pos_sq_err_ekf += ekf_pos_err.sum().item()
        self._pos_sq_err_raw += raw_pos_err.sum().item()

        if gt_vel is not None:
            vel_err = (ekf_vel - gt_vel).pow(2).sum(dim=-1)
            self._vel_sq_err_ekf += vel_err.sum().item()

        self._detected_count += detected.sum().item()
        self._total_count += N

    def summary_and_reset(self) -> dict[str, float]:
        """Compute RMSE summary and reset accumulators."""
        if self._total_count == 0:
            return {}

        n = self._total_count
        pos_rmse_ekf = (self._pos_sq_err_ekf / n) ** 0.5
        pos_rmse_raw = (self._pos_sq_err_raw / n) ** 0.5
        vel_rmse_ekf = (self._vel_sq_err_ekf / n) ** 0.5 if self._vel_sq_err_ekf > 0 else 0.0
        detection_rate = self._detected_count / n

        # Improvement: how much EKF reduces error vs raw noise
        if pos_rmse_raw > 1e-9:
            improvement_pct = (1.0 - pos_rmse_ekf / pos_rmse_raw) * 100.0
        else:
            improvement_pct = 0.0

        result = {
            "pos_rmse_ekf_mm": round(pos_rmse_ekf * 1000, 2),
            "pos_rmse_raw_mm": round(pos_rmse_raw * 1000, 2),
            "vel_rmse_ekf_mps": round(vel_rmse_ekf, 4),
            "detection_rate": round(detection_rate, 4),
            "ekf_improvement_pct": round(improvement_pct, 1),
            "num_samples": int(n),
        }

        self._reset_accumulators()
        return result


def _scaled_d435i_params(
    params: D435iNoiseParams, scale: float
) -> D435iNoiseParams:
    """Return a copy of D435i noise params with amplitudes multiplied by scale."""
    if scale == 1.0:
        return params
    return D435iNoiseParams(
        sigma_xy_base=params.sigma_xy_base * scale,
        sigma_z_base=params.sigma_z_base * scale,
        sigma_z_per_metre=params.sigma_z_per_metre * scale,
        dropout_prob=params.dropout_prob * scale,
        latency_steps=params.latency_steps,  # latency is not scaled
    )


def _scaled_noise_model_cfg(
    cfg: D435iNoiseModelCfg, scale: float
) -> D435iNoiseModelCfg:
    """Return a copy of D435iNoiseModelCfg with amplitudes multiplied by scale."""
    if scale == 1.0:
        return cfg
    return D435iNoiseModelCfg(
        sigma_xy_base=cfg.sigma_xy_base * scale,
        sigma_z_base=cfg.sigma_z_base * scale,
        sigma_z_per_metre=cfg.sigma_z_per_metre * scale,
        dropout_prob=cfg.dropout_prob * scale,
        latency_steps=cfg.latency_steps,  # latency is not scaled
        camera_hz=cfg.camera_hz,
    )


def _get_or_create_pipeline(
    env: "ManagerBasedRLEnv",
    noise_cfg: BallObsNoiseCfg,
) -> PerceptionPipeline:
    """Get or lazily create the PerceptionPipeline on the env object."""
    if not hasattr(env, "_perception_pipeline") or env._perception_pipeline is None:
        # Apply noise_scale to the noise model config before creating pipeline
        scaled_cfg = BallObsNoiseCfg(
            mode=noise_cfg.mode,
            noise_scale=noise_cfg.noise_scale,
            d435i=noise_cfg.d435i,
            noise_model_cfg=_scaled_noise_model_cfg(
                noise_cfg.noise_model_cfg, noise_cfg.noise_scale
            ),
            ekf_cfg=noise_cfg.ekf_cfg,
            policy_dt=noise_cfg.policy_dt,
        )
        # Enable diagnostics if env has the flag (set by test scripts)
        enable_diag = getattr(env, "_perception_diagnostics_enabled", False)
        env._perception_pipeline = PerceptionPipeline(
            num_envs=env.num_envs,
            device=env.device,
            noise_cfg=scaled_cfg,
            enable_diagnostics=enable_diag,
        )
    return env._perception_pipeline


# ---------------------------------------------------------------------------
# Internal: oracle ball state (same math as observations.py)
# ---------------------------------------------------------------------------

def _ball_pos_paddle_frame_gt(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    paddle_offset_b: tuple[float, float, float],
) -> torch.Tensor:
    """Ground-truth ball position in paddle frame. Shape (N, 3)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(
        paddle_offset_b, device=env.device
    ).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

    ball_pos_w = ball.data.root_pos_w
    diff_w = ball_pos_w - paddle_pos_w
    return math_utils.quat_apply_inverse(trunk_quat_w, diff_w)


def _ball_vel_paddle_frame_gt(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ground-truth ball velocity in trunk frame. Shape (N, 3)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_quat_w = robot.data.root_quat_w
    ball_vel_w = ball.data.root_lin_vel_w
    return math_utils.quat_apply_inverse(trunk_quat_w, ball_vel_w)


# ---------------------------------------------------------------------------
# Public ObsTerm-compatible functions
# ---------------------------------------------------------------------------

def ball_pos_perceived(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.070),
    noise_cfg: BallObsNoiseCfg | None = None,
) -> torch.Tensor:
    """Ball position in paddle frame, optionally with perception noise.

    Drop-in replacement for ``mdp.ball_pos_in_paddle_frame``.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    if noise_cfg is None:
        noise_cfg = BallObsNoiseCfg()

    pos_b = _ball_pos_paddle_frame_gt(env, ball_cfg, robot_cfg, paddle_offset_b)

    if noise_cfg.mode == "oracle":
        return pos_b

    if noise_cfg.mode == "d435i":
        scaled_params = _scaled_d435i_params(noise_cfg.d435i, noise_cfg.noise_scale)
        return _apply_d435i_pos_noise(pos_b, scaled_params)

    if noise_cfg.mode == "ekf":
        pipeline = _get_or_create_pipeline(env, noise_cfg)
        # Pass body-frame gravity + velocity for pseudo-force compensation
        robot: Articulation = env.scene[robot_cfg.name]
        gravity_b = robot.data.projected_gravity_b * 9.81  # (N, 3)
        robot_vel_b = robot.data.root_lin_vel_b  # (N, 3)
        pipeline.step(
            pos_b, env.common_step_counter,
            gravity_b=gravity_b, robot_vel_b=robot_vel_b,
        )
        out = pipeline.pos.clone()
        # Guard against EKF divergence — NaN/Inf would corrupt PPO gradients
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return out.clamp(-5.0, 5.0)

    raise ValueError(f"Unknown noise mode: {noise_cfg.mode!r}")


def ball_vel_perceived(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noise_cfg: BallObsNoiseCfg | None = None,
) -> torch.Tensor:
    """Ball velocity in trunk frame, optionally with perception noise.

    Drop-in replacement for ``mdp.ball_vel_in_paddle_frame``.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    if noise_cfg is None:
        noise_cfg = BallObsNoiseCfg()

    vel_b = _ball_vel_paddle_frame_gt(env, ball_cfg, robot_cfg)

    if noise_cfg.mode == "oracle":
        return vel_b

    if noise_cfg.mode == "d435i":
        scaled_params = _scaled_d435i_params(noise_cfg.d435i, noise_cfg.noise_scale)
        return _apply_d435i_vel_noise(vel_b, scaled_params)

    if noise_cfg.mode == "ekf":
        pipeline = _get_or_create_pipeline(env, noise_cfg)
        # step() is idempotent — if pos was called first, this is a no-op
        robot: Articulation = env.scene[robot_cfg.name]
        gravity_b = robot.data.projected_gravity_b * 9.81  # (N, 3)
        robot_vel_b = robot.data.root_lin_vel_b  # (N, 3)
        pipeline.step(
            _ball_pos_paddle_frame_gt(env, ball_cfg, robot_cfg, (0.0, 0.0, 0.070)),
            env.common_step_counter,
            gt_vel_b=vel_b if pipeline._diagnostics_enabled else None,
            gravity_b=gravity_b,
            robot_vel_b=robot_vel_b,
        )
        out = pipeline.vel.clone()
        out = torch.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0)
        return out.clamp(-10.0, 10.0)

    raise ValueError(f"Unknown noise mode: {noise_cfg.mode!r}")


# ---------------------------------------------------------------------------
# D435i noise model (structured, depth-dependent)
# ---------------------------------------------------------------------------

def _apply_d435i_pos_noise(
    pos_b: torch.Tensor,
    params: D435iNoiseParams,
) -> torch.Tensor:
    """Apply D435i-style structured noise to ball position in paddle frame.

    Noise model:
    - XY noise: Gaussian with std = sigma_xy_base (lateral pixel noise)
    - Z noise: Gaussian with std = sigma_z_base + sigma_z_per_metre * |z|
      (depth accuracy degrades with distance — D435i stereo baseline effect)
    - Dropout: with probability dropout_prob, return last known position
      (simulated as zeroing the update — full dropout buffer added with EKF)
    """
    device = pos_b.device
    N = pos_b.shape[0]

    # XY noise (lateral, from pixel quantisation + IR pattern matching)
    xy_noise = torch.randn(N, 2, device=device) * params.sigma_xy_base

    # Z noise (depth, distance-dependent)
    z_dist = pos_b[:, 2].abs()  # distance along z in paddle frame
    sigma_z = params.sigma_z_base + params.sigma_z_per_metre * z_dist
    z_noise = torch.randn(N, device=device) * sigma_z

    noise = torch.stack([xy_noise[:, 0], xy_noise[:, 1], z_noise], dim=-1)

    # Dropout: zero the noise for dropped frames (position freezes at GT
    # for now; proper hold-last-value requires EKF state buffer)
    if params.dropout_prob > 0:
        dropout_mask = (
            torch.rand(N, device=device) < params.dropout_prob
        ).unsqueeze(-1)
        # On dropout, return GT (no noise) — conservative stub.
        # Full pipeline will hold last EKF estimate instead.
        noise = noise * (~dropout_mask).float()

    return pos_b + noise


def _apply_d435i_vel_noise(
    vel_b: torch.Tensor,
    params: D435iNoiseParams,
) -> torch.Tensor:
    """Apply noise to ball velocity estimate.

    In the real pipeline, velocity comes from the EKF (finite-differenced
    noisy positions). For the d435i stub, we approximate this as additive
    Gaussian noise scaled by the position noise parameters and a typical
    frame-to-frame dt (~33ms at 30Hz).

    The velocity noise std is approximately sigma_pos / dt, which at
    30Hz gives ~3x amplification of position noise.
    """
    device = vel_b.device
    N = vel_b.shape[0]

    # Approximate velocity noise from finite-differenced position noise
    # at 30Hz camera rate: sigma_vel ≈ sqrt(2) * sigma_pos / dt
    dt_camera = 1.0 / 30.0  # 30Hz D435i frame rate
    sigma_vel_xy = (2 ** 0.5) * params.sigma_xy_base / dt_camera
    sigma_vel_z = (2 ** 0.5) * params.sigma_z_base / dt_camera

    noise = torch.zeros_like(vel_b)
    noise[:, :2] = torch.randn(N, 2, device=device) * sigma_vel_xy
    noise[:, 2] = torch.randn(N, device=device) * sigma_vel_z

    if params.dropout_prob > 0:
        dropout_mask = (
            torch.rand(N, device=device) < params.dropout_prob
        ).unsqueeze(-1)
        noise = noise * (~dropout_mask).float()

    return vel_b + noise


# ---------------------------------------------------------------------------
# Reset event for EKF pipeline (call from env_cfg reset events)
# ---------------------------------------------------------------------------

def update_perception_noise_scale(
    env: "ManagerBasedRLEnv",
    noise_scale: float,
) -> None:
    """Update the noise scale on a live perception pipeline (for curriculum).

    Call this from the curriculum callback when advancing stages::

        from go1_ball_balance.perception import update_perception_noise_scale
        update_perception_noise_scale(env, noise_scale=0.5)

    No-op if the pipeline hasn't been created yet (oracle mode).
    """
    pipeline: PerceptionPipeline | None = getattr(env, "_perception_pipeline", None)
    if pipeline is not None:
        pipeline.update_noise_scale(noise_scale)


def reset_perception_pipeline(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.070),
) -> None:
    """Reset the perception pipeline (noise model + EKF) for reset environments.

    Add this as an EventTerm in the env_cfg reset events when using mode="ekf"::

        reset_perception = EventTerm(
            func=reset_perception_pipeline,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "paddle_offset_b": _PADDLE_OFFSET_B,
            },
        )
    """
    pipeline: PerceptionPipeline | None = getattr(env, "_perception_pipeline", None)
    if pipeline is None:
        return  # pipeline not yet created (oracle mode or first step)

    # Get current ball position in paddle frame for these envs
    pos_b = _ball_pos_paddle_frame_gt(env, ball_cfg, robot_cfg, paddle_offset_b)
    vel_b = _ball_vel_paddle_frame_gt(env, ball_cfg, robot_cfg)

    pipeline.reset(env_ids, pos_b[env_ids], vel_b[env_ids])
