"""D435i-structured noise model for ball observations.

Stateful noise generator that simulates D435i depth camera characteristics:
- Depth-dependent position noise (XY and Z with different profiles)
- Measurement dropout (IR interference, specular reflection)
- Observation latency (frame capture → readout delay)
- Proper hold-last-value on dropout (not GT passthrough)

Usage::

    noise = D435iNoiseModel(num_envs=12288, device="cuda:0")

    # Each policy step:
    noisy_pos, detected = noise.sample(gt_pos_paddle_frame, dt=0.02)

    # On episode reset:
    noise.reset(env_ids, init_pos)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class D435iNoiseModelCfg:
    """D435i noise model configuration.

    Defaults calibrated from Ahn et al. 2019 (IEEE UR), Intel BKM whitepaper,
    and lit-review iter_023 D435i noise characterisation.
    Matches D435iNoiseParams in ball_obs_spec.py but designed for
    stateful use with latency and proper dropout.
    """

    # Position noise — XY (lateral, pixel quantisation + IR pattern matching)
    sigma_xy_per_metre: float = 0.0025  # σ_xy = 0.0025·z (linear in z; Ahn 2019)
    sigma_xy_floor: float = 0.001       # 1mm floor at close range

    # Position noise — Z (depth, quadratic in z: stereo disparity ε_d/(f·B)·z²)
    sigma_z_base: float = 0.001         # 1mm constant floor (sensor quantisation)
    sigma_z_quadratic: float = 0.005    # 0.005·z² (Ahn 2019 + 25% ball-geometry margin)

    # Dropout — distance-dependent (white ball specular + small apparent size)
    dropout_base: float = 0.20          # 20% baseline (curved white surface; lit-review §9.3)
    dropout_range: float = 0.30         # additional 30% rising with distance
    dropout_scale: float = 0.80         # exponential scale (metres)

    # Latency
    latency_steps: int = 1              # observation delay in policy steps

    # Camera frame rate (for velocity noise estimation)
    camera_hz: float = 30.0             # D435i depth frame rate


class D435iNoiseModel:
    """Stateful D435i noise model with latency buffer and hold-last-value dropout.

    Unlike the inline noise in ball_obs_spec._apply_d435i_pos_noise, this class:
    - Maintains a latency buffer (delays observations by N policy steps)
    - Holds the last valid measurement on dropout (not GT)
    - Provides a detected mask for the EKF update step
    - Shares dropout events between position and velocity
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cpu",
        cfg: D435iNoiseModelCfg | None = None,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or D435iNoiseModelCfg()

        # Last valid measurement (for hold-on-dropout)
        self._last_valid = torch.zeros(num_envs, 3, device=self.device)

        # Latency buffer: deque of (noisy_pos, detected) tuples
        # Pre-fill with zeros so first N steps still produce output
        self._latency_buf: deque[tuple[torch.Tensor, torch.Tensor]] = deque(
            maxlen=max(1, self.cfg.latency_steps + 1),
        )
        for _ in range(self.cfg.latency_steps + 1):
            self._latency_buf.append((
                torch.zeros(num_envs, 3, device=self.device),
                torch.zeros(num_envs, dtype=torch.bool, device=self.device),
            ))

    def sample(
        self,
        gt_pos_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate noisy D435i measurement from ground-truth position.

        Args:
            gt_pos_b: Ground-truth ball position in paddle frame (N, 3).

        Returns:
            noisy_pos: Noisy (or held) ball position (N, 3).
            detected: Boolean mask (N,) — True where a fresh measurement exists.
        """
        N = gt_pos_b.shape[0]
        device = gt_pos_b.device
        cfg = self.cfg

        # --- Apply position noise ---
        z_dist = gt_pos_b[:, 2].abs()

        # XY noise: σ_xy = max(floor, 0.0025·z) — linear in depth (Ahn 2019)
        sigma_xy = torch.clamp(cfg.sigma_xy_per_metre * z_dist, min=cfg.sigma_xy_floor)
        xy_noise = torch.randn(N, 2, device=device) * sigma_xy.unsqueeze(-1)

        # Z noise: σ_z = base + quadratic·z² — stereo disparity error ∝ z²
        sigma_z = cfg.sigma_z_base + cfg.sigma_z_quadratic * z_dist * z_dist
        z_noise = torch.randn(N, device=device) * sigma_z

        noisy_pos = gt_pos_b.clone()
        noisy_pos[:, 0] += xy_noise[:, 0]
        noisy_pos[:, 1] += xy_noise[:, 1]
        noisy_pos[:, 2] += z_noise

        # --- Distance-dependent dropout ---
        # p = base + range·(1 - exp(-max(0, z-0.5)/scale))
        # 20% at z≤0.5m, rising to ~50% at z=1.5m (white ball specular + small apparent size)
        z_excess = torch.clamp(z_dist - 0.5, min=0.0)
        p_dropout = cfg.dropout_base + cfg.dropout_range * (
            1.0 - torch.exp(-z_excess / cfg.dropout_scale)
        )
        detected = torch.rand(N, device=device) >= p_dropout  # True = valid

        # Hold last valid measurement on dropout
        output_pos = torch.where(
            detected.unsqueeze(-1),
            noisy_pos,
            self._last_valid,
        )

        # Update last valid where detected
        self._last_valid = torch.where(
            detected.unsqueeze(-1),
            noisy_pos,
            self._last_valid,
        )

        # --- Latency buffer ---
        self._latency_buf.append((output_pos, detected))

        # Return the oldest entry (delayed by latency_steps)
        delayed_pos, delayed_detected = self._latency_buf[0]
        return delayed_pos, delayed_detected

    def reset(
        self,
        env_ids: torch.Tensor,
        init_pos: torch.Tensor,
    ) -> None:
        """Reset noise state for specified environments.

        Args:
            env_ids: Indices of envs to reset (M,).
            init_pos: Initial ball position (M, 3) in paddle frame.
        """
        self._last_valid[env_ids] = init_pos

        # Reset latency buffer entries for these envs
        for i in range(len(self._latency_buf)):
            pos, det = self._latency_buf[i]
            pos[env_ids] = init_pos
            det[env_ids] = True
