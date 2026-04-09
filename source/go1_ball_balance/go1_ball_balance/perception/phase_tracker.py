"""Ball flight-phase tracker for detection scheduling and diagnostics.

Tracks the ball's phase (contact / ascending / descending) per environment
using the EKF's estimated state. This enables:

1. **Camera scheduling**: skip detection when ball is on paddle (saves compute
   on real hardware where YOLO inference is costly).
2. **Phase-aware metrics**: split EKF accuracy by phase without post-hoc
   recomputation from trajectory files.
3. **Bounce counting**: track bounce events for curriculum diagnostics.

Usage::

    tracker = BallPhaseTracker(num_envs=N, device="cuda:0")

    # Each step, after EKF predict+update:
    tracker.update(ekf.pos, ekf.vel, contact_z_threshold=paddle_z + 0.03)

    phase = tracker.phase  # (N,) int: 0=contact, 1=ascending, 2=descending
    in_flight = tracker.in_flight  # (N,) bool: True if ascending or descending
    bounces = tracker.bounce_count  # (N,) int: cumulative bounces per env

    # On episode reset:
    tracker.reset(env_ids)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class BallPhase(IntEnum):
    """Ball flight phase enum."""
    CONTACT = 0     # ball on or near paddle surface
    ASCENDING = 1   # ball moving upward after bounce
    DESCENDING = 2  # ball falling back toward paddle


@dataclass
class BallPhaseTrackerConfig:
    """Configuration for phase detection."""

    # Minimum upward velocity to transition contact → ascending (m/s).
    # Filters out noise: at 200Hz with 5mm paddle vibration, noise vz ≈ 1 m/s.
    # Real bounces have vz > 1.5 m/s (even Stage A target=0.10m → vz ≈ 1.4 m/s).
    launch_vz_threshold: float = 0.5

    # Hysteresis: ball must be above contact zone by this margin to count
    # as truly in flight (prevents jitter at the boundary).
    flight_height_margin: float = 0.01  # 10mm above contact_z_threshold


class BallPhaseTracker:
    """GPU-batched ball phase tracker.

    Maintains per-env phase state and bounce count. Designed to run alongside
    a BallEKF, consuming its pos/vel estimates each step.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cpu",
        cfg: BallPhaseTrackerConfig | None = None,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or BallPhaseTrackerConfig()

        # Per-env state
        self._phase = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._bounce_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._steps_in_phase = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Peak height tracking (per bounce)
        self._peak_height = torch.zeros(num_envs, device=self.device)

        # Cumulative phase step counters (for diagnostics)
        self._total_contact_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._total_flight_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    # --- Public properties ---

    @property
    def phase(self) -> torch.Tensor:
        """Current phase per env (N,) — 0=contact, 1=ascending, 2=descending."""
        return self._phase

    @property
    def in_flight(self) -> torch.Tensor:
        """Boolean mask (N,) — True if ascending or descending."""
        return self._phase > 0

    @property
    def in_contact(self) -> torch.Tensor:
        """Boolean mask (N,) — True if on paddle."""
        return self._phase == 0

    @property
    def is_ascending(self) -> torch.Tensor:
        """Boolean mask (N,) — True if moving upward."""
        return self._phase == 1

    @property
    def is_descending(self) -> torch.Tensor:
        """Boolean mask (N,) — True if falling."""
        return self._phase == 2

    @property
    def bounce_count(self) -> torch.Tensor:
        """Cumulative bounces per env (N,)."""
        return self._bounce_count

    @property
    def steps_in_phase(self) -> torch.Tensor:
        """Steps spent in current phase per env (N,)."""
        return self._steps_in_phase

    @property
    def peak_height(self) -> torch.Tensor:
        """Peak height (above contact_z) reached in current/last bounce (N,)."""
        return self._peak_height

    @property
    def flight_fraction(self) -> torch.Tensor:
        """Fraction of total steps spent in flight per env (N,)."""
        total = self._total_contact_steps + self._total_flight_steps
        return torch.where(total > 0,
                           self._total_flight_steps.float() / total.float(),
                           torch.zeros_like(total, dtype=torch.float))

    def update(
        self,
        ball_pos: torch.Tensor,
        ball_vel: torch.Tensor,
        contact_z_threshold: float | torch.Tensor,
    ) -> None:
        """Update phase classification from EKF state estimates.

        Call once per step, after EKF predict+update.

        Args:
            ball_pos: EKF position estimate (N, 3).
            ball_vel: EKF velocity estimate (N, 3).
            contact_z_threshold: Z threshold for contact detection (scalar or (N,)).
                Ball is in contact zone when ball_pos[:,2] < contact_z_threshold.
        """
        ball_z = ball_pos[:, 2]
        ball_vz = ball_vel[:, 2]

        if isinstance(contact_z_threshold, (int, float)):
            cz = contact_z_threshold
            flight_z = cz + self.cfg.flight_height_margin
        else:
            cz = contact_z_threshold
            flight_z = cz + self.cfg.flight_height_margin

        # Classify current physical state
        in_contact_zone = ball_z < cz
        above_flight_threshold = ball_z >= flight_z
        going_up = ball_vz > self.cfg.launch_vz_threshold
        going_down = ball_vz <= 0

        prev_phase = self._phase.clone()

        # --- State transitions ---

        # CONTACT → ASCENDING: ball has upward velocity AND has left contact zone
        launch = (prev_phase == BallPhase.CONTACT) & going_up & above_flight_threshold
        self._phase[launch] = BallPhase.ASCENDING
        self._steps_in_phase[launch] = 0
        self._bounce_count[launch] += 1
        self._peak_height[launch] = ball_z[launch]

        # ASCENDING → DESCENDING: vz ≤ 0 (reached apex)
        apex = (prev_phase == BallPhase.ASCENDING) & going_down
        self._phase[apex] = BallPhase.DESCENDING
        self._steps_in_phase[apex] = 0

        # Update peak height during ascent
        ascending_now = self._phase == BallPhase.ASCENDING
        self._peak_height[ascending_now] = torch.max(
            self._peak_height[ascending_now], ball_z[ascending_now]
        )

        # DESCENDING → CONTACT: ball enters contact zone
        landing = (prev_phase == BallPhase.DESCENDING) & in_contact_zone
        self._phase[landing] = BallPhase.CONTACT
        self._steps_in_phase[landing] = 0

        # ASCENDING → CONTACT: ball falls back without clear descent phase
        # (e.g., weak bounce that barely left the surface)
        weak_landing = (prev_phase == BallPhase.ASCENDING) & in_contact_zone
        self._phase[weak_landing] = BallPhase.CONTACT
        self._steps_in_phase[weak_landing] = 0

        # Increment step counters
        self._steps_in_phase += 1
        self._total_contact_steps += (self._phase == BallPhase.CONTACT).long()
        self._total_flight_steps += (self._phase > 0).long()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset tracker state for specified environments.

        Args:
            env_ids: Indices of envs to reset (M,).
        """
        self._phase[env_ids] = BallPhase.CONTACT
        self._bounce_count[env_ids] = 0
        self._steps_in_phase[env_ids] = 0
        self._peak_height[env_ids] = 0.0
        self._total_contact_steps[env_ids] = 0
        self._total_flight_steps[env_ids] = 0

    def summary(self) -> dict[str, float]:
        """Return aggregate statistics across all envs.

        Returns:
            Dictionary with mean bounce count, flight fraction, etc.
        """
        return {
            "mean_bounces": self._bounce_count.float().mean().item(),
            "mean_flight_fraction": self.flight_fraction.mean().item(),
            "mean_peak_height": self._peak_height.mean().item(),
            "pct_in_flight": self.in_flight.float().mean().item() * 100,
            "pct_ascending": self.is_ascending.float().mean().item() * 100,
            "pct_descending": self.is_descending.float().mean().item() * 100,
            "pct_contact": self.in_contact.float().mean().item() * 100,
        }
