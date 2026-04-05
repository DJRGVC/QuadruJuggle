"""Batched Kalman filter for ball state estimation under gravity.

State  : [px, py, pz, vx, vy, vz]   (6D per environment)
Model  : free flight under constant gravity.
         Contact discontinuities are absorbed by high process noise on velocity.
Observe: noisy position [px, py, pz] only — as a depth/stereo camera would provide.
         Velocity is estimated implicitly from position tracking + physics model.

Usage::

    kf = BallKalmanFilter(num_envs=4, dt=0.02, device="cuda:0",
                          pos_noise_std=0.01)
    kf.reset(env_ids, ball_pos_gt, ball_vel_gt)  # initialise on episode reset

    # each control step:
    pos_est, vel_est = kf.step(ball_pos_noisy)
"""

import torch


class BallKalmanFilter:
    """Per-environment Kalman filter for ball state estimation.

    Args:
        num_envs:          Number of parallel environments.
        dt:                Control timestep [s] (env step-size, after decimation).
        device:            Torch device string.
        pos_noise_std:     Measurement noise std for position [m].
        process_pos_std:   Process noise std for position [m/step].
                           Keep small — free flight position is very predictable.
        process_vel_std:   Process noise std for velocity [m/s per step].
                           Set large (≥ 2.0) so the filter recovers quickly after
                           a contact event that flips velocity sign.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: str,
        pos_noise_std: float,
        process_pos_std: float = 0.001,
        process_vel_std: float = 3.0,
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        N = num_envs

        # ── State and covariance ──────────────────────────────────────────────
        self.x = torch.zeros(N, 6, device=device)        # (N, 6)
        self.P = torch.eye(6, device=device).unsqueeze(0).repeat(N, 1, 1)  # (N,6,6)

        # ── State-transition matrix F: x_next = F @ x + b  ───────────────────
        # pos += vel * dt  (no gravity in matrix; gravity enters via affine term b)
        F = torch.eye(6, device=device)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.F = F          # (6, 6) — identical for all envs

        # Gravity affine term
        b = torch.zeros(6, device=device)
        b[2] = -0.5 * 9.81 * dt * dt   # Δpz
        b[5] = -9.81 * dt               # Δvz
        self.b = b                       # (6,)

        # ── Process noise Q ───────────────────────────────────────────────────
        Q_diag = torch.zeros(6, device=device)
        Q_diag[:3] = process_pos_std ** 2
        Q_diag[3:] = process_vel_std ** 2
        self.Q = torch.diag(Q_diag).unsqueeze(0).repeat(N, 1, 1)  # (N,6,6)

        # ── Observation: position only  H shape (3, 6) ───────────────────────
        H = torch.zeros(3, 6, device=device)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        self.H = H          # (3, 6)
        self.Ht = H.t()     # (6, 3)

        R_diag = torch.full((3,), pos_noise_std ** 2, device=device)
        self.R = torch.diag(R_diag).unsqueeze(0).repeat(N, 1, 1)  # (N,3,3)

        self.initialized = torch.zeros(N, dtype=torch.bool, device=device)

    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        env_ids: torch.Tensor,
        ball_pos: torch.Tensor,
        ball_vel: torch.Tensor,
    ) -> None:
        """Re-initialise filter state for the given environments.

        Call this on every episode reset with GROUND-TRUTH ball state so the
        filter starts from the correct position rather than the noisy estimate.

        Args:
            env_ids:  1-D tensor of environment indices to reset.
            ball_pos: (N_total, 3) ground-truth ball positions (world frame).
            ball_vel: (N_total, 3) ground-truth ball velocities (world frame).
        """
        n = len(env_ids)
        self.x[env_ids, :3] = ball_pos[env_ids]
        self.x[env_ids, 3:] = ball_vel[env_ids]
        self.P[env_ids] = torch.eye(6, device=self.device).unsqueeze(0).expand(n, -1, -1).clone()
        self.initialized[env_ids] = True

    def step(
        self,
        ball_pos_noisy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one predict + update cycle.

        Args:
            ball_pos_noisy: (N, 3) noisy ball position measurement (world frame).

        Returns:
            (pos_est, vel_est): each (N, 3) filtered estimates.
        """
        # Auto-initialise any envs that have not been reset yet (e.g. first step).
        # Seed from the measurement with zero velocity assumption.
        uninit = ~self.initialized
        if uninit.any():
            self.x[uninit, :3] = ball_pos_noisy[uninit]
            self.x[uninit, 3:] = 0.0
            self.initialized[uninit] = True

        F  = self.F   # (6, 6)
        Ht = self.Ht  # (6, 3)
        H  = self.H   # (3, 6)

        # ── Predict ───────────────────────────────────────────────────────────
        # x_pred = F @ x + b
        x_pred = (F @ self.x.unsqueeze(-1)).squeeze(-1) + self.b   # (N, 6)
        # P_pred = F @ P @ Ft + Q
        P_pred = F @ self.P @ F.t() + self.Q                        # (N, 6, 6)

        # ── Update ────────────────────────────────────────────────────────────
        # Innovation: y = z - H @ x_pred
       
        y = ball_pos_noisy - (H @ x_pred.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        # Innovation covariance: S = H @ P_pred @ Ht + R
        S = H @ P_pred @ Ht + self.R                                  # (N, 3, 3)
        # Kalman gain: K = P_pred @ Ht @ S^{-1}
        K = P_pred @ Ht @ torch.linalg.inv(S)                         # (N, 6, 3)
        # State update
        self.x = x_pred + (K @ y.unsqueeze(-1)).squeeze(-1)           # (N, 6)
        # Covariance update (simplified form; sufficient for symmetric R/Q)
        self.P = (torch.eye(6, device=self.device) - K @ H) @ P_pred  # (N, 6, 6)

        return self.x[:, :3].clone(), self.x[:, 3:].clone()
