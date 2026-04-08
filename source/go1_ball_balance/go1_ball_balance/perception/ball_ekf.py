"""Batched 6-state Extended Kalman Filter for ball tracking.

State: [x, y, z, vx, vy, vz] in paddle (body) frame.
Dynamics: ballistic with optional quadratic drag.
Measurement: noisy [x, y, z] from D435i noise model.

The filter runs on GPU, batched across all environments. Identical code
will run on the real robot (single-env, same PyTorch ops).

Usage::

    ekf = BallEKF(num_envs=12288, device="cuda:0")

    # Each policy step:
    z_meas = noisy_ball_pos  # (N, 3) from D435i noise model
    detected = ~dropout_mask  # (N,) bool — True if measurement available
    ekf.step(z_meas, detected, dt=0.02)  # dt = policy step period

    pos_est = ekf.pos   # (N, 3)
    vel_est = ekf.vel   # (N, 3)

    # On episode reset:
    ekf.reset(env_ids, init_pos, init_vel)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BallEKFConfig:
    """EKF tuning parameters."""

    # Process noise (how much we distrust the dynamics model)
    q_pos: float = 0.01     # position process noise std (m) per sqrt(s)
    q_vel: float = 1.0      # velocity process noise std (m/s) per sqrt(s)

    # Measurement noise (how much we distrust the camera)
    # These should roughly match D435iNoiseParams but can be tuned independently.
    r_xy: float = 0.003     # measurement noise std, XY (m)
    r_z: float = 0.005      # measurement noise std, Z (m)

    # Drag coefficient: F_drag = -drag_coeff * |v| * v
    # For a 40mm ping-pong ball at low speeds: Cd ≈ 0.4, A = pi*0.02^2,
    # rho = 1.2 kg/m^3 → drag_coeff = 0.5 * Cd * rho * A / m
    # = 0.5 * 0.4 * 1.2 * 1.257e-3 / 0.0027 ≈ 0.112
    # At v=4.4 m/s (Stage G): F_drag ≈ 0.49 * 4.4 ≈ 2.2 m/s^2 (non-negligible)
    drag_coeff: float = 0.112  # quadratic drag: a_drag = -drag * |v| * v_hat

    gravity_z: float = -9.81  # gravity in paddle frame Z (downward when paddle level)


class BallEKF:
    """GPU-batched 6-state Kalman filter for ball position + velocity.

    Notation:
        x = [pos(3), vel(3)]  — state vector (6,)
        P = state covariance   — (6, 6) per env
        F = state transition   — (6, 6), linearised dynamics
        Q = process noise cov  — (6, 6)
        H = measurement matrix — (3, 6), selects position from state
        R = measurement noise  — (3, 3)
        z = measurement        — (3,), noisy ball position
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cpu",
        cfg: BallEKFConfig | None = None,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or BallEKFConfig()

        # State: (N, 6) — [x, y, z, vx, vy, vz]
        self._x = torch.zeros(num_envs, 6, device=self.device)

        # Covariance: (N, 6, 6)
        self._P = torch.eye(6, device=self.device).unsqueeze(0).expand(num_envs, -1, -1).clone()
        # Initial uncertainty: large velocity, moderate position
        self._P[:, :3, :3] *= 0.01   # 10cm std in position
        self._P[:, 3:, 3:] *= 1.0    # 1 m/s std in velocity

        # Measurement matrix H: z = H @ x → selects position
        self._H = torch.zeros(3, 6, device=self.device)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0

        # Measurement noise R: (3, 3) diagonal
        self._R = torch.diag(torch.tensor(
            [self.cfg.r_xy**2, self.cfg.r_xy**2, self.cfg.r_z**2],
            device=self.device,
        ))

        # Default gravity vector (used when no body-frame gravity is provided)
        self._gravity_default = torch.tensor(
            [0.0, 0.0, self.cfg.gravity_z], device=self.device
        )

    # --- Public properties ---

    @property
    def pos(self) -> torch.Tensor:
        """Estimated ball position (N, 3)."""
        return self._x[:, :3]

    @property
    def vel(self) -> torch.Tensor:
        """Estimated ball velocity (N, 3)."""
        return self._x[:, 3:]

    @property
    def state(self) -> torch.Tensor:
        """Full state vector (N, 6)."""
        return self._x

    # --- Core EKF operations ---

    def predict(self, dt: float, gravity_b: torch.Tensor | None = None) -> None:
        """Predict step: propagate state and covariance forward by dt.

        Dynamics: ballistic with quadratic drag.
            pos_new = pos + vel * dt + 0.5 * a * dt^2
            vel_new = vel + a * dt
        where a = gravity + drag(vel).

        Args:
            dt: Time step (seconds).
            gravity_b: Gravity vector in body frame (N, 3). If None, uses
                default [0, 0, -9.81] (valid only when trunk is level).
                Pass ``robot.data.projected_gravity_b * 9.81`` to account
                for trunk tilt.
        """
        vel = self._x[:, 3:].clone()  # (N, 3) — clone to avoid view mutation

        # Acceleration: gravity + drag
        if gravity_b is not None:
            g = gravity_b  # (N, 3) — already in body frame
        else:
            g = self._gravity_default.unsqueeze(0)  # (1, 3) broadcasts
        speed = vel.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1)
        a_drag = -self.cfg.drag_coeff * speed * vel  # quadratic drag
        a = g + a_drag  # (N, 3)

        # Linearised state transition F = dxnew/dx (compute BEFORE state mutation)
        # F is identity + off-diagonal blocks from dynamics linearisation
        F = torch.eye(6, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        F[:, :3, 3:] = torch.eye(3, device=self.device) * dt  # d(pos)/d(vel) = I*dt

        # Drag Jacobian: d(a_drag)/d(vel)
        # a_drag = -c * |v| * v → da/dv = -c * (|v|*I + v⊗v/|v|)
        c = self.cfg.drag_coeff
        v_hat = vel / speed  # (N, 3) — safe because vel is cloned
        I3 = torch.eye(3, device=self.device).unsqueeze(0)
        da_dv = -c * (speed.unsqueeze(-1) * I3 + torch.bmm(vel.unsqueeze(-1), v_hat.unsqueeze(-2)))
        F[:, 3:, 3:] += da_dv * dt  # d(vel_new)/d(vel) = I + da_dv * dt

        # State prediction (mutates self._x)
        self._x[:, :3] += vel * dt + 0.5 * a * dt**2
        self._x[:, 3:] += a * dt

        # Process noise Q
        Q = torch.zeros(6, 6, device=self.device)
        Q[:3, :3] = torch.eye(3, device=self.device) * (self.cfg.q_pos * dt)**2
        Q[3:, 3:] = torch.eye(3, device=self.device) * (self.cfg.q_vel * dt)**2

        # Covariance prediction: P = F @ P @ F^T + Q
        self._P = torch.bmm(torch.bmm(F, self._P), F.transpose(-1, -2)) + Q.unsqueeze(0)
        # Enforce symmetry to prevent numerical drift
        self._P = 0.5 * (self._P + self._P.transpose(-1, -2))
        # State clamping — physically, ball can't be >5m away or >20m/s
        self._x[:, :3].clamp_(-5.0, 5.0)
        self._x[:, 3:].clamp_(-20.0, 20.0)

    def update(self, z: torch.Tensor, detected: torch.Tensor) -> None:
        """Update step: incorporate measurement where detected.

        Args:
            z: Measured ball position (N, 3).
            detected: Boolean mask (N,) — True where measurement is valid.
        """
        if not detected.any():
            return

        # Innovation: y = z - H @ x = z - pos_predicted
        y = z - self._x[:, :3]  # (N, 3)

        # Innovation covariance: S = H @ P @ H^T + R
        # Since H selects the first 3 rows: S = P[:3, :3] + R
        S = self._P[:, :3, :3] + self._R.unsqueeze(0)  # (N, 3, 3)

        # Kalman gain: K = P @ H^T @ S^{-1}
        # P @ H^T = P[:, :, :3] (since H^T has non-zero only in first 3 rows)
        PH_T = self._P[:, :, :3]  # (N, 6, 3)
        # Regularise S to prevent singular matrices from numerical drift
        S += torch.eye(3, device=self.device).unsqueeze(0) * 1e-8
        K = torch.linalg.solve(S.transpose(-1, -2), PH_T.transpose(-1, -2)).transpose(-1, -2)  # (N, 6, 3)

        # State update: x = x + K @ y (only for detected envs)
        dx = torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)  # (N, 6)
        mask = detected.unsqueeze(-1).float()  # (N, 1)
        self._x += dx * mask

        # Covariance update: P = (I - K @ H) @ P (standard form)
        I6 = torch.eye(6, device=self.device).unsqueeze(0)
        H_expanded = self._H.unsqueeze(0).expand(self.num_envs, -1, -1)
        IKH = I6 - torch.bmm(K, H_expanded)
        P_new = torch.bmm(IKH, self._P)

        # Blend: detected envs get new P, undetected keep old P
        self._P = torch.where(detected.view(-1, 1, 1), P_new, self._P)
        # Enforce symmetry + small regularization for positive definiteness
        self._P = 0.5 * (self._P + self._P.transpose(-1, -2))
        self._P += I6 * 1e-8

    def step(
        self,
        z: torch.Tensor,
        detected: torch.Tensor,
        dt: float,
        gravity_b: torch.Tensor | None = None,
    ) -> None:
        """Combined predict + update step.

        Args:
            z: Measured ball position (N, 3).
            detected: Boolean mask (N,) — True where measurement available.
            dt: Time step (seconds).
            gravity_b: Gravity in body frame (N, 3). See :meth:`predict`.
        """
        self.predict(dt, gravity_b=gravity_b)
        self.update(z, detected)

    def reset(
        self,
        env_ids: torch.Tensor,
        init_pos: torch.Tensor,
        init_vel: torch.Tensor | None = None,
    ) -> None:
        """Reset filter state for specified environments.

        Args:
            env_ids: Indices of envs to reset (M,).
            init_pos: Initial ball position (M, 3).
            init_vel: Initial ball velocity (M, 3). Defaults to zero.
        """
        if init_vel is None:
            init_vel = torch.zeros_like(init_pos)

        self._x[env_ids, :3] = init_pos
        self._x[env_ids, 3:] = init_vel

        # Reset covariance to initial values
        P_init = torch.eye(6, device=self.device)
        P_init[:3, :3] *= 0.01
        P_init[3:, 3:] *= 1.0
        self._P[env_ids] = P_init.unsqueeze(0).expand(len(env_ids), -1, -1)
