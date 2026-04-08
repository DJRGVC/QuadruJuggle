"""Batched Extended Kalman Filter for ball tracking.

State: [x, y, z, vx, vy, vz] (6D) or [x, y, z, vx, vy, vz, wx, wy, wz] (9D).
Dynamics: ballistic with optional quadratic drag and Magnus effect (spin).
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

9D spin mode (enable_spin=True)::

    cfg = BallEKFConfig(enable_spin=True)
    ekf = BallEKF(num_envs=N, cfg=cfg)
    # Spin is estimated from trajectory curvature (Magnus effect).
    # No direct spin measurement — spin is inferred from position obs.
    spin_est = ekf.spin  # (N, 3) angular velocity estimate
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BallEKFConfig:
    """EKF tuning parameters."""

    # Process noise — free-flight values (CWNA prescription at ~0.5m).
    # During contact, q_vel is inflated to q_vel_contact so the EKF trusts
    # measurements over its (wrong) ballistic prediction. This fixes NIS=970
    # (iter_024) without permanently degrading free-flight smoothing.
    # Ref: iter_024 NIS sweep; Bar-Shalom Ch. 6; IMM literature (Li & Jilkov 2005)
    q_pos: float = 0.003    # position process noise std (m) per sqrt(s)
    q_vel: float = 0.40     # velocity process noise std, FREE-FLIGHT (m/s) per sqrt(s)
    # ^^^ CWNA prescription for ballistic flight: σ_a * sqrt(dt) ≈ 0.40
    # Previously 7.0 to cover contact forces — now contact-aware mode handles that.

    # Contact-aware process noise: inflated q_vel during paddle contact
    contact_aware: bool = True  # enable contact detection + adaptive Q
    q_vel_contact: float = 50.0  # velocity process noise during contact (m/s/√s)
    # Contact normal force ≈ 10-100 m/s² → q_vel ≈ 50 covers worst case
    contact_z_threshold: float = 0.025  # ball Z < this → contact phase (m)
    # Ball radius = 0.020m, resting on paddle → centre at ~0.020m.
    # Threshold 0.025m gives 5mm margin for measurement noise.

    # Measurement noise — matched to D435iNoiseModelCfg (Ahn 2019 calibration)
    r_xy: float = 0.00125   # measurement noise std, XY (m) — 0.0025·z at z=0.5m
    r_z: float = 0.00225    # measurement noise std, Z (m) — 0.001 + 0.005·0.5² at z=0.5m
    r_z_per_metre: float = 0.005  # additional Z noise std per metre (matched to D435i quadratic)
    adaptive_r: bool = True  # if True, r_z varies with estimated ball height

    # Drag coefficient: F_drag = -drag_coeff * |v| * v
    # For a 40mm ping-pong ball at low speeds: Cd ≈ 0.4, A = pi*0.02^2,
    # rho = 1.2 kg/m^3 → drag_coeff = 0.5 * Cd * rho * A / m
    # = 0.5 * 0.4 * 1.2 * 1.257e-3 / 0.0027 ≈ 0.112
    # At v=4.4 m/s (Stage G): F_drag ≈ 0.49 * 4.4 ≈ 2.2 m/s^2 (non-negligible)
    drag_coeff: float = 0.112  # quadratic drag: a_drag = -drag * |v| * v_hat

    gravity_z: float = -9.81  # gravity in paddle frame Z (downward when paddle level)

    # --- Spin / Magnus effect (9D mode) ---
    enable_spin: bool = False  # if True, state is 9D with spin estimation

    # Magnus coefficient: a_magnus = magnus_coeff * (spin × vel)
    # Derivation (Kutta-Joukowski for sphere):
    #   F_M = (4/3) * π * R³ * ρ_air * (ω × v)
    #   a_M = F_M / m = (4/3) * π * R³ * ρ_air / m * (ω × v)
    # For 40mm ping-pong ball: R=0.02m, ρ=1.2 kg/m³, m=0.0027kg
    #   = (4/3) * π * 8e-6 * 1.2 / 0.0027 ≈ 0.0149 (dimensionless)
    # At ω=20 rad/s (off-centre contact), v=3 m/s → |a_magnus| ≈ 0.9 m/s²
    magnus_coeff: float = 0.0149

    # Spin decay: dω/dt = -spin_decay * ω (exponential decay from air friction)
    # Viscous torque on sphere: τ = -8πμR³ω (Stokes), I = (2/5)mR²
    # → decay = 8πμR³ / I ≈ 0.008 /s (spin half-life ~83s — barely decays)
    # During paddle contact, spin changes abruptly (covered by q_spin_contact).
    spin_decay_rate: float = 0.008

    # Spin process noise
    q_spin: float = 1.0       # spin process noise std, free-flight (rad/s per √s)
    q_spin_contact: float = 100.0  # spin process noise during contact (rad/s per √s)
    # Contact can abruptly change spin (friction torque at paddle surface)

    # Initial spin uncertainty
    p_spin_init: float = 10.0  # initial spin std (rad/s) — very uncertain

    # --- NIS gating (chi-squared outlier rejection) ---
    nis_gate_enabled: bool = True  # if True, reject measurements with NIS > threshold
    nis_gate_threshold: float = 11.345  # chi-squared 3DOF, 99th percentile
    # 95% = 7.815, 99% = 11.345, 99.5% = 12.838
    # 99% rejects ~1% of consistent measurements — tight enough to catch
    # real outliers (detector glitches, multi-ball confusion) without
    # discarding good data during contact transients.
    nis_gate_warmup: int = 50  # skip gating for first N updates per env
    # The EKF needs several updates to converge velocity from position-only
    # observations. During warm-up, all measurements are accepted. After
    # warm-up, the NIS gate rejects outliers. 50 updates ≈ 1s at 50Hz.


def _batch_skew(v: torch.Tensor) -> torch.Tensor:
    """Build batched skew-symmetric matrix [v]_x from (N, 3) vectors.

    Returns (N, 3, 3) such that [v]_x @ u = v x u (cross product).
    """
    # v = (v0, v1, v2)
    # [v]_x = [[  0, -v2,  v1],
    #          [ v2,   0, -v0],
    #          [-v1,  v0,   0]]
    N = v.shape[0]
    S = torch.zeros(N, 3, 3, device=v.device, dtype=v.dtype)
    S[:, 0, 1] = -v[:, 2]
    S[:, 0, 2] = v[:, 1]
    S[:, 1, 0] = v[:, 2]
    S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]
    S[:, 2, 1] = v[:, 0]
    return S


class BallEKF:
    """GPU-batched Kalman filter for ball position + velocity (+ optional spin).

    State dimensions:
        6D (default): x = [pos(3), vel(3)]
        9D (enable_spin): x = [pos(3), vel(3), spin(3)]

    Notation:
        x = state vector (D,) where D = 6 or 9
        P = state covariance   — (D, D) per env
        F = state transition   — (D, D), linearised dynamics
        Q = process noise cov  — (D, D)
        H = measurement matrix — (3, D), selects position from state
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
        self._spin_enabled = self.cfg.enable_spin
        self._state_dim = 9 if self._spin_enabled else 6

        D = self._state_dim
        # State: (N, D) — [x, y, z, vx, vy, vz, (wx, wy, wz)]
        self._x = torch.zeros(num_envs, D, device=self.device)

        # Covariance: (N, D, D)
        self._P = torch.eye(D, device=self.device).unsqueeze(0).expand(num_envs, -1, -1).clone()
        # Initial uncertainty: large velocity, moderate position
        self._P[:, :3, :3] *= 0.01   # 10cm std in position
        self._P[:, 3:6, 3:6] *= 1.0    # 1 m/s std in velocity
        if self._spin_enabled:
            self._P[:, 6:, 6:] *= self.cfg.p_spin_init ** 2  # large spin uncertainty

        # Measurement matrix H: z = H @ x → selects position
        self._H = torch.zeros(3, D, device=self.device)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0

        # Measurement noise R: (3, 3) diagonal (baseline; may be updated per-step)
        self._R_base = torch.diag(torch.tensor(
            [self.cfg.r_xy**2, self.cfg.r_xy**2, self.cfg.r_z**2],
            device=self.device,
        ))
        self._R = self._R_base.clone()

        # Default gravity vector (used when no body-frame gravity is provided)
        self._gravity_default = torch.tensor(
            [0.0, 0.0, self.cfg.gravity_z], device=self.device
        )

        # ANEES / NIS diagnostic (Bar-Shalom et al. 2001, Ch. 5)
        # Target: mean NIS ∈ [0.35, 7.81] for 3D measurements (95% χ²(3) band)
        self._nis_sum = 0.0
        self._nis_count = 0

        # NIS gate rejection counter (for diagnostics)
        self._gate_reject_count = 0
        self._gate_total_count = 0

        # Per-env update count (diagnostic)
        self._update_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    # --- Public properties ---

    @property
    def pos(self) -> torch.Tensor:
        """Estimated ball position (N, 3)."""
        return self._x[:, :3]

    @property
    def vel(self) -> torch.Tensor:
        """Estimated ball velocity (N, 3)."""
        return self._x[:, 3:6]

    @property
    def spin(self) -> torch.Tensor:
        """Estimated ball angular velocity (N, 3). Zero if spin disabled."""
        if self._spin_enabled:
            return self._x[:, 6:9]
        return torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def state(self) -> torch.Tensor:
        """Full state vector (N, D) where D=6 or 9."""
        return self._x

    @property
    def state_dim(self) -> int:
        """State dimension (6 or 9)."""
        return self._state_dim

    @property
    def mean_nis(self) -> float:
        """Average Normalized Innovation Squared (ANEES).

        For a well-tuned EKF with 3D measurements, this should be ~3.0
        (the mean of χ²(3)). The 95% consistency band is [0.35, 7.81].
        If consistently > 7.81: Q or R too small (overconfident).
        If consistently < 0.35: Q or R too large (over-conservative).
        """
        if self._nis_count == 0:
            return 0.0
        return self._nis_sum / self._nis_count

    def reset_nis(self) -> float:
        """Return current mean NIS and reset accumulators."""
        val = self.mean_nis
        self._nis_sum = 0.0
        self._nis_count = 0
        return val

    @property
    def gate_rejection_rate(self) -> float:
        """Fraction of detected measurements rejected by NIS gate."""
        if self._gate_total_count == 0:
            return 0.0
        return self._gate_reject_count / self._gate_total_count

    @property
    def gate_rejection_count(self) -> int:
        """Total number of measurements rejected by NIS gate."""
        return self._gate_reject_count

    def reset_gate_stats(self) -> tuple[int, int]:
        """Return (rejected, total) and reset gate counters."""
        val = (self._gate_reject_count, self._gate_total_count)
        self._gate_reject_count = 0
        self._gate_total_count = 0
        return val

    # --- Core EKF operations ---

    def predict(
        self,
        dt: float,
        gravity_b: torch.Tensor | None = None,
        robot_acc_b: torch.Tensor | None = None,
        robot_ang_vel_b: torch.Tensor | None = None,
    ) -> None:
        """Predict step: propagate state and covariance forward by dt.

        Dynamics: ballistic with quadratic drag, compensated for robot motion.
        In the non-inertial body frame, three pseudo-force corrections apply:

            a_ball = gravity_b + drag(vel) + magnus(spin, vel) - robot_acc_b
                     - 2 * omega x vel       (Coriolis)
                     - omega x (omega x pos)  (centrifugal)

        When spin is enabled (9D mode), Magnus force couples spin into the
        translational dynamics: a_magnus = magnus_coeff * (spin × vel).
        Spin decays exponentially: spin_new = spin * exp(-spin_decay * dt).

        Args:
            dt: Time step (seconds).
            gravity_b: Gravity vector in body frame (N, 3). If None, uses
                default [0, 0, -9.81] (valid only when trunk is level).
            robot_acc_b: Robot body-frame linear acceleration (N, 3).
            robot_ang_vel_b: Robot body-frame angular velocity (N, 3), rad/s.
        """
        D = self._state_dim
        pos = self._x[:, :3].clone()
        vel = self._x[:, 3:6].clone()
        if self._spin_enabled:
            spin = self._x[:, 6:9].clone()

        # Acceleration: gravity + drag + magnus - robot_acceleration (pseudo-force)
        if gravity_b is not None:
            g = gravity_b
        else:
            g = self._gravity_default.unsqueeze(0)
        speed = vel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a_drag = -self.cfg.drag_coeff * speed * vel
        a = g + a_drag

        # Magnus force: a_magnus = magnus_coeff * (spin × vel)
        if self._spin_enabled:
            a_magnus = self.cfg.magnus_coeff * torch.linalg.cross(spin, vel)
            a = a + a_magnus

        # Subtract robot body-frame acceleration (pseudo-force compensation)
        if robot_acc_b is not None:
            a = a - robot_acc_b

        # Coriolis + centrifugal corrections from robot angular velocity
        if robot_ang_vel_b is not None:
            omega = robot_ang_vel_b
            a_coriolis = -2.0 * torch.linalg.cross(omega, vel)
            a_centrifugal = -torch.linalg.cross(omega, torch.linalg.cross(omega, pos))
            a = a + a_coriolis + a_centrifugal

        # Linearised state transition F = dxnew/dx
        F = torch.eye(D, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        F[:, :3, 3:6] = torch.eye(3, device=self.device) * dt  # d(pos)/d(vel) = I*dt

        # Drag Jacobian: d(a_drag)/d(vel)
        c = self.cfg.drag_coeff
        v_hat = vel / speed
        I3 = torch.eye(3, device=self.device).unsqueeze(0)
        da_dv_drag = -c * (speed.unsqueeze(-1) * I3 + torch.bmm(vel.unsqueeze(-1), v_hat.unsqueeze(-2)))
        F[:, 3:6, 3:6] += da_dv_drag * dt

        # Magnus Jacobians (9D mode only)
        if self._spin_enabled:
            # a_magnus = Cm * (spin × vel)
            # d(a_magnus)/d(vel) = Cm * [spin]_x  (skew of spin)
            # d(a_magnus)/d(spin) = -Cm * [vel]_x  (because a×b = -b×a → d(a×b)/da = -[b]_x)
            # Actually: d(spin × vel)/d(vel) = -[spin]_x (cross product derivative)
            # Wait: (a × b)_i = ε_ijk a_j b_k → d(a×b)/db = [a]_x, d(a×b)/da = -[b]_x
            # So d(spin × vel)/d(vel) = [spin]_x
            #    d(spin × vel)/d(spin) = -[vel]_x
            Cm = self.cfg.magnus_coeff
            spin_skew = _batch_skew(spin)  # (N, 3, 3)
            vel_skew = _batch_skew(vel)    # (N, 3, 3)
            F[:, 3:6, 3:6] += Cm * spin_skew * dt   # d(vel_new)/d(vel) from Magnus
            F[:, 3:6, 6:9] = -Cm * vel_skew * dt     # d(vel_new)/d(spin)
            # Position gets second-order Magnus terms (small but correct)
            F[:, :3, 3:6] += 0.5 * Cm * spin_skew * dt**2
            F[:, :3, 6:9] = -0.5 * Cm * vel_skew * dt**2

            # Spin dynamics: spin_new = spin * exp(-decay * dt)
            # d(spin_new)/d(spin) = exp(-decay * dt) * I  (already on diagonal of F)
            decay_factor = torch.tensor(
                (-self.cfg.spin_decay_rate * dt), device=self.device
            ).exp()
            F[:, 6:9, 6:9] = torch.eye(3, device=self.device).unsqueeze(0) * decay_factor

        # Angular velocity Jacobians for Coriolis + centrifugal
        if robot_ang_vel_b is not None:
            omega_skew = _batch_skew(omega)
            F[:, 3:6, 3:6] += -2.0 * omega_skew * dt
            F[:, 3:6, :3] += -torch.bmm(omega_skew, omega_skew) * dt
            F[:, :3, :3] += -0.5 * torch.bmm(omega_skew, omega_skew) * dt**2
            F[:, :3, 3:6] += -1.0 * omega_skew * dt**2

        # State prediction
        self._x[:, :3] += vel * dt + 0.5 * a * dt**2
        self._x[:, 3:6] += a * dt
        if self._spin_enabled:
            decay_factor = torch.tensor(
                (-self.cfg.spin_decay_rate * dt), device=self.device
            ).exp()
            self._x[:, 6:9] = spin * decay_factor

        # Process noise Q — contact-aware: per-env q_vel depends on ball Z
        q_pos_sq = (self.cfg.q_pos * dt) ** 2
        if self.cfg.contact_aware:
            ball_z = self._x[:, 2]
            in_contact = ball_z < self.cfg.contact_z_threshold
            q_vel_val = torch.where(
                in_contact,
                torch.tensor(self.cfg.q_vel_contact, device=self.device),
                torch.tensor(self.cfg.q_vel, device=self.device),
            )
            q_vel_sq = (q_vel_val * dt) ** 2
            Q = torch.zeros(self.num_envs, D, D, device=self.device)
            I3 = torch.eye(3, device=self.device)
            Q[:, :3, :3] = I3.unsqueeze(0) * q_pos_sq
            Q[:, 3:6, 3:6] = I3.unsqueeze(0) * q_vel_sq.view(-1, 1, 1)
            if self._spin_enabled:
                q_spin_val = torch.where(
                    in_contact,
                    torch.tensor(self.cfg.q_spin_contact, device=self.device),
                    torch.tensor(self.cfg.q_spin, device=self.device),
                )
                q_spin_sq = (q_spin_val * dt) ** 2
                Q[:, 6:9, 6:9] = I3.unsqueeze(0) * q_spin_sq.view(-1, 1, 1)
        else:
            Q = torch.zeros(D, D, device=self.device)
            Q[:3, :3] = torch.eye(3, device=self.device) * q_pos_sq
            Q[3:6, 3:6] = torch.eye(3, device=self.device) * (self.cfg.q_vel * dt) ** 2
            if self._spin_enabled:
                Q[6:9, 6:9] = torch.eye(3, device=self.device) * (self.cfg.q_spin * dt) ** 2
            Q = Q.unsqueeze(0)

        # Covariance prediction: P = F @ P @ F^T + Q
        self._P = torch.bmm(torch.bmm(F, self._P), F.transpose(-1, -2)) + Q
        # Enforce symmetry to prevent numerical drift
        self._P = 0.5 * (self._P + self._P.transpose(-1, -2))
        # State clamping
        self._x[:, :3].clamp_(-5.0, 5.0)
        self._x[:, 3:6].clamp_(-20.0, 20.0)
        if self._spin_enabled:
            self._x[:, 6:9].clamp_(-500.0, 500.0)  # max ~80 rev/s

    def update(self, z: torch.Tensor, detected: torch.Tensor) -> None:
        """Update step: incorporate measurement where detected.

        Args:
            z: Measured ball position (N, 3).
            detected: Boolean mask (N,) — True where measurement is valid.
        """
        if not detected.any():
            return

        D = self._state_dim
        # Time-varying R: depth noise grows with distance (D435i stereo baseline)
        if self.cfg.adaptive_r:
            z_height = self._x[:, 2].abs()
            sigma_z = self.cfg.r_z + self.cfg.r_z_per_metre * z_height
            R = self._R_base.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
            R[:, 2, 2] = sigma_z ** 2
        else:
            R = self._R.unsqueeze(0).expand(self.num_envs, -1, -1)

        # Innovation: y = z - H @ x = z - pos_predicted
        y = z - self._x[:, :3]

        # Innovation covariance: S = H @ P @ H^T + R
        # Since H selects the first 3 rows: S = P[:3, :3] + R
        S = self._P[:, :3, :3] + R

        # Kalman gain: K = P @ H^T @ S^{-1}
        # P @ H^T = P[:, :, :3] (since H^T has non-zero only in first 3 rows)
        PH_T = self._P[:, :, :3]  # (N, D, 3)
        # Regularise S to prevent singular matrices from numerical drift
        S += torch.eye(3, device=self.device).unsqueeze(0) * 1e-8
        K = torch.linalg.solve(S.transpose(-1, -2), PH_T.transpose(-1, -2)).transpose(-1, -2)  # (N, D, 3)

        # NIS per env: NIS_k = y^T S^{-1} y (scalar per env)
        # Compute for ALL envs (needed for gating), accumulate stats for detected only.
        S_inv_y = torch.linalg.solve(S, y.unsqueeze(-1))  # (N, 3, 1)
        nis = torch.bmm(y.unsqueeze(1), S_inv_y).squeeze(-1).squeeze(-1)  # (N,)

        # Accumulate NIS diagnostic for detected envs (before gating)
        if detected.any():
            nis_det = nis[detected]
            self._nis_sum += nis_det.sum().item()
            self._nis_count += nis_det.numel()

        # NIS gating: reject measurements where NIS > threshold (outlier).
        # Per-env warm-up: skip gating until env has had enough successful
        # updates for both position and velocity to converge. Needed because
        # position converges in 1 step but velocity (indirectly observed
        # through position changes) takes many more.
        if self.cfg.nis_gate_enabled:
            past_warmup = self._update_count >= self.cfg.nis_gate_warmup
            gate_eligible = detected & past_warmup
            if gate_eligible.any():
                n_eligible = gate_eligible.sum().item()
                self._gate_total_count += int(n_eligible)
                gated_out = gate_eligible & (nis > self.cfg.nis_gate_threshold)
                n_rejected = gated_out.sum().item()
                self._gate_reject_count += int(n_rejected)
                # Mask out gated measurements — treat as undetected
                detected = detected & ~gated_out

        if not detected.any():
            return

        # Increment per-env update count for envs that pass gating
        self._update_count[detected] += 1

        # State update: x = x + K @ y (only for detected envs)
        dx = torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)  # (N, D)
        mask = detected.unsqueeze(-1).float()
        self._x += dx * mask

        # Covariance update: P = (I - K @ H) @ P (standard form)
        I_D = torch.eye(D, device=self.device).unsqueeze(0)
        H_expanded = self._H.unsqueeze(0).expand(self.num_envs, -1, -1)
        IKH = I_D - torch.bmm(K, H_expanded)
        P_new = torch.bmm(IKH, self._P)

        # Blend: detected envs get new P, undetected keep old P
        self._P = torch.where(detected.view(-1, 1, 1), P_new, self._P)
        # Enforce symmetry + small regularization for positive definiteness
        self._P = 0.5 * (self._P + self._P.transpose(-1, -2))
        self._P += I_D * 1e-8

    def step(
        self,
        z: torch.Tensor,
        detected: torch.Tensor,
        dt: float,
        gravity_b: torch.Tensor | None = None,
        robot_acc_b: torch.Tensor | None = None,
        robot_ang_vel_b: torch.Tensor | None = None,
    ) -> None:
        """Combined predict + update step.

        Args:
            z: Measured ball position (N, 3).
            detected: Boolean mask (N,) — True where measurement available.
            dt: Time step (seconds).
            gravity_b: Gravity in body frame (N, 3). See :meth:`predict`.
            robot_acc_b: Robot body-frame acceleration (N, 3). See :meth:`predict`.
            robot_ang_vel_b: Robot angular velocity (N, 3). See :meth:`predict`.
        """
        self.predict(dt, gravity_b=gravity_b, robot_acc_b=robot_acc_b,
                     robot_ang_vel_b=robot_ang_vel_b)
        self.update(z, detected)

    def reset(
        self,
        env_ids: torch.Tensor,
        init_pos: torch.Tensor,
        init_vel: torch.Tensor | None = None,
        init_spin: torch.Tensor | None = None,
    ) -> None:
        """Reset filter state for specified environments.

        Args:
            env_ids: Indices of envs to reset (M,).
            init_pos: Initial ball position (M, 3).
            init_vel: Initial ball velocity (M, 3). Defaults to zero.
            init_spin: Initial ball angular velocity (M, 3). Defaults to zero.
                Only used when enable_spin=True.
        """
        if init_vel is None:
            init_vel = torch.zeros_like(init_pos)

        D = self._state_dim
        self._x[env_ids, :3] = init_pos
        self._x[env_ids, 3:6] = init_vel
        if self._spin_enabled:
            if init_spin is not None:
                self._x[env_ids, 6:9] = init_spin
            else:
                self._x[env_ids, 6:9] = 0.0

        # Reset covariance to initial values
        P_init = torch.eye(D, device=self.device)
        P_init[:3, :3] *= 0.01
        P_init[3:6, 3:6] *= 1.0
        if self._spin_enabled:
            P_init[6:9, 6:9] *= self.cfg.p_spin_init ** 2
        self._P[env_ids] = P_init.unsqueeze(0).expand(len(env_ids), -1, -1)

        # Reset per-env update count so gating warm-up restarts
        self._update_count[env_ids] = 0
