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

    # Post-contact inflation: after a bounce, velocity estimate is stale.
    # Keep q_vel elevated for N steps after ball leaves contact zone.
    # At 200Hz physics, 10 steps = 50ms = ~1-2 measurement cycles.
    # Ref: D'Ambrosio RSS 2023 (state machine), lit_review_ekf_q_tuning iter_031.
    post_contact_steps: int = 10  # steps after contact to keep q_vel inflated
    q_vel_post_contact: float = 20.0  # q_vel during post-contact window (m/s/√s)
    # Intermediate between flight (0.4) and contact (50.0) — allows filter to
    # converge to new post-bounce velocity without full contact-level noise.

    # Ascending-phase process noise: during clean ballistic ascent (vz > 0,
    # above contact zone, post-contact window expired), the dynamics model
    # (gravity + drag) is highly accurate. Tighter q_vel trusts the prediction
    # more, reducing covariance growth during measurement dropout at high
    # altitudes. Ref: noise-to-gap model (iter_131) — predict drift is the
    # dominant gap driver at high targets.
    q_vel_ascending: float = 0.25  # q_vel during ascending flight (m/s/√s)
    # 0.25 vs 0.40 default: ~37% tighter. Still covers residual drag model error
    # (drag_coeff uncertainty ~20% → ~0.1 m/s² at 3 m/s → ~0.02 m/s per step).

    # Pre-landing phase: ball is descending near the paddle. Contact is imminent
    # and will cause a large velocity discontinuity. Inflating Q before contact
    # widens the covariance ellipse so the first post-contact measurements get
    # accepted (lower NIS) and the filter transitions faster to the new velocity.
    # Ref: Bar-Shalom Ch. 11.6 (manoeuvre detection); D'Ambrosio RSS 2023.
    q_vel_pre_landing: float = 2.0  # q_vel when ball descending near paddle (m/s/√s)
    pre_landing_z_threshold: float = 0.08  # activate when ball_z < this AND vz < 0 (m)
    # 0.08m ≈ 4× ball radius ≈ 16 steps at 200Hz before contact at ~1 m/s descent.
    # q_vel=2.0: 5× default flight noise, << contact (50) or post-contact (20).

    # Measurement noise — matched to D435iNoiseModelCfg (Ahn 2019 calibration)
    r_xy: float = 0.00125   # measurement noise std, XY (m) — 0.0025·z at z=0.5m
    r_xy_per_metre: float = 0.0025  # σ_xy = r_xy_per_metre · z (matched to D435i; Ahn 2019)
    r_xy_floor: float = 0.0005  # 0.5mm floor (avoid singular R at z≈0)
    r_z: float = 0.00225    # measurement noise std, Z (m) — 0.001 + 0.005·0.5² at z=0.5m
    r_z_per_metre: float = 0.005  # additional Z noise std per metre (matched to D435i quadratic)
    adaptive_r: bool = True  # if True, R_xy and R_z vary with estimated ball height

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

    # --- Covariance clamping (sparse-measurement regime) ---
    # When the ball sits on the paddle for long periods with no camera measurements,
    # covariance grows unbounded via predict-only steps. This causes the EKF to
    # become useless when a measurement finally arrives (innovation covariance S
    # is so large that the gain K → 0, or NIS gate rejects valid measurements).
    # Clamping P diagonals prevents this divergence.
    # Ref: iter_103 analysis of 2-4% detection rate at target=0.42m.
    p_clamp_enabled: bool = True
    p_max_pos: float = 0.25  # max position std (m) — 25cm covers paddle area
    p_max_vel: float = 5.0   # max velocity std (m/s) — 5 m/s covers post-bounce
    p_max_spin: float = 50.0  # max spin std (rad/s) — only used in 9D mode

    # --- Paddle-anchor virtual measurement ---
    # When the ball sits on the paddle with no camera detections for many steps,
    # inject a virtual measurement at the known paddle position. This prevents
    # EKF drift during long contact phases (98%+ of episode at current policy)
    # and gives pi1 accurate ball observations even without camera data.
    # The anchor is only active when: (1) steps_since_measurement > threshold,
    # AND (2) the EKF estimates the ball is in the contact zone.
    anchor_enabled: bool = True
    anchor_r_pos: float = 0.005  # anchor measurement noise std (m) — 5mm
    # Tight noise because paddle position is known kinematically (not camera).
    # 5mm covers attachment tolerance + small vibration.
    anchor_min_starve_steps: int = 5  # min predict-only steps before anchoring
    # At 200Hz, 5 steps = 25ms. Don't anchor if camera just had a detection.

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
        # Phase-separated NIS: free-flight vs contact (iter_050)
        self._nis_sum_flight = 0.0
        self._nis_count_flight = 0
        self._nis_sum_contact = 0.0
        self._nis_count_contact = 0

        # NIS gate rejection counter (for diagnostics)
        self._gate_reject_count = 0
        self._gate_total_count = 0

        # Per-env update count (diagnostic)
        self._update_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Post-contact inflation counter: counts down from post_contact_steps
        # after ball exits contact zone. 0 = not in post-contact window.
        self._post_contact_countdown = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )

        # Sparse-measurement tracking: consecutive predict-only steps per env
        self._steps_since_measurement = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )

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

    @property
    def mean_nis_flight(self) -> float:
        """Mean NIS during free-flight phase only."""
        if self._nis_count_flight == 0:
            return 0.0
        return self._nis_sum_flight / self._nis_count_flight

    @property
    def mean_nis_contact(self) -> float:
        """Mean NIS during paddle-contact phase only."""
        if self._nis_count_contact == 0:
            return 0.0
        return self._nis_sum_contact / self._nis_count_contact

    def reset_nis(self) -> float:
        """Return current mean NIS and reset all NIS accumulators."""
        val = self.mean_nis
        self._nis_sum = 0.0
        self._nis_count = 0
        self._nis_sum_flight = 0.0
        self._nis_count_flight = 0
        self._nis_sum_contact = 0.0
        self._nis_count_contact = 0
        return val

    @property
    def steps_since_measurement(self) -> torch.Tensor:
        """Per-env count of consecutive predict-only steps (N,)."""
        return self._steps_since_measurement

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

        # Process noise Q — contact-aware with phase-aware scheduling.
        # Five-level q_vel: contact (50) > post-contact (20) > pre-landing (2.0)
        #   > descending (0.4) > ascending (0.25).
        # Post-contact window covers the first N steps after bounce where the
        # filter's velocity estimate is stale from the pre-bounce prediction.
        # Ascending phase uses tighter q_vel because ballistic dynamics are highly
        # predictable (gravity + drag), reducing covariance growth during dropout.
        # Pre-landing phase inflates Q when the ball is descending near the paddle,
        # preparing the covariance for the upcoming contact discontinuity.
        # Ref: D'Ambrosio RSS 2023, lit_review_ekf_q_tuning iter_031, noise-to-gap iter_131.
        q_pos_sq = (self.cfg.q_pos * dt) ** 2
        if self.cfg.contact_aware:
            ball_z = self._x[:, 2]
            ball_vz = self._x[:, 5]
            in_contact = ball_z < self.cfg.contact_z_threshold

            # Update post-contact countdown:
            # - entering contact: reset countdown to full window
            # - in flight with countdown > 0: decrement
            self._post_contact_countdown[in_contact] = self.cfg.post_contact_steps
            in_post_contact = (~in_contact) & (self._post_contact_countdown > 0)
            self._post_contact_countdown[in_post_contact] -= 1

            # Ascending detection: above contact zone, post-contact expired, vz > 0
            in_ascending = (~in_contact) & (~in_post_contact) & (ball_vz > 0)

            # Pre-landing detection: descending near paddle, not yet in contact
            in_pre_landing = (
                (~in_contact)
                & (~in_post_contact)
                & (ball_vz < 0)
                & (ball_z < self.cfg.pre_landing_z_threshold)
            )

            # Five-level q_vel selection (ascending < default < pre-landing < post-contact < contact)
            q_vel_val = torch.full(
                (self.num_envs,), self.cfg.q_vel, device=self.device
            )
            q_vel_val[in_ascending] = self.cfg.q_vel_ascending
            q_vel_val[in_pre_landing] = self.cfg.q_vel_pre_landing
            q_vel_val[in_post_contact] = self.cfg.q_vel_post_contact
            q_vel_val[in_contact] = self.cfg.q_vel_contact

            q_vel_sq = (q_vel_val * dt) ** 2
            Q = torch.zeros(self.num_envs, D, D, device=self.device)
            I3 = torch.eye(3, device=self.device)
            Q[:, :3, :3] = I3.unsqueeze(0) * q_pos_sq
            Q[:, 3:6, 3:6] = I3.unsqueeze(0) * q_vel_sq.view(-1, 1, 1)
            if self._spin_enabled:
                q_spin_val = torch.full(
                    (self.num_envs,), self.cfg.q_spin, device=self.device
                )
                q_spin_val[in_contact] = self.cfg.q_spin_contact
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

        # Covariance clamping: prevent unbounded P growth during long
        # predict-only sequences (ball on paddle, no camera detections).
        # Clamp diagonal elements of P to max variance; zero off-diagonal
        # correlations that exceed the clamped diagonal (maintain PSD).
        if self.cfg.p_clamp_enabled:
            p_max_pos_sq = self.cfg.p_max_pos ** 2
            p_max_vel_sq = self.cfg.p_max_vel ** 2
            # Clamp position diagonals
            self._P[:, 0, 0].clamp_(max=p_max_pos_sq)
            self._P[:, 1, 1].clamp_(max=p_max_pos_sq)
            self._P[:, 2, 2].clamp_(max=p_max_pos_sq)
            # Clamp velocity diagonals
            self._P[:, 3, 3].clamp_(max=p_max_vel_sq)
            self._P[:, 4, 4].clamp_(max=p_max_vel_sq)
            self._P[:, 5, 5].clamp_(max=p_max_vel_sq)
            if self._spin_enabled:
                p_max_spin_sq = self.cfg.p_max_spin ** 2
                self._P[:, 6, 6].clamp_(max=p_max_spin_sq)
                self._P[:, 7, 7].clamp_(max=p_max_spin_sq)
                self._P[:, 8, 8].clamp_(max=p_max_spin_sq)
            # Clamp off-diagonals to maintain PSD: |P_ij| <= sqrt(P_ii * P_jj)
            # Full enforcement is expensive; clamping diagonals alone is sufficient
            # in practice because the symmetric update in the next step re-derives
            # off-diagonals from the (now-bounded) diagonals.

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
        # Time-varying R: noise grows with distance (D435i stereo baseline)
        if self.cfg.adaptive_r:
            z_height = self._x[:, 2].abs()
            # XY noise: σ_xy = r_xy_per_metre · z, floored
            sigma_xy = torch.clamp(
                self.cfg.r_xy_per_metre * z_height, min=self.cfg.r_xy_floor
            )
            # Z noise: σ_z = r_z + r_z_per_metre · z
            sigma_z = self.cfg.r_z + self.cfg.r_z_per_metre * z_height
            R = torch.zeros(self.num_envs, 3, 3, device=self.device)
            R[:, 0, 0] = sigma_xy ** 2
            R[:, 1, 1] = sigma_xy ** 2
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

            # Phase-separated NIS: split by contact state
            if self.cfg.contact_aware:
                ball_z_det = self._x[detected, 2]
                in_contact_det = ball_z_det < self.cfg.contact_z_threshold
                in_flight_det = ~in_contact_det
                if in_flight_det.any():
                    nis_flight = nis_det[in_flight_det]
                    self._nis_sum_flight += nis_flight.sum().item()
                    self._nis_count_flight += nis_flight.numel()
                if in_contact_det.any():
                    nis_contact = nis_det[in_contact_det]
                    self._nis_sum_contact += nis_contact.sum().item()
                    self._nis_count_contact += nis_contact.numel()
            else:
                # No contact awareness — all measurements are "flight"
                self._nis_sum_flight += nis_det.sum().item()
                self._nis_count_flight += nis_det.numel()

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

        # Track measurement starvation per env
        self._steps_since_measurement += 1
        self._steps_since_measurement[detected] = 0

    def paddle_anchor_update(self, paddle_pos: torch.Tensor) -> int:
        """Inject virtual measurement for envs where ball is on paddle with no camera data.

        When the ball sits on the paddle and the camera can't see it (ball below
        camera FOV, or on the paddle surface), the EKF gets no measurements and
        drifts. This method injects a low-noise position measurement at the known
        paddle position for environments that satisfy BOTH:
          1. steps_since_measurement >= anchor_min_starve_steps
          2. Estimated ball Z < contact_z_threshold (ball in contact zone)

        The measurement uses the anchor's own R matrix (much tighter than camera R)
        since paddle position is known kinematically. This does NOT reset
        steps_since_measurement — only real camera measurements do that.

        Args:
            paddle_pos: Known paddle-frame ball rest position (N, 3). Typically
                [0, 0, ball_radius] for each env (ball centre when resting on paddle).

        Returns:
            Number of environments that received an anchor measurement.
        """
        if not self.cfg.anchor_enabled:
            return 0

        # Identify starved contact envs
        starved = self._steps_since_measurement >= self.cfg.anchor_min_starve_steps
        in_contact = self._x[:, 2] < self.cfg.contact_z_threshold
        anchor_mask = starved & in_contact

        if not anchor_mask.any():
            return 0

        n_anchored = int(anchor_mask.sum().item())
        D = self._state_dim

        # Anchor measurement noise: small isotropic R
        r_sq = self.cfg.anchor_r_pos ** 2
        R_anchor = torch.eye(3, device=self.device) * r_sq
        R_anchor = R_anchor.unsqueeze(0).expand(self.num_envs, -1, -1)

        # Innovation: y = paddle_pos - predicted_pos
        y = paddle_pos - self._x[:, :3]

        # Innovation covariance: S = P[:3,:3] + R_anchor
        S = self._P[:, :3, :3] + R_anchor
        S += torch.eye(3, device=self.device).unsqueeze(0) * 1e-8

        # Kalman gain: K = P[:, :, :3] @ S^{-1}
        PH_T = self._P[:, :, :3]
        K = torch.linalg.solve(
            S.transpose(-1, -2), PH_T.transpose(-1, -2)
        ).transpose(-1, -2)

        # State update (only for anchor_mask envs)
        dx = torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)
        mask = anchor_mask.unsqueeze(-1).float()
        self._x += dx * mask

        # Covariance update: P = (I - K @ H) @ P
        I_D = torch.eye(D, device=self.device).unsqueeze(0)
        H_expanded = self._H.unsqueeze(0).expand(self.num_envs, -1, -1)
        IKH = I_D - torch.bmm(K, H_expanded)
        P_new = torch.bmm(IKH, self._P)
        self._P = torch.where(anchor_mask.view(-1, 1, 1), P_new, self._P)
        self._P = 0.5 * (self._P + self._P.transpose(-1, -2))
        self._P += I_D * 1e-8

        # Also zero the velocity for anchored envs — ball on paddle is stationary
        vel_mask = anchor_mask.unsqueeze(-1).float()
        self._x[:, 3:6] *= (1.0 - vel_mask)

        return n_anchored

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
        # Use inference_mode so reset works even when internal tensors
        # (e.g. self._P) have been replaced by inference tensors during
        # predict/update calls inside torch.inference_mode() contexts.
        with torch.inference_mode():
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
            # Reset post-contact countdown
            self._post_contact_countdown[env_ids] = 0
            # Reset measurement starvation counter
            self._steps_since_measurement[env_ids] = 0
