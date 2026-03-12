# Next Steps Plan: From Privileged Training to Real Hardware

**Created:** 2026-03-11
**Status:** Pi2 trained (lean-then-catch gait). Pi1 training in progress.

This is the actionable execution plan. For design rationale, see:
- `docs/perception_roadmap.md` — ETH-style EKF approach details
- `docs/sim_to_real_plan.md` — five sim-to-real gaps and how to close them

---

## Step 1: Validate Pi1 (now)

Pi1 is currently training. Once it finishes (early stop or max iterations), validate.

### How to validate

**1a. Check training logs for curriculum progression:**
```bash
# See which stage pi1 reached and how rewards progressed
uv run --active python scripts/tbdump.py logs/rsl_rl/go1_ball_juggle_hier/<latest_run>/
```

Look for:
- `Episode_Reward/ball_apex_height` — should be increasing over training
- `Episode_Termination/time_out` — should be >0.75 at advanced stages (robot surviving)
- `Episode_Termination/ball_off` + `ball_below` — should decrease over training
- `Train/mean_episode_length` — should approach 1500 (30s × 50Hz) at later stages

**1b. Play back the policy visually:**
```bash
uv run --active python scripts/rsl_rl/play.py \
    --task Isaac-BallJuggleHier-Go1-Play-v0 \
    --pi2-checkpoint logs/rsl_rl/go1_torso_tracking/<run>/model_best.pt \
    --num_envs 16
```

Watch for:
- Does the ball bounce at all? (basic competence)
- How high does it bounce? (curriculum stage reached)
- Does the robot stay standing? (pi2 stability under pi1 commands)
- Does the ball drift laterally off the paddle? (centering quality)
- Does the robot move erratically? (pi1 sending wild commands to pi2)

**1c. Success criteria:**

| Outcome | Interpretation | Next action |
|---|---|---|
| Stage D+ (0.25m+ apex), timeout >75% | Good — move to Step 2 | Proceed |
| Stage A-C, timeout >75% | Pi1 learning but slow — needs more training time or curriculum tuning | Increase max_iterations or reduce _BJ_THRESHOLD |
| Timeout <50% at any stage | Robot falling — pi2 instability under pi1 commands | Fix pi2 gait first (lean-then-catch issue) |
| Ball never bounces, just sits on paddle | Pi1 not discovering bouncing — likely init_noise_std or reward issue | Check ball_apex_height reward, verify ball spawn dynamics |
| Policy loads but crashes immediately | Likely pi2 checkpoint incompatibility or obs dimension mismatch | Check error logs, verify pi2 checkpoint matches expected architecture |

**Decision point:** If pi1 can't juggle past Stage C, the lean-then-catch pi2 gait
is likely the bottleneck. Fix pi2 before proceeding — add trunk_tilt penalty or
gait symmetry reward, retrain pi2, then retrain pi1 on top.

---

## Step 2: Domain Randomization + Latency + Observation Noise

**Why now:** This is the single most impactful thing before touching hardware. Every
successful sim-to-real paper does this (Rudin 2022, Kumar 2021, Ma 2025). Without
it, the policy memorizes sim-specific dynamics and fails on real hardware. The
perception pipeline is useless if the robot can't stand on a real floor.

**Method:** Warm-start from current checkpoints. Retrain pi2 first, then pi1 on
top of the DR-trained pi2.

### What to add to pi2 (torso_tracking_env_cfg.py)

```
Domain randomization (new EventCfg terms):
  - Base mass:         ±15%              (covers camera + paddle + cables)
  - Link masses:       ±10%              (manufacturing tolerances)
  - Friction:          [0.4, 1.2]        (floor surface variation)
  - Motor strength:    [0.85, 1.15]×     (actuator degradation, voltage)
  - Random pushes:     ±3 N, every 5-15s (external disturbances)
  - Gravity direction: ±0.05 rad tilt    (IMU calibration error)

Observation noise (on ObservationsCfg terms):
  - Joint positions:   ±0.01 rad Gaussian
  - Joint velocities:  ±0.1 rad/s Gaussian
  - Angular velocity:  ±0.02 rad/s Gaussian
  - Projected gravity: ±0.02 per axis Gaussian

Latency:
  - Action delay: 1 physics step (5ms) — joint commands take effect one step late
```

### What to add to pi1 (ball_juggle_hier_env_cfg.py)

All of the above, plus:
```
Ball physics randomization:
  - Ball mass:         [2.5g, 3.0g]      (regulation range)
  - Ball restitution:  [0.75, 0.95]      (ball wear, paddle surface)

Observation:
  - Ball position noise: ±5mm Gaussian   (preview of perception noise)
  - Observation delay: 1 policy step (20ms) — ball state from previous step
```

### Training

```bash
# Retrain pi2 with domain randomization (warm-start)
uv run --active python scripts/rsl_rl/train_torso_tracking.py \
    --task Isaac-TorsoTracking-Go1-v0 --num_envs 12288 --headless \
    --resume True --load_run <latest_pi2_run>

# Then retrain pi1 on top of DR-trained pi2 (warm-start)
uv run --active python scripts/rsl_rl/train_juggle_hier.py \
    --task Isaac-BallJuggleHier-Go1-v0 \
    --pi2-checkpoint logs/rsl_rl/go1_torso_tracking/<dr_run>/model_best.pt \
    --num_envs 12288 --headless \
    --resume True --load_run <latest_pi1_run>
```

Expected: ~1-2 hours each. 10-20% initial performance drop, recovering with training.

**Decision point:** If DR-trained policies degrade by >40% from clean baseline,
the randomization ranges are too wide. Narrow them (e.g., friction [0.5, 1.0],
mass ±10%) and retrain.

---

## Step 3: Build the EKF (can overlap with Step 2)

### What the EKF does

Replaces ground-truth ball state with estimated ball state. Tracks position and
velocity using a ballistic dynamics model + noisy position measurements.

### Specification

**State vector:** `[px, py, pz, vx, vy, vz]` — 6D, paddle frame

**Prediction step** (200Hz, every physics step):

```
Drag coefficient for 40mm ping-pong ball:
  drag_coeff = 0.5 × ρ × Cd × A / m
             = 0.5 × 1.225 × 0.4 × π×0.020² / 0.0027
             = 0.114 s⁻¹·m⁻¹

Acceleration:
  a = [0, 0, -9.81] - drag_coeff × |v| × v

Integration (Euler):
  v_new = v + a × dt                       (dt = 0.005s)
  p_new = p + v × dt + 0.5 × a × dt²
```

Covariance propagated via Jacobian of these dynamics (standard EKF).

**Update step** (60-120Hz, when measurement arrives):

```
innovation    = z_meas - H × x̂_predicted      (z = [px, py, pz], H = [I₃ | 0₃])
S             = H × P × Hᵀ + R                 (R = diag(σ²) from noise model)
K             = P × Hᵀ × S⁻¹                   (Kalman gain)
x̂_updated    = x̂_predicted + K × innovation
P_updated     = (I - K × H) × P
```

**When measurement drops out:** Skip update, predict-only. Ballistic model is
accurate to ~2mm over 50-100ms for a ping-pong ball.

### Implementation

Pure PyTorch, GPU-batched for 12288 envs. File: `perception/ball_ekf.py`

**Validation:** Run trained pi1 with ground-truth obs, record ball trajectories.
Feed same trajectories through EKF with clean measurements. Verify:
- Velocity estimate error <5% during free flight
- Position tracks within 2mm RMS
- No divergence over 30s episodes with bouncing
- Survives 5-frame measurement dropout

---

## Step 4: Perception-Aware Pi1 Retraining

Once EKF is validated (Step 3), swap pi1's ball observations.

### Sim training loop (each step)

```
1. Read ground-truth ball position from sim
2. Add noise:    pos_noisy = pos_true + N(0, σ(d, ω))
                 where σ = σ_base + σ_dist × d + σ_omega × |ω|
                 placeholder: σ_base=3mm, σ_dist=5mm/m, σ_omega=2mm/(rad/s)
3. Dropout:      with P=5-15%, skip measurement entirely
4. Latency:      use position from 1-2 steps ago (not current)
5. EKF update:   feed noisy/delayed measurement (or skip if dropout)
6. Actor obs:    [pos_est, vel_est] from EKF          ← noisy
7. Critic obs:   [pos_true, vel_true] from sim         ← clean (asymmetric)
```

### Perception quality reward (optional, following ETH)

Add a dense reward term (weight ~1-5):
```
r_percep = 1 / (1 + ||pos_est - pos_true||)
```
This encourages the policy to stabilize its body, reducing angular velocity →
reducing perception noise. In the ETH paper, this caused emergent active
perception behavior without any explicit FOV reward.

### Training

Warm-start from DR-trained pi1 (Step 2). Same curriculum. ~1-2 hours.

**Validation:** Pi1 should still juggle at comparable curriculum stages. Expect
10-20% performance degradation vs privileged baseline (consistent with ETH results
of near-100% clean vs 60-80% with real perception noise).

---

## Step 5: Hardware Setup (parallel with Steps 2-4)

No dependency on sim work. Do while sim training runs.

### Shopping list

| Item | Approx cost | Notes |
|---|---|---|
| Global shutter USB camera (e.g., ELP OV9281) | ~$30 | 120fps, 1MP, monocular |
| USB extension cable (short, low-latency) | ~$10 | Route along Go1 spine |
| 3D-printed camera bracket | ~$5 filament | Upward-facing, rigid, near paddle |
| Paddle disc (acrylic/aluminum, 170mm) | ~$15 | Must match sim: 170mm diameter, flat |
| Mounting hardware (M3 screws, standoffs) | ~$10 | Attach paddle + camera to Go1 back |
| Ping-pong balls (orange, regulation 40mm) | ~$5 | Match sim ball properties |

### Setup tasks

1. Design bracket in CAD — camera points straight up, mounted near paddle center
2. Print and assemble
3. Mount paddle at correct offset (70mm above trunk center, matching sim _PADDLE_OFFSET_B)
4. Route cables
5. Weigh full assembly → record for domain randomization center value
6. Calibrate camera intrinsics (OpenCV checkerboard)
7. Measure camera-to-paddle rigid transform (ruler + careful alignment)

---

## Step 6: System Identification (once hardware ready)

### 6A. Actuator characterization

```
For each joint (12 total):
  1. Command sine wave position trajectories at 0.5, 1, 2, 5, 10 Hz
  2. Record commanded vs actual positions via Go1 SDK at 500Hz
  3. Measure: tracking delay, overshoot, bandwidth
  4. In Isaac Lab: tune PD gains + actuator delay to match real response
  5. Optimize via CMA-ES or manual tuning
```

**Key parameters to identify:** effective Kp/Kd, actuator delay (expect 5-10ms),
joint friction (static + viscous), position/velocity limits.

**Critical check:** If actuator bandwidth is <10Hz on any leg joint, juggling may
not be feasible at 50Hz policy rate. This is the highest-risk finding.

### 6B. Ball physics calibration

```
1. Drop ball from 5, 10, 15, 20, 30 cm onto real paddle
2. Film with slow-mo phone camera (240fps)
3. Measure bounce height
4. Compute: e = sqrt(h_bounce / h_drop)
5. If |e_real - 0.85| > 0.05, update sim restitution and retrain
```

---

## Step 7: Noise Model Calibration

Once camera + detector are working on the Go1:

```
1. Fix a ping-pong ball at a known position (tape it to a stick)
2. Move the Go1 around (various distances, angular velocities)
3. Record camera detections vs known position
4. Also record with robot walking (pi2 circle mode) for ω-dependent noise
5. Vary lighting conditions (bright, dim, mixed)
6. ≥500 measurements across conditions
7. Regress: σ(d, ω) = σ_base + σ_dist × d + σ_omega × |ω|
8. Measure dropout rate as f(height, velocity, lighting)
9. Measure end-to-end latency (LED flash → detection timestamp)
```

If calibrated parameters differ from Step 4 placeholders by >2×, retrain pi1
with updated noise model (warm-start, ~1 hour).

---

## Step 8: Progressive Deployment

Build up incrementally. If any level fails, the previous levels tell you what
component broke.

### Level 0 — Comms check (no policy)

Send fixed joint position commands via Go1 SDK. Verify:
- Round-trip latency <5ms
- Command tracking is smooth
- Proprioception feedback rate matches expectations (500Hz)

### Level 1 — Pi2 only, manual commands (no ball)

Run pi2 on Jetson. Send 8D torso commands via keyboard/gamepad:
- Height: verify robot goes up/down
- Roll/pitch: verify tilt tracking
- vx/vy: verify walking
- Compare to sim behavior — first real sim-to-real gap measurement

### Level 2 — Pi2 circle walking

Same circle playback as sim validation. Robot should walk in circles
without falling. If it falls, iterate on system ID (Step 6A) or widen
domain randomization (Step 2).

### Level 3 — Pi2 + Pi1, external ball tracking (no onboard perception)

Use a laptop with a camera pointed at the robot, tracking the ball position
via ArUco marker or manual annotation. Send ball state to Jetson over WiFi.
This tests the control pipeline independently of onboard perception.

Start with ball balancing (ball resting on paddle). Then gentle 5cm drops.

### Level 4 — Add onboard perception

Switch from external tracking to onboard camera + EKF. Start with:
- Ball resting on paddle (EKF just tracks stationary position)
- Then gentle bouncing (5cm drop, Stage A equivalent)

### Level 5 — Full juggling

Ramp up drop height and apex target. Compare to sim performance at matched
curriculum stages. Iterate on noise model if performance gap is large.

---

## Decision Points Summary

| After | If | Then |
|---|---|---|
| Step 1 | Pi1 stuck at Stage A-C | Fix pi2 gait quality first |
| Step 1 | Pi1 reaches Stage D+ | Proceed to Step 2 |
| Step 2 | DR degrades performance >40% | Narrow randomization ranges |
| Step 6A | Actuator bandwidth <10Hz | Reduce policy to 25Hz or consider torque control |
| Step 6B | Real restitution differs >0.05 from sim | Update sim, retrain |
| Step 8 Level 2 | Pi2 falls on real robot | Iterate system ID or widen DR |
| Step 8 Level 3 | Balancing works with external tracking | Perception is the bottleneck → iterate noise model |
| Step 8 Level 3 | Balancing fails with external tracking | Control is the bottleneck → iterate system ID |

---

## Timeline

| Week | Sim work | Hardware work |
|---|---|---|
| Now | Validate pi1 (Step 1) | — |
| 1 | DR + latency + obs noise, retrain pi2 then pi1 (Step 2) | Order camera, design mount (Step 5) |
| 2 | Build + validate EKF (Step 3) | Assemble hardware, calibrate camera (Step 5) |
| 3 | Perception-aware pi1 retraining (Step 4) | Actuator sys ID + ball physics cal (Step 6) |
| 4 | Noise model calibration + retrain if needed (Step 7) | Deploy Level 0-2 (Step 8) |
| 5 | Final integration | Deploy Level 3-5 (Step 8) |
| 6 | Iterate based on deployment results | Full juggling attempts |
