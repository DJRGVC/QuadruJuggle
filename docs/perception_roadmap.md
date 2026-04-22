# Perception Pipeline Roadmap: Privileged State → Real Hardware

**Created:** 2026-03-11
**Approach:** ETH-style noise-injected EKF (Ma et al., Science Robotics 2025)
**NOT teacher-student distillation** — see rationale below.

## Design Decisions

### Why EKF, not teacher-student distillation?
Teacher-student requires rendering cameras for 12288 envs during training (extremely
expensive in Isaac Lab), training a CNN student that may not match the teacher, and
domain randomization for visual appearance (ball color, lighting, shadows, background).
The EKF approach sidesteps all of this: inject a 6-number noise model instead of
rendering millions of images. The EKF is deterministic code that runs identically in
sim and real — no visual domain gap. The only gap is the noise model parameters,
calibrated from real data.

**Reference:** Ma, Cramariuc, Farshidian, Hutter. "Learning coordinated badminton
skills for legged manipulators." Science Robotics 10, 2025. They showed near-100%
success with perfect perception, 60-80% with real noise — and that's *with*
noise-injected training. Without it, the gap is catastrophic.

### Why does the hierarchy help?
Pi2 (torso tracker) doesn't see ball state at all — it just tracks torso commands.
Perception noise only affects pi1 (ball planner). We only need to make pi1 robust
to noisy observations; pi2 is completely isolated from perception. This is simpler
than the ETH flat-policy case where perception noise propagates through the entire
network.

```
Sim training:  ground-truth + noise model → EKF → pi1 (8D cmd) → pi2 (12D joints)
Real deploy:   camera + detector          → EKF → pi1 (8D cmd) → pi2 (12D joints)
                                            ^^^^
                                     identical code
```

### Why EKF, not a learned CNN estimator?
A CNN for state estimation doesn't solve the problem:
1. **You need velocity, not just position.** A single frame can't give velocity. Either
   use an EKF (physics model infers velocity from position history) or a temporal CNN
   (frame stack). The EKF is ~30 lines of math; a temporal CNN needs architecture design,
   training data, and a separate training pipeline.
2. **Where do CNN training images come from?** Rendering cameras for 12288 envs in Isaac
   Lab is 10-50× slower than headless training. Or collect real labeled images — requires
   hardware + ground-truth labels. The EKF needs zero rendered images.
3. **A CNN creates a new domain gap.** Sim-rendered images don't look like real images.
   You'd need visual domain randomization (lighting, background, colors). The EKF has
   zero visual domain gap — it's deterministic code, identical in both domains.
4. **The physics are known and simple.** A ball under gravity + drag is fully described by
   one equation. There's nothing to "learn." You use ML when the model is unknown or too
   complex to write down. This is neither.

HSV detection (classical) handles the 2D detection. The EKF handles the rest. If HSV
fails in some conditions, a tiny CNN *detector* can replace it while still feeding the
EKF — but this has not been needed even by ETH for a harder target (feathered
shuttlecock at 3-10m range).

### Key references
| Paper | Venue | Relevance |
|---|---|---|
| Ma et al. (ETH) | Sci. Robotics 2025 | Primary approach: EKF + noise model + asymmetric actor-critic |
| D'Ambrosio et al. (DeepMind) | 2024 | Latency modeling "crucial"; 27k-param CNN on raw Bayer |
| Su et al. (Berkeley, HITTER) | 2025 | Hierarchical planner+controller in Isaac Lab + PPO, zero-shot sim-to-real |
| Ziegler et al. (Tübingen) | 2025 | Event cameras 27.8× faster than frame-based (future option) |

---

## Phase 0: Privileged Training ← CURRENT

Train pi2 (torso tracker) and pi1 (ball planner) with ground-truth ball state in
simulation. These policies are the foundation — everything downstream depends on them.

### Validation criteria
- [ ] Pi2: sustained circle walking in play mode without falling
- [ ] Pi1: juggling to at least Stage D-E (0.25-0.30m apex) with timeout >75%
- [ ] Both checkpoints saved as `model_best.pt`

### Why this is first
The privileged pi1 generates realistic ball trajectories (bouncing, fast ascents,
lateral drift) that serve as the test data distribution for the EKF. Without it,
you'd validate the EKF on hand-crafted trajectories that don't represent real usage.

**Deliverable:** pi2 + pi1 checkpoints that reliably juggle in sim.

---

## Phase 1: EKF in Sim (clean measurements)

Build the EKF as a standalone PyTorch module. Feed it ground-truth ball position
from sim (no noise yet) and verify it reconstructs position + velocity accurately.

### Why this follows Phase 0
The trained pi1 generates the realistic ball trajectories the EKF needs for testing.
Also: you must verify the EKF works with perfect measurements before adding noise.
If it diverges on clean data, noise will only make it worse and you won't know whether
the bug is in the filter or the noise model.

### EKF specification

**State vector:**
```
x̂ = [px, py, pz, vx, vy, vz]    (6×1, paddle frame)
P  = 6×6 covariance matrix        (uncertainty in estimate)
```

**Predict step (200Hz — every physics step, dt=0.005s):**
Uses ballistic dynamics with quadratic air drag:
```
# Drag for 40mm ping-pong ball:
drag_coeff = 0.5 × ρ_air × Cd × A / m
           = 0.5 × 1.225 × 0.4 × π×0.020² / 0.0027
           = 0.114 s⁻¹·m⁻¹

# Acceleration
a = [0, 0, -9.81] - drag_coeff × |v| × v

# Euler integration
v_new = v + a × dt
p_new = p + v × dt + 0.5 × a × dt²

# Covariance propagation via Jacobian F = ∂f/∂x
P_predicted = F × P × Fᵀ + Q
where Q = diag([q_pos², q_pos², q_pos², q_vel², q_vel², q_vel²])
      q_pos ≈ 1e-4, q_vel ≈ 2e-3  (process noise — how much we distrust the model)
```

**Update step (60-120Hz — only when camera measurement arrives):**
```
z = [px_meas, py_meas, pz_meas]   (3D position from detector)
H = [I₃ | 0₃]                      (3×6 — we observe position, not velocity)
y = z - H × x̂_predicted            (innovation: measurement vs prediction)
S = H × P × Hᵀ + R                 (R = diag([σ²]) from noise model)
K = P × Hᵀ × S⁻¹                   (Kalman gain)
x̂_updated = x̂_predicted + K × y
P_updated = (I₆ - K × H) × P
```

**When measurement drops out:** Skip update entirely. Predict-only coasts on the
ballistic model. Accurate to ~1-2mm over 50-100ms for a ping-pong ball. The
covariance P grows each step (Q adds uncertainty), so when the next measurement
arrives, the large P makes K large and the filter snaps back.

**No DL component.** Zero learned parameters. ~50 FLOPs per step. The "parameters"
(Q, R, drag_coeff) are set from physics or measured from real data.

**Timing:**
```
Physics (200Hz):  |---|---|---|---|---|---|---|---|---|---|
                  P   P   P   P   P   P   P   P   P   P

Camera (60Hz):    |---------|---------|---(miss)--|---------|
                        U         U                   U

Policy (50Hz):    |---------|---------|---------|---------|
                        R         R         R         R

P=predict, U=update, R=pi1 reads latest [x̂,v̂]
```

- **Implementation:** pure PyTorch, GPU-batched for 12288 envs
- **Measurement input:** `[x, y, z]` — 3D position only (from detector or sim GT+noise)

### Validation criteria
- [ ] Velocity estimate error <5% during free flight (compare to ground-truth)
- [ ] Position estimate tracks ground-truth within 2mm RMS on clean data
- [ ] No divergence over full 30s episodes with bouncing
- [ ] Handles measurement dropout (skip updates) without blowing up

### Key files to create
```
source/go1_ball_balance/go1_ball_balance/perception/
├── __init__.py
├── ball_ekf.py          # batched EKF (PyTorch, runs on GPU)
├── noise_model.py       # perception noise model (placeholder params)
├── sim_perception.py    # Isaac Lab integration: GT → noise → EKF → obs
└── tests/
    └── test_ekf.py      # validation against ground-truth trajectories
```

**Deliverable:** `ball_ekf.py` passing all validation criteria on sim trajectories.

---

## Phase 2: Noise Model + Perception-Aware Retraining

### Why this follows Phase 1
The clean EKF is now validated. Adding noise tests whether the filter is robust, and
retraining pi1 makes the policy robust to the noise the filter can't remove.

The ETH insight: if you train with clean state and deploy with noisy state,
performance collapses. The policy must learn during training that observations have
jitter and occasional dropouts, so it develops compensating behaviors (e.g., relying
on proprioception during dropouts, not overreacting to velocity spikes).

### Step 2a: Noise model (placeholder parameters)
Position noise std = `σ_base + σ_dist × d + σ_omega × |ω|`
- `σ_base` ≈ 3mm (detector pixel noise at close range)
- `σ_dist` ≈ 5mm/m (depth error grows with distance)
- `σ_omega` ≈ 2mm/(rad/s) (motion blur from robot rotation)
- Detection dropout: P(miss) ≈ 5-15% per frame
- Latency: 1-3 policy steps (20-60ms) modeled as measurement delay

These are educated guesses. Phase 5 calibrates them from real data.

**Empirical residual-noise budget (2026-04-21):** the mirror-law controller (the simpler
hybrid substitute for the learned pi1) was swept against i.i.d. Gaussian noise on
ball position and velocity separately, at apex 0.30 m + vx = 0.1 m/s. Results at
n=20 envs per noise level, reported as drop rate with Wilson 95 % CI:

| Quantity | Safe (≤ ~20 % drop) | Knee |
|---|---|---|
| Position σ | ≤ 6 mm (0 %, CI 0–16 %) | 10 mm: 40 % (22–61 %) |
| Velocity σ | ≤ 0.05 m/s (5 %, CI 1–24 %) | 0.10 m/s: 70 % (48–85 %) |

(An earlier n=4 sweep suggested the position knee was at ~12 mm, but n=20 shows
the true rate at 10 mm is ~40 %. The n=4 CI overlapped this, so the smaller
sweep was coarse but not wrong. Details in `docs/mirror_law_noise_sweep.md`.)

**Design targets:** EKF post-filter residual σ ≤ 6 mm (position) and σ ≤ 0.05 m/s
(velocity). The educated-guess `σ_base ≈ 3 mm` above is comfortably inside the
safe region; `σ_base + σ_dist × d` must stay below 6 mm for paddle-distance
measurements (d ≈ 0.1–0.3 m gives ~0.5–1.5 mm from the distance term, so fine).
If the EKF cannot hold σ ≤ 6 mm on real hardware, the fallback is
perception-aware pi1 retraining (Step 2c) to recover the gap. Asymmetric
sensitivity — position noise dominates — means detector pixel precision matters
more than frame rate for a given sensor budget.

### Step 2b: Perception-aware env config
New env config replacing privileged ball obs with EKF output:
- **Asymmetric actor-critic:** actor gets `[pos_est(3), vel_est(3)]` from EKF;
  critic gets ground-truth `[pos(3), vel(3)]` (Ma et al. 2025)
- Sim loop each step: `ground_truth → add_noise → maybe_drop → EKF_update → obs`

### Step 2c: Retrain pi1
- **Warm-start** from privileged pi1 checkpoint
- Fine-tune with noisy EKF observations
- Same curriculum, but expect slightly slower progression due to noise
- Compare performance to privileged baseline (expect 10-20% degradation, per ETH)

### Validation criteria
- [ ] Pi1 juggles with noisy EKF obs (same curriculum stages as privileged)
- [ ] Performance within 80% of privileged baseline at matched stages
- [ ] Policy doesn't overreact to noise (smooth torso commands despite jittery obs)
- [ ] Policy recovers from 5-frame measurement dropout without falling

### Key files to create / modify
```
source/go1_ball_balance/go1_ball_balance/perception/
    sim_perception.py            # GT + noise → EKF wrapper for training
    noise_model.py               # parameterized noise model

source/go1_ball_balance/go1_ball_balance/tasks/ball_juggle_hier/
    ball_juggle_hier_percep_env_cfg.py   # new env cfg with EKF obs
    mdp/observations_percep.py           # EKF-based observation terms

scripts/rsl_rl/
    train_juggle_hier_percep.py          # training script for perception-aware pi1
```

**Deliverable:** pi1 that juggles in sim with realistic perception noise.

---

## Phase 3: Hardware Setup (parallel with Phases 1-2)

No causal dependency on sim work. Do this while Phases 1-2 are in progress.

### Camera selection
**Recommended: global shutter USB camera** (e.g., ELP OV9281, ~$30, 120fps, 1MP).
- Monocular depth from known ball diameter (40mm) — simpler than stereo
- Global shutter avoids motion blur on fast-moving ball
- Lightweight, easy to mount
- At 0-1m range, monocular depth via apparent size is accurate enough

Alternative: Intel RealSense D435i (RGB-D) — but IR structured light struggles
with small shiny balls at close range.

### Physical setup
- 3D-print upward-facing camera bracket for Go1's back, near paddle mount
- Camera must see ball from rest (z≈0) to 1m above paddle
- Fixed rigid transform from camera frame → paddle frame (calibrate once)
- USB cable routing to Go1's onboard compute

### Compute
- Go1 onboard: Jetson Nano (stock) or Jetson Orin Nano (upgrade)
- Detection + EKF must run <5ms per frame at 60-120Hz
- Pi1 + pi2 inference: <2ms per step at 50Hz (small MLPs, trivial on Jetson)

### Validation criteria
- [ ] Camera mounted, capturing frames, intrinsics calibrated
- [ ] Ball visible in frame from rest to 1m height
- [ ] Frame rate stable at target (60-120 fps)
- [ ] Camera-to-paddle rigid transform measured

**Deliverable:** Camera producing frames on Go1's Jetson.

---

## Phase 4: Real Detection Pipeline

### Why this follows Phase 3
You need a camera producing frames before you can write and test the detector.

### Full detection pipeline: RGB frame → EKF input

**Step 1: Find the ball in the image (HSV detection)**
```
1. Convert frame BGR → HSV
2. Threshold: keep pixels where H is in ball color range,
   S > 60 (saturated), V > 100 (bright)
   → binary mask (white where ball is, black elsewhere)
3. Morphological close (dilate then erode) — fill small holes
4. Find contours in the mask
5. Take the largest contour (the ball)
6. Fit minimum enclosing circle → gives:
   - (u, v): centroid in pixel coordinates
   - d_px: diameter in pixels
```

**Step 2: Get depth (monocular, from known ball size)**

The ball is exactly 40mm diameter. The camera has known focal length f (pixels,
from calibration). Apparent size tells us distance:
```
z = (real_diameter × f) / d_px
  = (0.040 × f) / d_px
```

Accuracy: at 0.5m, ±1 pixel error in d_px → ±16mm depth error.
At 1.0m, ±1 pixel → ±65mm. This distance-dependent degradation is why the noise
model has a σ_dist × d term.

**Step 3: Get X and Y (back-projection)**
```
# Camera intrinsics from calibration: fx, fy, cx, cy
x_cam = (u - cx) × z / fx
y_cam = (v - cy) × z / fy
```

**Step 4: Transform to paddle frame**

Camera is rigidly mounted on robot's back. The transform is measured once:
```
p_paddle = R_cam_to_paddle × [x_cam, y_cam, z] + t_cam_to_paddle
```

**Full pipeline per frame:**
```
Camera frame (8ms readout at 120fps)
    ↓
BGR → HSV                                          (~0.3ms)
    ↓
Threshold + morphology                             (~0.2ms)
    ↓
Contours → largest → centroid (u,v) + d_px         (~0.2ms)
    ↓
Depth:  z = 0.040 × f / d_px                      (~0 ms)
    ↓
3D:     x = (u-cx)×z/fx,  y = (v-cy)×z/fy         (~0 ms)
    ↓
Transform: p_paddle = R × p_cam + t                (~0 ms)
    ↓
Feed [x, y, z] to EKF update step                  (~0.02ms)
```

**Total compute: <1ms.** Bottleneck is camera readout (8ms at 120fps), not processing.
No neural network. No learned parameters. Just geometry and one division.

**When ball not detected:** No contour found → EKF skips update, predict-only coasts.
**Multiple orange blobs:** Take largest. Add size filter: reject <10px or >60px diameter.

### Calibration (once, before deployment)

1. **Camera intrinsics** [fx, fy, cx, cy]: OpenCV checkerboard, 20 photos, 5 minutes.
2. **HSV thresholds** [H_lo, H_hi, S_min, V_min]: Point camera at ball under
   deployment lighting, adjust until clean segmentation.
3. **Camera-to-paddle transform** [R, t]: Place ball at known position on paddle,
   read camera detection, solve for rigid transform. Or measure with ruler.

### Latency budget
HSV + contour: ~1ms on Jetson. Dominated by camera readout (~8ms at 120fps).
Total: <10ms shutter-to-position.

### Key files to create
```
source/go1_ball_balance/go1_ball_balance/perception/
    ball_detector.py        # HSV detection + monocular depth
    camera_config.py        # intrinsics, extrinsics, color thresholds
```

### Validation criteria
- [ ] Ball detected at >95% of frames when visible
- [ ] Position error <10mm at 0.5m range (against ruler/tape measure)
- [ ] Runs at target framerate on Jetson
- [ ] Handles ball at rest, slow motion, and fast (post-bounce) motion

**Deliverable:** Real-time ball `[x, y, z]` stream at 60-120Hz on Jetson.

---

## Phase 5: Noise Model Calibration

### Why this follows Phases 3+4
You can't measure perception noise without a working detector (Phase 4) on real
hardware (Phase 3).

### Data collection
1. Drop ball onto paddle at various heights, record camera detections
2. Simultaneously record ground-truth (mocap, or carefully measured drop heights)
3. Vary robot pose (tilted, walking via pi2) to capture ω-dependent noise
4. Vary lighting (bright, dim, mixed)
5. ≥500 drops across conditions

### Regression
- Detection error = `|camera_position - ground_truth|` per frame per axis
- Fit: `σ(d, ω) = σ_base + σ_dist × d + σ_omega × |ω|` (linear regression)
- Measure dropout rate as f(height, velocity, lighting)
- Measure end-to-end latency: strobe flash → detection timestamp

### Update pipeline
Plug calibrated parameters into `noise_model.py`, replacing Phase 2 placeholders.
If parameters differ significantly (>2× from placeholders), retrain pi1 from the
Phase 2 checkpoint with updated noise. This is cheap — warm-start, same curriculum.

### Validation criteria
- [ ] Noise model R² > 0.7 (explains most of the variance)
- [ ] Simulated noise distribution matches real detection error histogram
- [ ] If retrained, pi1 performance within 80% of privileged baseline

**Deliverable:** Calibrated noise model. Retrained pi1 if needed.

---

## Phase 6: Real Deployment

### Why this is last
Every component has been validated independently:
- EKF: validated on sim trajectories (Phase 1) and under noise (Phase 2)
- Detection: tested on real camera (Phase 4)
- Noise model: calibrated from real data (Phase 5)
- Pi1: trained against realistic perception noise (Phase 2, maybe updated Phase 5)
- Pi2: validated with circle walking (Phase 0)

If deployment fails, the phased validation lets you isolate which component broke.

### Deployment stack
```
Camera (120Hz) → HSV detect → monocular depth → EKF (identical to training)
→ [pos_est, vel_est] → pi1 (50Hz) → 8D torso cmd → pi2 (50Hz) → 12D joints
→ Go1 low-level controller (500Hz)
```

All on Go1's Jetson. Pi1 + pi2 exported as TorchScript or ONNX.

### Validation sequence (incremental)
1. **Static robot + ball drop:** verify EKF tracks real ball correctly
2. **Pi2 only (fixed commands) + ball drop:** verify perception under robot vibration
3. **Pi2 + pi1 + perception, ball balancing:** low apex target, verify stability
4. **Full juggling:** ramp up apex target, compare to sim performance
5. **Stress test:** varied lighting, nudge robot, off-center drops

### Success criteria
- [ ] Ball balancing for >30s with real perception
- [ ] Juggling to 0.10m apex (Stage A equivalent) for >10s
- [ ] Recovery from single missed detection (<5 consecutive dropouts)

**Deliverable:** Ball juggling on real Go1 hardware.

---

## Parallel Schedule

```
Week 1-2:  Phase 0 (finish privileged training)
           Phase 3 (order camera, mount, calibrate) ← parallel
Week 2-3:  Phase 1 (EKF in sim, clean)
           Phase 4 (real detection pipeline) ← parallel with Phase 1
Week 3-4:  Phase 2 (noise model + perception-aware retraining)
           Phase 5 (noise calibration from real data) ← parallel with Phase 2
Week 4-5:  Phase 6 (real deployment + iteration)
```

Phases 1-2 (sim-side) and 3-5 (hardware-side) run on parallel tracks.
Phase 6 merges both tracks.
