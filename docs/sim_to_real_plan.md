# Sim-to-Real Plan: QuadruJuggle on Real Go1

**Created:** 2026-03-11
**Status:** Planning. Pi2 trained (lean-then-catch gait). Pi1 training in progress.

This document covers everything needed to go from working sim policies to ball
juggling on a real Unitree Go1. It draws heavily from:
- Ma et al. (ETH, Sci. Robotics 2025) — perception, system ID, constrained RL
- D'Ambrosio et al. (DeepMind, 2024) — latency modeling ("crucial for good performance")
- Su et al. (Berkeley HITTER, 2025) — hierarchical control in Isaac Lab, zero-shot sim-to-real on Unitree G1
- Rudin et al. (CoRL 2022) — legged_gym domain randomization for quadrupeds
- Kumar et al. (Science Robotics 2021, RMA) — rapid motor adaptation for sim-to-real

---

## What Sim-to-Real Actually Requires

There are **five gaps** between simulation and reality. Each must be closed:

| Gap | What's different | How to close | Who demonstrated it |
|---|---|---|---|
| **Actuator dynamics** | Sim motors are ideal; real motors have friction, delay, bandwidth limits | System identification + domain randomization | ETH (CMA-ES), Kumar (RMA) |
| **Perception** | Sim has ground-truth ball state; real has noisy D435i depth | EKF + noise model (ETH approach) | Ma et al. 2025 |
| **Latency** | Sim is synchronous; real has pipeline delays | Explicit latency modeling in training | DeepMind 2024 |
| **Physics** | Sim contact/friction/restitution differ from real | Domain randomization + ball physics calibration | Rudin et al. 2022 |
| **Mechanical** | Sim robot is rigid; real has flex, backlash, paddle mounting tolerances | System ID + mechanical robustness | General |

---

## Track 1: Sim-Side Robustness (do this NOW, before hardware)

These changes make the sim policies robust to real-world conditions. They require
NO hardware and can be done while pi1 is still training or immediately after.

### 1A. Domain Randomization

Add randomization to the training environment. These are the standard randomizations
from Rudin et al. (2022) and Kumar et al. (2021) that have been shown to be necessary
for sim-to-real on quadrupeds:

| Parameter | Sim default | Randomization range | Why |
|---|---|---|---|
| Robot base mass | nominal | ±15% | Real Go1 has D435i (~90g), paddle, bracket, cables mounted |
| Link masses | nominal | ±10% | Manufacturing tolerances |
| Friction coefficient | 0.7 | [0.4, 1.2] | Floor surface varies |
| Restitution (ball) | 0.85 | [0.75, 0.95] | Ball wear, paddle surface variation |
| Ball mass | 2.7g | [2.5g, 3.0g] | Regulation range for ping-pong balls |
| Motor strength | 1.0× | [0.85, 1.15]× | Actuator degradation, voltage variation |
| PD gain perturbation | nominal | ±10% | Real PD tracking differs from sim |
| Random pushes | none | ±3 N, random interval 5-15s | External disturbances |
| Initial base orientation | ±0.3 rad yaw | ±0.1 rad roll/pitch too | Imperfect placement |
| Gravity direction | pure -z | ±0.05 rad tilt | IMU calibration error |

**Where to add:** New `EventCfg` terms in both `torso_tracking_env_cfg.py` and
`ball_juggle_hier_env_cfg.py`. Use Isaac Lab's `randomize_rigid_body_mass`,
`randomize_actuator_gains`, `push_by_setting_velocity` event terms.

**When to add:** After pi1 training converges on the current (clean) setup. Then
retrain with randomization — warm-start from clean checkpoint. Expect ~20% performance
drop initially, recovering with continued training.

### 1B. Latency Modeling

DeepMind found this "crucial for good performance." The real system has pipeline delays
that sim doesn't model:

| Component | Expected latency | How to model |
|---|---|---|
| D435i depth readout | 11ms (at 90Hz depth) or 33ms (at 30Hz) | — |
| HSV detect + depth lookup | <1ms | — |
| EKF update | <1ms | — |
| Pi1 inference | ~1ms | — |
| Pi2 inference | ~1ms | — |
| Go1 SDK communication | 2-5ms | — |
| **Total observation delay** | **~15-20ms** | **Delay observations by 1 policy step** |
| **Total action delay** | **~5ms** | **Delay actions by 1 physics step** |

**Implementation:** Add observation and action delay buffers to the env:
- Observation delay: the policy sees ball state from 1 policy step ago (20ms at 50Hz)
- Action delay: joint commands take effect 1 physics step later (5ms at 200Hz)

This is a simple ring buffer. Isaac Lab supports this via `ObservationTermCfg` with
history, or manually in the env step function.

**When to add:** Together with domain randomization (1A). Both are "robustness" changes
that should be trained together.

### 1C. Observation Noise on Proprioception

The sim provides perfect joint positions/velocities and IMU readings. Reality doesn't.

| Observation | Noise to add |
|---|---|
| Joint positions | ±0.01 rad Gaussian |
| Joint velocities | ±0.1 rad/s Gaussian |
| IMU angular velocity | ±0.02 rad/s Gaussian |
| IMU linear acceleration | ±0.05 m/s² Gaussian |
| Projected gravity | ±0.02 Gaussian per axis |

These values are from Rudin et al. (2022) and are conservative for the Go1's encoders
and IMU. Add as `enable_corruption = True` with noise models on the observation terms.

---

## Track 2: Perception Pipeline (see perception_roadmap.md)

This is the previously documented ETH-style approach. Summary of the phases:

1. **EKF in sim (clean)** — validate filter on trajectories from trained pi1
2. **Noise model + perception-aware retraining** — asymmetric actor-critic, retrain pi1
3. **Camera hardware** — Intel RealSense D435i, rear-paddle-mounted 45° upward
4. **Real detection** — HSV on RGB + D435i stereo depth lookup
5. **Noise calibration** — regress real noise model, update training if needed

Full details in `docs/perception_roadmap.md`.

**Key ETH lesson:** The perception error reward (weight=3, dense) enabled emergent
active perception. Consider adding a similar term for QuadruJuggle — reward the policy
for having a low EKF prediction error. This could encourage the robot to stabilize its
body (reducing angular velocity → reducing perception noise).

---

## Track 3: Hardware Preparation (parallel with Track 1)

### 3A. Physical Modifications to Go1

| Component | Details | Notes |
|---|---|---|
| **Paddle** | 170mm diameter disc, mounted on back | Must match sim geometry exactly (offset, mass) |
| **Camera** | Intel RealSense D435i, rear-paddle-mounted, 45° upward | USB3, ~90g; must see ball from rest to 1m above paddle |
| **Camera bracket** | 3D-printed, rigid, bolted behind paddle on Go1's back | Tilted 45° from horizontal toward zenith; must not flex — any deflection is unmodeled perception error |
| **Camera-paddle transform** | Rigid 6-DOF offset measured once at assembly | Critical: EKF converts D435i frame → paddle frame via this fixed transform |
| **Cabling** | USB3 cable routed along spine to Jetson | Secure to prevent snagging; USB3 required for D435i depth stream bandwidth |
| **Added mass** | Weigh full assembly (paddle + D435i + bracket ≈ 170g) | Record for domain randomization center value; heavier than mono camera (was ~25g) |
| **Paddle material** | Match sim restitution | Test real ball bounce height vs sim; adjust sim or paddle surface |

**D435i mount orientation — why 45° upward:**
The D435i sits behind the paddle, tilted 45° from horizontal toward zenith. This
balances two competing constraints: (1) the ball at rest on the paddle (z≈0.07m above
trunk) must be in-frame, and (2) the ball at 1m apex must also be in-frame. The D435i
has ~86°×57° FoV (depth stream). At 45° tilt, the vertical FoV spans roughly -14° to
+71° from horizontal — covering 0m (paddle surface) to >1m above. At 0° (straight up),
the ball at rest would be out of the depth minimum range (~0.1m). At 90° (horizontal),
the ball at apex would be out of frame. 45° is the sweet spot.

The camera-to-paddle transform is a fixed rigid offset: ~50mm behind paddle center,
~30mm below paddle surface, rotated 45° about the lateral (Y) axis. Measured once at
assembly and burned into the EKF config. Any flex in the bracket is unmodeled error.

### 3B. Compute Stack

The Go1 ships with:
- **Jetson Nano** (stock) — sufficient for HSV detection + depth lookup + small MLP inference
- Alternative: **Jetson Orin Nano** (~$200 upgrade) — more headroom; recommended if running D435i depth at 90Hz

Required software:
- Intel RealSense SDK 2.0 (`librealsense2`) — D435i driver, depth-color alignment, intrinsics
- OpenCV for HSV detection on aligned RGB frame
- PyTorch (or TorchScript/ONNX Runtime) for policy inference
- Unitree Go1 SDK (Python or C++) for joint command interface
- ROS2 (optional — useful for debugging, not required for deployment)

### 3C. Communication Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Jetson (onboard)                  │
│                                                     │
│  D435i (30-90Hz) ──→ HSV+Depth ──→ EKF (200Hz)     │
│                                      │              │
│                     Proprioception ──┤              │
│                                      ↓              │
│                               Pi1 (50Hz) ──→ 8D cmd │
│                                      │              │
│                     Proprioception ──┤              │
│                                      ↓              │
│                               Pi2 (50Hz) ──→ 12D   │
│                                      │              │
│                                      ↓              │
│                          Go1 SDK (500Hz)            │
│                              │                      │
└──────────────────────────────│──────────────────────┘
                               ↓
                     Go1 Low-Level Controller
                     (onboard MCU, 1kHz PD)
```

**Timing budget per policy step (20ms at 50Hz):**
- Read proprioception from SDK: 1ms
- Read latest EKF estimate: <1ms (runs async)
- Pi1 forward pass: ~1ms
- Pi2 forward pass: ~1ms
- Send joint commands via SDK: 2ms
- **Total: ~5ms active, 15ms idle** — plenty of headroom

---

## Track 4: System Identification

### 4A. Go1 Actuator Characterization

The Go1's low-level controller is a **black box PD controller** running at 1kHz on
the onboard MCU. We send joint position targets; it tracks them internally. We need
to characterize the tracking behavior.

**Method (following ETH's CMA-ES approach):**
1. Command sine wave joint trajectories at varying frequencies (0.5, 1, 2, 5, 10 Hz)
2. Record commanded vs actual joint positions (via SDK feedback at 500Hz)
3. Measure: tracking delay, overshoot, steady-state error, bandwidth
4. In Isaac Lab: tune PD gains + actuator model parameters to match real tracking
5. Optimize via CMA-ES: minimize |sim_trajectory - real_trajectory|

**What to identify:**
- Effective PD gains (Kp, Kd per joint)
- Actuator delay (command → response, typically 5-10ms)
- Joint friction (static + viscous)
- Joint damping
- Position/velocity limits (match real behavior at limits)

**Alternative (simpler):** If CMA-ES sys-ID is too involved, use **Rapid Motor
Adaptation (RMA, Kumar et al. 2021)** — train an adaptation module that infers
actuator parameters online from recent joint position history. This sidesteps
explicit system identification by learning to adapt at deployment time.

### 4B. Ball Physics Calibration

The ball-paddle interaction is critical for juggling. Sim restitution and friction
must match reality.

**Method:**
1. Drop ball from known heights (5, 10, 15, 20, 30cm) onto the real paddle
2. Measure bounce height (slow-motion phone camera, 240fps)
3. Compute effective coefficient of restitution: e = sqrt(h_bounce / h_drop)
4. Adjust sim restitution to match (currently 0.85)
5. Repeat at various angles to check friction model

**Expected findings:** Real restitution will depend on paddle surface material.
If it differs from 0.85 by >0.05, update sim and retrain.

---

## Track 5: Deployment and Validation

### 5A. Progressive Deployment Sequence

Do NOT attempt full juggling first. Build up incrementally:

**Level 0 — Comms check (no policy):**
- Send fixed joint position commands via SDK
- Verify round-trip latency and command tracking
- Confirm proprioception feedback rate and quality

**Level 1 — Pi2 only, fixed commands (no ball):**
- Run pi2 on Jetson, send 8D torso commands manually (keyboard/gamepad)
- Verify robot tracks height, roll, pitch commands
- Test velocity tracking (gentle walk commands)
- Compare to sim behavior — if different, iterate on system ID (Track 4A)

**Level 2 — Pi2 + circle walking:**
- Same circle playback as sim validation
- Robot should walk in circles without falling
- **This is the first real sim-to-real gap measurement**

**Level 3 — Pi2 + Pi1, ball balancing (no perception):**
- Use external tracking (phone camera + ArUco marker on ball, or manual)
- Feed ball state to pi1 via external system (laptop sends ball pos over network)
- Verify ball stays on paddle with approximate state estimation
- This tests the control pipeline without relying on onboard perception

**Level 4 — Add onboard perception:**
- Switch from external tracking to onboard camera + EKF
- Start with ball balancing (static ball on paddle — EKF just tracks position)
- Then gentle bouncing (5cm drop height)

**Level 5 — Full juggling:**
- Ramp up drop height / apex target
- Compare to sim performance at matched curriculum stages
- Iterate on noise model if performance gap is large

### 5B. Safety

| Concern | Mitigation |
|---|---|
| Robot falls and damages itself | Test on soft surface (gym mat). Go1 is robust to falls. |
| Wild joint commands | Clip joint targets to ±0.5 rad from default before sending to SDK |
| Ball flies off and hits something | Test in enclosed area. Ping-pong ball is 2.7g — not dangerous |
| Overheating | Monitor motor temperatures via SDK. Add cooldown pauses. |
| Emergency stop | Unitree remote has e-stop. Keep within arm's reach. |

### 5C. Debugging Sim-to-Real Gaps

When reality doesn't match sim (it won't, at first), diagnose systematically:

| Symptom | Likely cause | Fix |
|---|---|---|
| Robot falls immediately | Actuator model mismatch | System ID (Track 4A) or add more domain randomization |
| Robot stands but won't walk | Action delay not modeled | Add latency buffer (Track 1B) |
| Ball bounces differently | Restitution/friction mismatch | Ball physics calibration (Track 4B) |
| Ball tracking is erratic | Perception noise worse than model | Recalibrate noise model (perception Track 5) |
| Robot oscillates/vibrates | PD gains too high in sim vs real | Reduce sim stiffness or add joint damping randomization |
| Works for 5s then diverges | Observation drift (IMU, joint encoders) | Add proprioception noise to training (Track 1C) |

---

## Execution Order and Dependencies

```
                    ┌──────────────────────────────────┐
                    │  Phase 0: Privileged Training     │
                    │  (pi2 done, pi1 in progress)      │
                    └──────────┬───────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ↓                ↓                ↓
    ┌─────────────────┐ ┌───────────┐ ┌─────────────────┐
    │ Track 1: Sim    │ │ Track 2:  │ │ Track 3: HW     │
    │ Robustness      │ │ Perception│ │ Preparation      │
    │                 │ │ Pipeline  │ │                  │
    │ 1A. Domain rand │ │ (see      │ │ 3A. Paddle+cam   │
    │ 1B. Latency     │ │ perception│ │ 3B. Compute      │
    │ 1C. Obs noise   │ │ roadmap)  │ │ 3C. Comms arch   │
    └────────┬────────┘ └─────┬─────┘ └────────┬────────┘
             │                │                 │
             │     ┌──────────┘                 │
             ↓     ↓                            ↓
    ┌─────────────────┐              ┌─────────────────┐
    │ Retrain pi1+pi2 │              │ Track 4: Sys ID │
    │ with all robust │              │ 4A. Actuator    │
    │ additions       │              │ 4B. Ball physics │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             └──────────┬─────────────────────┘
                        ↓
              ┌─────────────────┐
              │ Track 5: Deploy │
              │ Level 0 → 5    │
              │ Progressive     │
              └─────────────────┘
```

### Suggested Calendar

| Week | Sim work | Hardware work |
|---|---|---|
| 1 | Finish pi1 privileged training | Order D435i, design rear-paddle bracket (45° upward) |
| 2 | Track 1: domain rand + latency + obs noise; retrain pi2+pi1 | 3D print bracket, assemble D435i + paddle, verify depth stream, calibrate camera-paddle transform |
| 3 | Track 2: EKF in sim (clean), then noisy | Track 4A: actuator sine sweeps, sys ID |
| 4 | Track 2: perception-aware pi1 retraining | Track 4B: ball bounce calibration |
| 5 | Integrate all robustness + perception, final retrain | Track 5: Level 0-2 (comms, pi2 only, walking) |
| 6 | Iterate based on Level 2 results | Track 5: Level 3-4 (ball balancing, onboard perception) |
| 7 | Final tuning | Track 5: Level 5 (full juggling) |

---

## What Could Go Wrong (Honest Assessment)

**High risk:**
- Go1 actuator bandwidth may be too low for the rapid corrections juggling requires.
  The Go1's motors are geared (not quasi-direct-drive like ETH's DynaArm). If the
  PD controller can't track pi2's commands at the required bandwidth, ball contact
  timing will be off. Mitigation: test actuator bandwidth early (Track 4A) and
  potentially reduce policy frequency from 50Hz to 25Hz if needed.

- Ball physics mismatch. Sim contact models are approximate. Ping-pong ball contact
  with a real paddle involves complex dynamics (spin, deformation, surface texture).
  Mitigation: extensive ball physics calibration (Track 4B) and wide domain
  randomization on restitution.

**Medium risk:**
- Pi2's lean-then-catch gait may not transfer. The current gait exploits sim
  physics (perfect friction, no flex). Real Go1 may slip or flex differently.
  Mitigation: fix the gait quality before sim-to-real (noted as TODO in
  train_torso_tracking.py), and domain randomization on friction.

- Camera motion blur at high angular velocities. The robot tilts/rotates during
  juggling. Mitigation: D435i IR stereo cameras are global shutter (no rolling
  shutter blur); RGB is rolling shutter but only used for HSV detection (tolerant
  to moderate blur). IR dot pattern can smear at high angular rates → noise model
  σ_omega term accounts for this.

**Low risk:**
- Compute budget on Jetson. The total pipeline is <5ms per step — well within
  the 20ms budget. Pi1 and pi2 are small MLPs (~50k params each).

- Ball detection. HSV filtering on a colored ping-pong ball at close range is
  a solved problem. D435i adds native stereo depth — eliminates the mono depth
  estimation error that would have been the dominant noise source. ETH used HSV
  for a harder target (feathered shuttlecock at 3-10m) with success.
