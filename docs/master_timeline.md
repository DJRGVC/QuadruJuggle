# QuadruJuggle Master Timeline

**Created:** 2026-03-11
**Last updated:** 2026-03-11

Quick reference for what's done, what's next, and why. For full details, see the
linked documents.

---

## Status Key
- [x] Done
- [~] In progress
- [ ] Not started

---

## Phase A: Privileged Training

| Step | Status | What | Why |
|---|---|---|---|
| A1 | [x] | Train pi2 (torso tracker) | Low-level controller: proprioception → joint targets |
| A2 | [x] | Validate pi2 circle walking | Confirms pi2 can track sustained velocity commands |
| A3 | [~] | Train pi1 (ball planner) | High-level controller: ball state → 8D torso commands |
| A4 | [ ] | Validate pi1 juggling | Must reach Stage D+ (0.25m apex), timeout >75% |

**Pi2 note:** Exhibits lean-then-catch gait. Functional but not a proper walking gait.
TODO in `train_torso_tracking.py`. May need fixing before sim-to-real.

**Pi1 status:** Currently on Stage F (0.42m apex). Looking good.

**Reference:** `scripts/rsl_rl/train_torso_tracking.py`, `scripts/rsl_rl/train_juggle_hier.py`

---

## Phase B: Domain Randomization + Latency + Observation Noise

**Why:** Every successful sim-to-real paper does this (Rudin 2022, Kumar 2021, Ma 2025).
Without it, policies memorize sim-specific dynamics and fail on real hardware. Must be
done BEFORE perception work — the perception pipeline is useless if the robot can't
stand on a real floor.

| Step | Status | What | Why |
|---|---|---|---|
| B1 | [ ] | Add DR to pi2 env config | Mass ±15%, friction [0.4,1.2], motor strength [0.85,1.15]×, pushes ±3N, obs noise |
| B2 | [ ] | Add latency modeling | Action delay: 1 physics step (5ms). Obs delay: 1 policy step (20ms) |
| B3 | [ ] | Retrain pi2 with DR (warm-start) | ~1-2 hours. Expect 10-20% initial drop, recovers. |
| B4 | [ ] | Add DR to pi1 env config | Same as pi2 plus ball mass [2.5,3.0]g, restitution [0.75,0.95], ball pos noise ±5mm |
| B5 | [ ] | Retrain pi1 with DR (warm-start from A3, on top of B3 pi2) | ~1-2 hours |
| B6 | [ ] | Validate DR policies match clean baseline within 60% | If >40% degradation, narrow DR ranges |

**Reference:** `docs/sim_to_real_plan.md` (Track 1), `docs/next_steps_plan.md` (Step 2)

---

## Phase C: EKF + Perception-Aware Training (sim-side)

**Why:** Replaces ground-truth ball state with estimated state from an Extended Kalman
Filter. The EKF is identical code in sim and real — the core sim-to-real bridge for
perception (Ma et al., Science Robotics 2025). No rendering, no CNN, no visual domain gap.

| Step | Status | What | Why |
|---|---|---|---|
| C1 | [ ] | Implement batched EKF in PyTorch | 6D state [pos,vel], ballistic+drag dynamics, predict@200Hz, update@60-120Hz |
| C2 | [ ] | Validate EKF on clean sim trajectories | Velocity error <5%, position <2mm RMS, no divergence over 30s |
| C3 | [ ] | Implement noise model (placeholder params) | σ = σ_base(3mm) + σ_dist(5mm/m)×d + σ_omega(2mm/(rad/s))×\|ω\|, 5-15% dropout |
| C4 | [ ] | Create perception-aware env config | Asymmetric actor-critic: actor=EKF output, critic=ground-truth |
| C5 | [ ] | Retrain pi1 with noisy EKF obs (warm-start from B5) | Policy learns to handle jitter, dropout, latency |
| C6 | [ ] | Validate: pi1 juggles within 80% of privileged baseline | If >20% degradation, check noise model params or add perception quality reward |

**The detection pipeline (real robot) converts camera frames to EKF input:**
```
Camera (120Hz) → BGR→HSV → threshold → contour → centroid (u,v) + diameter (d_px)
→ monocular depth: z = 0.040×f/d_px → back-project to 3D → transform to paddle frame
→ [x,y,z] → EKF update step
```
No neural network. <1ms compute. See `docs/perception_roadmap.md` Phase 4 for full detail.

**Reference:** `docs/perception_roadmap.md` (Phases 1-2), `docs/next_steps_plan.md` (Steps 3-4)

---

## Phase D: Hardware Preparation (parallel with B and C)

**Why:** No dependency on sim work. Camera + paddle assembly takes calendar time (ordering,
printing, calibrating). Start early so hardware is ready when sim work completes.

| Step | Status | What | Why |
|---|---|---|---|
| D1 | [ ] | Order global shutter USB camera (~$30) | 120fps, monocular depth from known ball size |
| D2 | [ ] | Design + 3D-print upward-facing camera bracket | Rigid mount on Go1's back, near paddle |
| D3 | [ ] | Mount paddle (170mm disc, 70mm above trunk) | Must match sim geometry exactly |
| D4 | [ ] | Assemble + route cables | Weigh full assembly → update DR center values |
| D5 | [ ] | Calibrate camera intrinsics | OpenCV checkerboard, 5 minutes |
| D6 | [ ] | Measure camera-to-paddle rigid transform | Fixed rotation + translation, calibrate once |

**Reference:** `docs/sim_to_real_plan.md` (Track 3), `docs/next_steps_plan.md` (Step 5)

---

## Phase E: System Identification + Calibration

**Why:** Sim actuators and ball physics don't match reality. System ID closes the gap.
ETH used CMA-ES for arm actuators. DeepMind found latency modeling "crucial."

| Step | Status | What | Why |
|---|---|---|---|
| E1 | [ ] | Actuator sine sweep characterization | Command 0.5-10Hz sine waves, record tracking. Identify PD gains, delay, friction |
| E2 | [ ] | Ball physics calibration | Drop from 5-30cm, measure bounce height. Compute real restitution. |
| E3 | [ ] | Update sim parameters if needed | If restitution differs >0.05 or actuator delay >10ms, update and retrain |
| E4 | [ ] | Real camera noise model calibration | Fixed ball + moving robot → regress σ(d,ω). ≥500 measurements. |
| E5 | [ ] | Update noise model, retrain pi1 if needed | If calibrated params differ >2× from placeholders |

**Critical check at E1:** If actuator bandwidth <10Hz, juggling may not be feasible
at 50Hz policy rate. This is the highest-risk finding in the entire project.

**Reference:** `docs/sim_to_real_plan.md` (Track 4), `docs/next_steps_plan.md` (Steps 6-7)

---

## Phase F: Deployment (progressive)

**Why:** Incremental validation isolates failures. If level N+1 fails but level N works,
you know exactly which component broke.

| Step | Status | What | Why |
|---|---|---|---|
| F1 | [ ] | Level 0: Comms check (no policy) | Verify SDK round-trip <5ms, proprioception feedback |
| F2 | [ ] | Level 1: Pi2 manual commands (no ball) | Verify height/tilt/velocity tracking on real robot |
| F3 | [ ] | Level 2: Pi2 circle walking | First real sim-to-real gap measurement |
| F4 | [ ] | Level 3: Pi2+Pi1 with external ball tracking | Control pipeline test, independent of onboard perception |
| F5 | [ ] | Level 4: Onboard perception, ball balancing | Camera + EKF on Jetson, gentle bouncing |
| F6 | [ ] | Level 5: Full juggling | Ramp up apex target, compare to sim |

**Debugging guide:**
- Falls immediately → actuator model mismatch (E1)
- Stands but won't walk → action delay not modeled (B2)
- Ball bounces wrong → restitution mismatch (E2)
- Ball tracking erratic → noise model too optimistic (E4)
- Works 5s then diverges → observation drift, need more proprioception noise (B1)

**Reference:** `docs/sim_to_real_plan.md` (Track 5), `docs/next_steps_plan.md` (Step 8)

---

## Calendar View

```
         Sim track                    Hardware track
         ─────────                    ──────────────
Week 1   [A] Validate pi1            [D1-D2] Order camera, design mount
         [B1-B2] Add DR + latency

Week 2   [B3-B5] Retrain pi2, pi1    [D3-D6] Assemble, calibrate
         [C1-C2] Build + validate EKF

Week 3   [C3-C5] Noisy EKF training  [E1-E2] Actuator + ball sys ID

Week 4   [C6] Validate perception    [E4-E5] Noise model calibration
         pi1

Week 5   Iterate based on deploy     [F1-F3] Comms, pi2, walking
         results

Week 6   Final tuning                [F4-F6] Ball balancing → juggling
```

---

## Documents Index

| Document | Contents |
|---|---|
| `docs/master_timeline.md` | This file — status tracker and quick reference |
| `docs/next_steps_plan.md` | Detailed 8-step execution plan with decision points |
| `docs/perception_roadmap.md` | EKF design, detection pipeline, noise model, all perception details |
| `docs/sim_to_real_plan.md` | Five sim-to-real gaps, domain randomization spec, deployment architecture |
