# Perception Pipeline — Key References

**Scope:** Papers that directly inform the perception architecture for QuadruJuggle.

---

## Primary Reference: ETH Noise-Injection Approach

### Ma et al., "Robotic Table Tennis with Model-Free Reinforcement Learning", Science Robotics 2025, ETH RSL

**Architecture:** Instead of rendering camera images in simulation and training a vision encoder,
they inject structured noise onto ground-truth object state during RL training. The policy learns
to be robust to the noise distribution that matches the real sensor. At deployment, a real
perception pipeline (camera + detector + EKF) produces noisy estimates that fall within the
trained noise envelope.

**Key design choices adopted in QuadruJuggle:**
- **No camera rendering during training** — GT ball state + noise injection + EKF, batched
  across all envs on GPU. Orders of magnitude faster than rendering 12,288 cameras.
- **Noise model matches real sensor characteristics** — position noise is depth-dependent
  (larger σ_z at greater distance), with stochastic dropout simulating detection failures.
- **EKF runs identically in sim and on hardware** — same code, same gains. The only
  difference is the measurement source (GT+noise in sim vs. real camera detections on hardware).
- **Only the high-level policy (pi1) needs perception-aware retraining** — the low-level
  controller (pi2) receives clean torso commands and is isolated from perception noise.

**Relevance to our D435i pipeline:**
- D435i depth noise model: σ_xy ≈ 2 mm base, σ_z ≈ 3 mm + 2 mm/m (depth-dependent)
- Dropout rate ~2% (IR reflection failures on shiny surfaces)
- Velocity noise derived from finite-differenced position at 30 Hz camera rate
- See `ball_obs_spec.py` for implementation

---

## Ball Manipulation on Legged Robots

### Ji et al., "DribbleBot: Dynamic Legged Manipulation in the Wild", ICRA 2023, MIT

**Task:** Quadruped soccer-ball dribbling — the most directly analogous prior work.

**Perception relevance:**
- Used onboard camera with depth for ball detection
- Gaussian proximity reward (height-gated) — adopted in our `ball_on_paddle_exp`
- Body-velocity penalty to prevent the robot from swinging its trunk to influence the ball —
  adopted in our `body_lin_vel` and `body_ang_vel` penalties
- Foot contact constraints to maintain stable stance — adopted in our `feet_off_ground_penalty`

---

## Sigma Curriculum

### Portela et al., "ROGER: Reinforcement Learning for Object Grasping", RSS 2025, Oxford

- Sigma curriculum on Gaussian proximity reward: std decays 0.50 → 0.08 m as
  `mean_episode_length` stabilises
- Directly adopted in our 7-stage curriculum (σ: 250 mm → 80 mm)
- σ floor of 80 mm prevents reward from becoming too peaky (gradient vanishes for
  ball offsets > 2σ)

---

## Event-Based Perception (Not Adopted)

### Forrai et al., 2023 — Event-Based ANYmal Badminton (if it exists)

**Status:** Paper not available in this repository. Referenced in fix_plan but could not be located.

**Sensor modality difference vs. our approach:**
- Event cameras output asynchronous pixel-level brightness changes (not frames)
- Extremely high temporal resolution (~μs) but no direct depth information
- Our D435i provides synchronous RGB + depth frames at 30–90 Hz
- The ETH noise-injection approach (Ma et al.) is a better architectural match for our
  frame-based D435i than event-camera methods

**If found later:** Compare their state estimation pipeline (event → ball state) against our
D435i pipeline (depth frame → 3D detection → EKF). Key question: does the event camera's
temporal resolution advantage matter for ball trajectories at our apex heights (0.1–1.0 m)?

---

## Distillation (Future — Stage 4)

### Ross et al., "A Reduction of Imitation Learning to No-Regret Online Learning", AISTATS 2011

- DAgger algorithm for teacher-student distillation
- Planned for Stage 4: teacher (GT state policy) labels student (camera observation) states
- Corrects distributional shift that pure behaviour cloning suffers from

### Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024

- Pre-trained vision encoder candidate for Stage 4 student network
- Fine-tune on synthetic ball/paddle imagery, freeze backbone, train small adapter head via DAgger

---

## Summary: Why Noise-Injection over Teacher-Student for Our Current Stage

| Approach | Pros | Cons | When to Use |
|---|---|---|---|
| **Noise injection (Ma et al.)** | No rendering overhead; same EKF code sim↔real; policy naturally robust to sensor noise | Requires accurate noise model; no visual feature learning | Stage 2–3 (current) |
| **Teacher-student / DAgger** | Learns visual features; handles unknown noise distributions | Requires camera rendering; distributional shift risk; slower training | Stage 4–5 (future) |

Our current approach (noise injection) is the right choice for Stage 2–3 because:
1. We have a well-characterised sensor (D435i depth accuracy specs are published)
2. Ball detection in depth images is straightforward (known size + colour)
3. Rendering 12,288 cameras would be prohibitively slow
4. The EKF code transfers directly to hardware
