# go1-ball-balance

> **Status: early development**

Training a [Unitree Go1](https://www.unitree.com/go1/) quadruped to balance a ping-pong ball on a paddle mounted to its back, using [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). The robot cannot touch the ball directly — it tilts and moves its body to keep the ball centered on the paddle (body-as-actuator balancing).

Two training phases:
1. **Privileged state** — ball position/velocity fed directly to the policy
2. **Egocentric camera** — policy trained from an on-board camera via teacher-student distillation

Target deployment: physical Go1 on Ubuntu 24.04.

---

## Training Pipeline

### Phase 1 — Privileged State (PPO)

| | |
|---|---|
| **Obs** | Ball pos + vel relative to paddle center, paddle orientation, robot proprioception (joint pos/vel, foot contacts, IMU) |
| **Act** | 12-DOF joint position targets |
| **Reward** | `exp(-k · ‖ball_xy − paddle_center_xy‖)` + penalty on body linear velocity |
| **Term** | Ball leaves paddle boundary |

Milestones:
- [ ] Static balance — robot standing still, ball starts on paddle
- [ ] Active correction — randomized ball starting offset, noisy state
- [ ] Balance under locomotion (optional / later)

### Phase 2 — Egocentric Camera (Teacher-Student)

A camera is mounted at a fixed transform on the base link, looking down at the paddle. A student policy maps the camera image to a latent that replaces the privileged ball observations, trained via DAgger/BC against the Phase 1 teacher.

Milestones:
- [ ] Simulated camera sensor integrated
- [ ] Student policy trained (CNN or DINOv2 encoder)
- [ ] Domain randomization: ball color, lighting, background, camera noise
- [ ] Sim-to-real rollout on Go1

---

## Stack

| Component | Choice |
|-----------|--------|
| Simulator | Isaac Lab (Isaac Sim) |
| Algorithm | PPO (Phase 1), DAgger/BC (Phase 2) |
| Robot | Unitree Go1 — 12 DOF |
| Vision encoder | CNN / DINOv2 |
| OS | Ubuntu 24.04 |

---

## Setup

> Installation instructions to be added once environment is confirmed.

Requires: NVIDIA GPU (Ampere or newer), CUDA 12.1+, Isaac Lab.

---

## References

- [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/)
- [GaussGym](https://gauss-gym.com/) — Gaussian Splatting + IsaacGym; candidate for improving visual sim-to-real in Phase 2
- [Holosoma](https://github.com/amazon-far/holosoma) / [holosoma-extensions](https://github.com/amazon-far/holosoma-extensions) — alternative framework considered but not used
