# Sim-to-Real Robustness: Domain Randomization & Balance Rewards

**Created:** 2026-03-11

## Overview

Extensions to pi2 (torso tracker) for zero-shot sim-to-real transfer:
1. Three new balance/robustness reward terms
2. Seven domain randomization event terms

Pi1 (ball planner) inherits DR benefits via the frozen pi2 — only pi2 needs retraining with these changes.

---

## Reward Changes (balance robustness)

Three terms added based on DribbleBot (Ji et al. ICRA 2023) and ANYmal Parkour (Hoeller et al. Science Robotics 2024):

| Term | Weight | Function | Rationale |
|---|---|---|---|
| `body_ang_vel` | -0.10 | `body_ang_vel_penalty()` | Suppress oscillatory trunk rotations |
| `trunk_tilt` | -0.5 | `trunk_tilt_penalty()` | Smooth transitions between orientations |
| `foot_contact` | -1.5 (was -0.5) | `feet_off_ground_penalty()` | 3-foot stance now costs -1.5 vs +1.0 alive = unprofitable |

### Full pi2 reward table

| Term | Weight | Description |
|---|---|---|
| alive | +1.0 | Per-step survival |
| height_tracking | +5.0 | Gaussian (std=0.10), ungated |
| height_vel_tracking | +2.0 | Gaussian (std=0.5), gated on height |
| roll_tracking | +3.0 | Gaussian (std=0.05), gated on height |
| pitch_tracking | +3.0 | Gaussian (std=0.05), gated on height |
| roll_rate_tracking | +1.5 | Gaussian (std=1.5), gated on height |
| pitch_rate_tracking | +1.5 | Gaussian (std=1.5), gated on height |
| vx_tracking | +4.0 | Gaussian (std=0.20), gated on height + mask |
| vy_tracking | +4.0 | Gaussian (std=0.20), gated on height + mask |
| base_height | -5.0 | Linear penalty below 0.17m |
| base_height_max | -5.0 | Linear penalty above 0.53m |
| **foot_contact** | **-0.5** | Count of airborne feet (0-4) |
| **body_ang_vel** | **-0.05** | L2 angular velocity norm |
| **trunk_tilt** | **-0.25** | Projected gravity XY magnitude |
| **feet_slide** | **-0.25** | Foot XY speed while in contact (anti-slip) |
| action_rate | -0.05 | L2 action deltas |
| joint_torques | -5e-4 | L2 joint torques |

---

## Domain Randomization Parameters

Added to pi2 `EventCfg` for zero-shot sim-to-real transfer.

### DR parameter table

| Category | Parameter | Range | Mode | Rationale |
|---|---|---|---|---|
| **Mass** | Robot body masses | x[0.85, 1.15] (+-15%) | reset | Payload, battery, cable variation |
| **CoM** | Robot center of mass | +-20mm XY, +-10mm Z | reset | Asymmetric payload, battery position |
| **Motors** | Actuator stiffness | x[0.80, 1.20] (+-20%) | reset | Servo wear, temperature drift |
| **Motors** | Actuator damping | x[0.80, 1.20] (+-20%) | reset | Same |
| **Joints** | Joint friction | x[0.5, 1.5] (+-50%) | reset | Mechanical wear, lubrication variance |
| **Ground** | Static friction | [0.4, 1.5] | reset | Tile/carpet/rubber mat surfaces |
| **Ground** | Dynamic friction | [0.4, 1.5] | reset | Same |
| **Perturbation** | Push velocity (lin) | +-0.3 m/s XY | reset | Robustness to initial perturbations |
| **Perturbation** | Push velocity (ang) | +-0.2 rad/s RPY | reset | Same |
| **Perturbation** | External force | +-2.0 N (trunk) | interval | Wind, cable tug, steady disturbances |
| **Perturbation** | External torque | +-1.0 Nm (trunk) | interval | Same |

### Notes on range selection

- **Mass +-15%**: Go1 is 12kg; +-15% = +-1.8kg covers most payloads.
- **Ground friction [0.4, 1.5]**: narrower than Goalkeeper's [0.5, 4.5]; we don't need extreme high-friction since the robot isn't diving.
- **External force +-2N**: ~0.2g lateral at 12kg. Comparable to cable drag or light wind.
- **Motor gains +-20%**: standard across DribbleBot and Walk-These-Ways.
- **Joint friction x[0.5, 1.5]**: covers fresh to worn joints.

### What's NOT randomized (and why)

- **Gravity**: same for all envs (PhysX constraint), marginal benefit for flat-ground tasks
- **Ball restitution/friction**: not relevant to pi2 (no ball contact); will be added to pi1
- **Visual appearance**: not relevant (no camera in sim loop)
- **Joint limits**: Go1 limits are well-characterized, unlikely source of sim-real gap

---

## Academic References

| Paper | Venue | Key contribution to our DR |
|---|---|---|
| Huang et al. "Dynamic Quadrupedal Robotic Goalkeeper" | IROS 2023 | Ground friction [0.5,4.5], external wrench, ball perception noise |
| Ji et al. "DribbleBot" | ICRA 2023 | Mass/inertia/CoM/motor DR, foot contact rewards |
| Margolis & Agrawal "Walk-These-Ways" | RSS 2022 | Motor gain DR, velocity tracking std |
| Yu et al. "Rapid Locomotion via RL" | RSS 2023 | Push perturbations, joint friction |
| Hoeller et al. "ANYmal Parkour" | Sci. Robotics 2024 | Posture-gated rewards, angular velocity penalty |

---

## Training Order

1. Retrain pi2 with new rewards + DR (this doc)
2. Retrain pi1 with new pi2 checkpoint (uses updated `ball_xy_dist=-5.0`, no `paddle_catch`, CCD enabled)
3. Then proceed to EKF (perception_roadmap.md Phase 1)
