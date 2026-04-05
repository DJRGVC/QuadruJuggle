# QuadruJuggle — Operation Manual

> Go1 ball juggling via hierarchical RL + mirror-law control in Isaac Lab 2.3.2 / Isaac Sim 5.1.0
> Ubuntu 24.04 | Python 3.11 | NVIDIA ≥ 550.x driver | ≥ 8 GB VRAM

> **All commands assume:**
> ```bash
> export REPO=<path-to-this-repo>
> export ISAACLAB_PATH=~/IsaacLab   # adjust if different
> export PYTHONPATH=$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH
> cd $REPO
> conda activate isaaclab
> ```

---

## System Architecture

```
pi1  (high-level juggling controller)
  │  6-D normalized torso command
  │  [h_norm, ḣ_norm, roll_norm, pitch_norm, ω_roll_norm, ω_pitch_norm]
  ▼
TorsoCommandAction  (frozen pi2, loaded from .pt checkpoint)
  │  scales + offsets command → physical units
  │  runs inference on frozen pi2 MLP
  │  12-D joint position targets
  ▼
Go1 robot  +  kinematic disc paddle  +  ping-pong ball
```

Physical command mapping (from `TorsoCommandAction._CMD_SCALES / _OFFSETS`):

**Critical architecture clarification**

The system uses a hierarchical control architecture where a high-level controller (π1) outputs a 6D torso command, and a low-level controller (π2) tracks this command.

The π1→π2 interface is:
`[h, h_dot, roll, pitch, omega_roll, omega_pitch]`

π2 is a **torso pose tracking** policy, not a velocity controller. This distinction is critical: π1 directly specifies paddle orientation and vertical motion, enabling physically grounded control via reflection geometry.

| Dimension | Normalized | Physical range |
|-----------|-----------|----------------|
| `h` | [-1, 1] | [0.25, 0.50] m |
| `ḣ` | [-1, 1] | [-1, 1] m/s |
| `roll` | [-1, 1] | [-0.4, 0.4] rad |
| `pitch` | [-1, 1] | [-0.4, 0.4] rad |
| `ω_roll` | [-1, 1] | [-3, 3] rad/s |
| `ω_pitch` | [-1, 1] | [-3, 3] rad/s |

---

## Checkpoints

| Model | Path | Notes |
|-------|------|-------|
| Pi2 (best) | `logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt` | Use this, not `model_19999.pt` (over-trained) |
| Pi1 learned (best) | `logs/rsl_rl/go1_ball_juggle_hier/2026-03-20_20-25-22/model_best.pt` | Trained at fixed 0.20 m apex; 1741 iterations |

---

## Pi2 — Torso Tracking

### Training

```bash
python scripts/rsl_rl/train_torso_tracking.py \
    --task Isaac-TorsoTracking-Go1-v0 \
    --num_envs 4096 \
    --headless
```

Checkpoints → `logs/rsl_rl/go1_torso_tracking/<TIMESTAMP>/`
Monitor → `tensorboard --logdir logs/rsl_rl/go1_torso_tracking`

Converges around iteration 5 000–8 000. `model_best.pt` is saved when mean episode reward beats previous best.

### Reward Terms (Pi2)

| Term | Weight | Description |
|------|--------|-------------|
| `height_tracking` | +10.0 | Gaussian reward on trunk Z matching h_target |
| `height_vel_tracking` | +5.0 | Gaussian on ż matching ḣ_target |
| `roll_tracking` | +10.0 | Gaussian on roll matching roll_target |
| `pitch_tracking` | +10.0 | Gaussian on pitch matching pitch_target |
| `roll_rate_tracking` | +3.0 | Gaussian on ω_roll |
| `pitch_rate_tracking` | +3.0 | Gaussian on ω_pitch |

### Known Limitation

Pi2 h_dot bandwidth is ~0.125 Hz vs ball bounce ~4 Hz — pi2 cannot synchronise its upward stroke to ball impact timing. Juggling is sustained via near-elastic restitution (0.99) rather than active energy injection.

---

## Pi1 Variant A — Mirror Law (analytic, no training)

**File**: `source/go1_ball_balance/go1_ball_balance/tasks/torso_tracking/mirror_law_action.py`

### Math

```
n = normalize(v_out - v_in)
v_out_z  = sqrt(2 * g * apex_height)       # speed for target apex
v_out_xy = -K * ball_xy_rel_paddle         # lateral centering correction

pitch = atan2( nx_body, nz_body)
roll  = atan2(-ny_body, nz_body)
```

Energy sustaining (h_dot impulse):
```
v_paddle_target = (v_out_z + e * |v_in_z|) / (1 + e)
h_dot_impulse   = 2 * v_paddle_target      # 2× compensates tracking lag
```
Impulse fires when ball is descending and within 0.50 m above paddle.
Baseline `h_dot = 0.15` prevents body droop between impacts.

### Play Command

```bash
python scripts/play_mirror_law.py \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/<TIMESTAMP>/model_best.pt \
    --apex_height 0.20 \
    --num_envs 4
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--apex_height` | 0.20 | Target ball apex above paddle [m] |
| `--h_nominal` | 0.38 | Trunk height command [m] |
| `--centering_gain` | 2.0 | Lateral correction gain |
| `--ball_pos_noise` | 0.0 | Gaussian std added to ball position [m] |
| `--ball_vel_noise` | 0.0 | Gaussian std added to ball velocity [m/s] |
| `--cmd_smooth_alpha` | 1.0 | EMA alpha for roll/pitch/ḣ (1=off, 0.25=strong smoothing) |

### Recommended Noise-Robust Config

```bash
python scripts/play_mirror_law.py \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --ball_pos_noise 0.01 --ball_vel_noise 0.3 \
    --cmd_smooth_alpha 0.25 --h_nominal 0.41 \
    --num_envs 4
```

---

## Kalman Filter

**File**: `source/go1_ball_balance/go1_ball_balance/tasks/torso_tracking/ball_kalman_filter.py`

| Property | Value |
|----------|-------|
| State | `[px, py, pz, vx, vy, vz]` — 6D per env |
| Transition | Free flight under constant gravity |
| Observations | Noisy position only (mimics stereo camera) |
| Process noise Q | `pos_std=0.001 m`, `vel_std=3.0 m/s` (large for contact recovery) |
| Measurement noise R | `pos_noise_std²` diagonal |

KF is active when `ball_pos_noise_std > 0`. High `vel_std` lets the filter recover in 1–2 steps after ball bounce (velocity discontinuity). The EMA smoothing (`cmd_smooth_alpha=0.25`) is the main fix for robot shaking with noisy velocity; KF handles position estimation.

---

## Pi1 Variant B — Learned RL

**Files**:
- Action term: `ball_juggle_hier/pi1_learned_action.py`
- Env config: `ball_juggle_hier/ball_juggle_pi1_env_cfg.py`
- Train script: `scripts/rsl_rl/train_pi1.py`
- Play script: `scripts/play_pi1.py`

Pi1 is a PPO policy (MLP `[256, 128, 64]`, ELU) that directly outputs the 6-D normalized torso command. No mirror-law math. The frozen pi2 converts it to joint targets.

In addition to the analytic controller, we trained a learned π1 using PPO on top of the frozen π2, enabling a direct comparison between model-based and learned high-level control.

### Observations (46-D)

| Term | Dim | Description |
|------|-----|-------------|
| `ball_pos` | 3 | Ball position in paddle frame |
| `ball_vel` | 3 | Ball velocity in paddle frame |
| `base_lin_vel` | 3 | Robot base linear velocity |
| `base_ang_vel` | 3 | Robot base angular velocity |
| `projected_gravity` | 3 | Gravity vector in body frame |
| `joint_pos` | 12 | Joint positions (relative to default) |
| `joint_vel` | 12 | Joint velocities |
| `target_apex_height` | 1 | Normalized target apex height [0, 1] |
| `last_action` | 6 | Previous pi1 output (temporal context) |
| **Total** | **46** | |

π2 observes the 6D torso command produced by π1, not a desired base position.

### Reward Terms (Pi1 Training)

| Term | Weight | Description |
|------|--------|-------------|
| `alive` | +1.0 | Per-step survival bonus |
| `ball_apex_height` | +8.0 | Gaussian reward: ball reaches target apex height |
| `ball_bouncing` | +2.0 | Half-Gaussian on \|ball_vz\|: penalises cradling |
| `ball_xy_dist` | **-5.0** | Penalise ball lateral drift from paddle centre |
| `trunk_tilt` | **-5.0** | Penalise roll/pitch away from level |
| `trunk_contact` | **-5.0** | Penalise ball hitting robot body (not paddle) |
| `base_height` | -5.0 | Penalise trunk below 0.34 m |
| `base_height_max` | -8.0 | Penalise trunk above 0.43 m |
| `body_lin_vel` | -0.10 | Penalise lateral body drift |
| `body_ang_vel` | -0.05 | Penalise yaw/angular motion |
| `action_rate` | -0.005 | Penalise rapid pi1 command changes |
| `joint_torques` | -2e-4 | Energy penalty |
| `foot_contact` | -3.0 | Penalise airborne feet |
| `early_termination` | -10.0 | One-time penalty on non-timeout episode end |

> **Note**: Weights were reduced in v3 to prevent value-function explosion. The earlier values (`ball_apex_height=25`, `early_termination=-200`) caused `value_loss → inf` at ~6k iterations.

### Termination Conditions (Pi1)

| Term | Condition |
|------|-----------|
| `time_out` | Episode length ≥ 1500 steps (30 s) |
| `robot_tilt` | Trunk tilt > 0.5 rad (after 50 grace steps) |
| `ball_off` | Ball XY distance from paddle > 0.25 m |
| `ball_below` | Ball Z below paddle − 0.05 m |

### Variable Apex Height

Each episode, `randomize_apex_height` assigns a random target height per env from **[0.10, 0.60] m**. The policy observes the normalized target via `target_apex_height_obs` and learns to condition on it.

Gaussian std is automatically tightened at higher targets (harder) and loosened at lower targets (easier): `std ∈ [0.08, 0.12] m`.

### Training

```bash
python scripts/rsl_rl/train_pi1.py \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/<TIMESTAMP>/model_best.pt \
    --num_envs 1024 \
    --headless
```

Monitor: `tensorboard --logdir logs/rsl_rl/go1_ball_juggle_hier`

Key metrics to watch:
- `Episode/time_out` → should approach 1.0 (robot survives full episode)
- `Episode/alive` → should be > 0.95
- `Reward/ball_apex_height` → target ~0.8–1.0
- `Reward/ball_bouncing` → target > 2.0 (active bouncing, not cradling)
- `Reward/trunk_contact` → should approach 0.0 (no body cheating)

### Play

```bash
python scripts/play_pi1.py \
    --pi1_checkpoint logs/rsl_rl/go1_ball_juggle_hier/<TIMESTAMP>/model_best.pt \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/<TIMESTAMP>/model_best.pt \
    --apex_height 0.20 \
    --num_envs 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pi1_checkpoint` | required | Path to trained pi1 `.pt` |
| `--pi2_checkpoint` | required | Path to frozen pi2 `.pt` |
| `--num_envs` | 4 | Parallel environments |
| `--apex_height` | None | Fix apex height [m]. If omitted, randomises 0.10–0.60 m per episode |

---

## Scene Setup

### Kinematic Paddle

- Separate `RigidObjectCfg` with `kinematic_enabled=True`, `disable_gravity=True`
- Teleported every 5 ms via `update_paddle_pose` event: `pos = robot_base + quat_apply(rot, [0, 0, 0.07])`
- No spring/damping coupling to body — the "floating" look is by design
- Disc scale `(1.8, 1.8, 1.0)` → effective radius ≈ 0.153 m

### Ball Physics

| Parameter | Value | Reason |
|-----------|-------|--------|
| `restitution` | 0.99 | Near-elastic — preserves energy across bounces |
| `linear_damping` | 0.01 | Low — preserves energy |
| `mass` | 0.0027 kg | Ping-pong ball mass |
| `radius` | 0.020 m | Ping-pong ball radius |
| Reset `vel_z_mean` | 1.2 m/s | Starts ball airborne immediately |

---

## Results Summary

| Date | Controller | Apex | ep_len | ep_rew | Notes |
|------|-----------|------|--------|--------|-------|
| 2026-03-13 | Mirror law (GT) | 0.20 m | 1500 (4/4) | ~+100 | Baseline |
| 2026-03-13 | Mirror law + noise + KF + EMA | 0.20 m | 400–1500 | +50 to +273 | `vel_noise=0.3, alpha=0.25, h_nom=0.41` |
| 2026-03-20 | Learned pi1 (1741 iters) | 0.20 m fixed | 1500 (3/4) | ~+670 | Active bouncing (ball_vz ±2 m/s) |
| In training | Learned pi1 v2 | 0.10–0.60 m | — | — | + tilt termination, trunk contact penalty |

---

## Key Contributions (for report framing)

1. **Hybrid analytic + learned control architecture**  
   Combines a physics-based mirror law controller (π1) with a learned torso tracking policy (π2).  
   Enables interpretable control while retaining learned locomotion robustness.

2. **Direct comparison: analytic vs. learned high-level control**  
   Trained a PPO-based π1 on top of frozen π2.  
   Trade-off: RL π1 gives higher peak performance; mirror-law π1 gives stronger robustness and generalization.

3. **Noise-aware control pipeline**  
   Explicitly studies velocity-noise impact on control and shows learned-policy fragility outside training distribution.

4. **End-to-end perception → control pipeline**  
   Position-only observation + Kalman filtering to bridge toward real-world stereo deployment.

## Figure Interpretation Notes (for report text)

### Fig 1 — Noise Sensitivity
Observation: Learned π1 degrades sharply once velocity noise exceeds its training distribution (σ ≈ 0.1 m/s), while mirror law degrades approximately linearly.

Interpretation: Mirror law is grounded in reflection physics and does not rely on interpolation from training data; learned π1 fails to extrapolate reliably out-of-distribution.

Implication: Under noisy velocity estimation, analytic control is more robust for deployment.

### Fig 2 — Sample Efficiency
Training learned π1 requires an additional ~568M environment steps (~74% increase) versus mirror-law π1.

Since π2 is shared in both settings, this additional cost is entirely due to learning the high-level controller.

### Fig 3 — Mirror Law Geometry
Mirror law computes paddle normal from reflection consistency:
- Incoming and outgoing velocities define desired surface normal.
- This maps directly to roll/pitch commands.

The mapping is deterministic, interpretable, and distribution-invariant.

### Fig 5 — Command Stability
Raw velocity noise causes high-frequency pitch oscillations (RMSE ≈ 11°), destabilizing control.

Exponential smoothing reduces RMSE to ≈4.7° (43% of raw), showing instability is mainly estimation-noise-driven and can be mitigated with lightweight filtering.

### Fig 6 — Kalman Filter (correct interpretation)
While Kalman filtering improves position estimation smoothness, velocity estimation remains difficult around contact discontinuities.

In this setup, finite differences can yield lower velocity RMSE, while Kalman filtering provides smoother trajectories and robustness to missing data.

Therefore KF is retained for stability and real-world compatibility, not because it always wins on velocity RMSE.

## High-Level Comparison (for Results section)

| Method | Strength | Weakness |
|---|---|---|
| Mirror law π1 | Robust, interpretable, zero training cost | Slightly lower peak performance |
| Learned π1 | Higher in-distribution performance | Poorer OOD generalization, high training cost |

**Key takeaway**: Learned π1 improves in-distribution performance; mirror-law π1 provides stronger out-of-distribution robustness and eliminates high-level training cost.

## Core Insight (thesis-level)

The control structure should match task physics.

Ball bouncing follows reflection geometry, which is low-dimensional and analytically solvable.

Learning this mapping with RL introduces unnecessary sample complexity and can reduce robustness.

Learning is best reserved for π2, where dynamics are high-dimensional and difficult to model analytically.

## File Map

```
scripts/
  play_mirror_law.py               Mirror-law play (analytic pi1)
  play_pi1.py                      Learned pi1 play
  rsl_rl/
    train_torso_tracking.py        Train pi2
    train_pi1.py                   Train learned pi1

source/go1_ball_balance/go1_ball_balance/tasks/
  torso_tracking/
    action_term.py                 TorsoCommandAction: loads + runs frozen pi2
    mirror_law_action.py           Mirror-law pi1 (analytic)
    ball_kalman_filter.py          Batched 6-state Kalman filter
    agents/rsl_rl_ppo_cfg.py       PPO config for pi2
    mdp/rewards.py                 Pi2 tracking rewards
  ball_juggle_hier/
    ball_juggle_mirror_env_cfg.py  Mirror-law env (for play_mirror_law.py)
    ball_juggle_pi1_env_cfg.py     Learned pi1 training env
    pi1_learned_action.py          Pi1 action term (wraps TorsoCommandAction)
    mdp/
      __init__.py                  Re-exports all MDP terms
      events.py                    randomize_apex_height (per-env reset)
  ball_juggle/mdp/
    rewards.py                     ball_apex_height_reward, ball_bouncing_reward
    observations.py                target_apex_height_obs
  ball_balance/mdp/
    events.py                      Ball + robot reset
    rewards.py                     trunk_tilt, foot_contact, trunk_contact_penalty
    terminations.py                robot_tilt, trunk_height_collapsed, ball_off/below

logs/rsl_rl/
  go1_torso_tracking/
    2026-03-13_03-02-24/           Pi2 best checkpoint
  go1_ball_juggle_hier/
    2026-03-20_20-25-22/           Pi1 v1 checkpoint (fixed 0.20 m apex)
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Ball sits on paddle, no bounce | Restitution too low | Set `restitution=0.99` in scene |
| Robot too passive | Used `model_19999.pt` (over-trained) | Use `model_best.pt` |
| `FileNotFoundError: .../TIMESTAMP/...` | Literal "TIMESTAMP" in path | Use actual timestamp directory |
| `IndexError: tuple index out of range` on `action.shape[1]` | Policy received wrong obs type | Pass plain dict `{"policy": tensor}` to policy, not TensorDict |
| Robot shaking with noisy velocity | Raw noisy velocity fed to mirror law | Use `--cmd_smooth_alpha 0.25` |
| Robot raises head to hit ball (cheating) | No trunk contact penalty | Train with `trunk_contact_penalty` weight=-20 |
| Robot falls over after a while | No tilt termination during training | Train with `robot_tilt` termination (max_tilt=0.5 rad) |
| CUDA OOM during training | Too many envs for 8 GB VRAM | Reduce `--num_envs` to 1024 |
| Ball cradles instead of bouncing | No velocity reward | Ensure `ball_bouncing` reward is active (weight=5.0) |
| Play crashes with carb warnings | Normal Isaac Sim shutdown noise | Ignore — look for Python traceback above these lines |

---

## Known Limitations

1. **Pi2 bandwidth**: Body oscillates at ~0.125 Hz vs ball bounce at ~4 Hz. Juggling relies on near-elastic restitution, not active energy injection per bounce.
2. **Kinematic paddle lag**: Compliant legs create phase lag between commanded and actual paddle height. The 2× h_dot amplification in mirror law compensates.
3. **No velocity sensor sim-to-real**: Current system uses ground-truth `base_lin_vel`. On hardware, replace with state estimator output.
4. **Sim-to-real gap**: No domain randomization for mass, friction, or motor dynamics yet. Required before hardware deployment.
5. **Pi1 v1 trained at fixed 0.20 m**: The checkpoint in `2026-03-20_20-25-22` only generalizes to heights near 0.20 m. Retrain with `randomize_apex_height` for variable-height control.
