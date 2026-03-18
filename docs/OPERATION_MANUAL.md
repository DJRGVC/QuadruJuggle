# QuadruJuggle — Operation Manual

> Go1 ball juggling via hierarchical RL + mirror-law control in Isaac Lab / Isaac Sim 5.1.0

---

## System Architecture

```
pi1  (mirror-law, closed-form)
  │  1-D input: normalised apex height [0, 1]
  │  6-D output: [h, h_dot, roll, pitch, ω_roll, ω_pitch]  (physical → normalised)
  ▼
TorsoCommandAction  (frozen pi2, loaded from .pt checkpoint)
  │  12-D output: joint position targets
  ▼
Go1 robot + kinematic paddle
```

### Pi2 — Torso Tracking (RL policy)

- **Task**: track a 6-D command `[trunk_height, height_vel, roll, pitch, ω_roll, ω_pitch]`
- **Obs**: proprioception (joint pos/vel, foot contacts, IMU) + 6-D command
- **Network**: 2 hidden layers `[256, 128]`, ELU activation
- **Training**: PPO, 20 000 iterations, ~45 min on RTX 4090
- **Best checkpoint**: `logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt`
  - `model_best.pt` > `model_19999.pt`: final checkpoint over-converged to minimise motion
    penalties (action_rate, joint_torques), becoming too passive for juggling

### Pi1 — Mirror-Law Controller (closed-form)

Mirror law: choose paddle normal that reflects ball to target apex height.

```
n = normalize(v_out - v_in)
v_out_z  = sqrt(2 * g * apex_height)          # upward speed for target apex
v_out_xy = -K * ball_xy_rel_paddle            # lateral centering correction
pitch = atan2( nx_body, nz_body)
roll  = atan2(-ny_body, nz_body)
```

Energy sustaining strategy (h_dot impulse):
```
v_paddle_target = (v_out_z + e * |v_in_z|) / (1 + e)
h_dot_impulse   = 2 * v_paddle_target    # 2× to compensate tracking lag
```
The impulse fires when ball is descending and within 0.50 m above paddle.
A baseline `h_dot = 0.15` prevents body droop on springy legs between impacts.

### Kinematic Paddle

- Separate `RigidObjectCfg` with `kinematic_enabled=True`
- Teleported every 5 ms via `update_paddle_pose` event: `pos = robot_base + [0, 0, 0.07]`
- This gives zero spring/damping coupling to body — the "floating" look is by design
- Paddle shape: disc, scale `(1.8, 1.8, 1.0)` → effective radius ≈ 0.153 m

### Ball Physics

| Parameter | Value |
|-----------|-------|
| `linear_damping` | 0.01 (low — preserves energy) |
| `angular_damping` | 0.01 |
| `restitution` | 0.99 (near-elastic) |
| Reset `vel_z_mean` | 1.2 m/s upward (sustains juggling from step 0) |

---

## Training Pi2

```bash
cd /home/frank/berkeley_mde/QuadruJuggle
python scripts/rsl_rl/train_torso_tracking.py \
    --task Isaac-TorsoTracking-Go1-v0 \
    --num_envs 4096 \
    --headless
```

Checkpoints saved to `logs/rsl_rl/go1_torso_tracking/<TIMESTAMP>/`.
TensorBoard: `tensorboard --logdir logs/rsl_rl/go1_torso_tracking/`.

Training converges around iteration 5 000–8 000; `model_best.pt` is saved whenever
mean episode reward exceeds the previous best.

---

## Playing Mirror-Law Juggling

### Basic (ground-truth ball state)

```bash
cd /home/frank/berkeley_mde/QuadruJuggle
PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH \
python scripts/play_mirror_law.py \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --apex_height 0.20 \
    --num_envs 4
```

Expected: all 4 envs should sustain juggling for the full 1500-step episode (~30 s).

### With Perception Noise (no Kalman)

```bash
python scripts/play_mirror_law.py \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --ball_pos_noise 0.01 \
    --ball_vel_noise 0.10 \
    --num_envs 4
```

Typical degradation at `pos_noise=0.01 m, vel_noise=0.10 m/s`:
- Episode reward drops from ~+100 to −6 to −19
- Episode length drops from 1500 to 120–525 steps

### With Kalman Filter (position-only noise)

Enable Kalman by setting `use_kalman=True` (default) and providing only `ball_pos_noise`:

```bash
python scripts/play_mirror_law.py \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --ball_pos_noise 0.01 \
    --num_envs 4
```

`ball_vel_noise` is ignored when Kalman is active — velocity is estimated from
position tracking + gravity model, not provided directly.

---

## Kalman Filter Design

**File**: `source/go1_ball_balance/go1_ball_balance/tasks/torso_tracking/ball_kalman_filter.py`

| Property | Detail |
|----------|--------|
| State | `[px, py, pz, vx, vy, vz]` — 6D per environment |
| Transition model | Free flight under constant gravity (`b[2] = -½g·dt²`, `b[5] = -g·dt`) |
| Observations | Noisy position only (`[px, py, pz]`) — mimics a depth/stereo camera |
| Process noise `Q` | `pos_std=0.001 m`, `vel_std=3.0 m/s` (large to absorb contact discontinuities) |
| Measurement noise `R` | `pos_noise_std²` diagonal — matched to actual sensor noise |
| Batching | Fully vectorised over N environments; no Python loops |

### Key Design Choices

- **KF provides position smoothing only** — velocity is taken from the direct
  measurement (`ball_vel_noise_std` applied if > 0), not from KF estimation.
  Reason: KF-derived velocity (differencing noisy positions) is far noisier than
  a direct measurement. Using GT velocity alongside KF position is the correct
  separation of concerns. For a real robot with no velocity sensor, switch to
  using KF velocity explicitly.
- **High `process_vel_std`**: Contact events flip ball velocity discontinuously.
  A large Q on velocity lets the filter recover in 1–2 steps rather than diverging.
- **Auto-initialization**: On the first `step()` call, any uninitialized environment
  is seeded from the noisy measurement (zero velocity assumption) rather than zeros.
  This prevents the huge innovations that would occur if the filter started at world origin.

### Kalman Filter Tuning

| Parameter | Config field | Effect |
|-----------|-------------|--------|
| Measurement noise | `ball_pos_noise_std` | Match to actual sensor σ |
| Velocity process noise | `kalman_process_vel_std` | Higher → faster recovery after contact; lower → smoother estimates |
| Enable/disable | `use_kalman` | `True` by default when `ball_pos_noise_std > 0` |

---

## Configuration Reference

### `MirrorLawTorsoActionCfg`

| Field | Default | Description |
|-------|---------|-------------|
| `pi2_checkpoint` | MISSING | Path to frozen pi2 `.pt` file |
| `ball_cfg` | `SceneEntityCfg("ball")` | Ball scene entity |
| `paddle_offset_b` | `(0, 0, 0.07)` | Paddle centre in body frame [m] |
| `apex_height_min` | `0.05` | Min bounce height [m] |
| `apex_height_max` | `0.40` | Max bounce height [m] |
| `h_nominal` | `0.38` | Trunk height command [m] |
| `centering_gain` | `2.0` | Lateral correction gain [1/s] |
| `restitution` | `0.85` | Ball–paddle restitution (should match scene physics) |
| `ball_pos_noise_std` | `0.0` | Perception noise on ball position [m] |
| `ball_vel_noise_std` | `0.0` | Perception noise on ball velocity [m/s] |
| `use_kalman` | `True` | Enable KF when `ball_pos_noise_std > 0` |
| `kalman_process_vel_std` | `3.0` | KF velocity process noise [m/s/step] |
| `cmd_smooth_alpha` | `1.0` | EMA alpha for roll/pitch/h_dot (1=off, 0.25=strong). Use 0.25–0.4 with noisy velocity to reduce shaking. |

---

## Current Results (as of 2026-03-13)

| Condition | Flags | ep_len (steps / 1500 max) | ep_rew |
|-----------|-------|--------------------------|--------|
| Ground truth state | (default) | 1500 all 4 envs | ~+100 |
| `pos_noise=0.01 m, vel_noise=0.10 m/s`, no KF | — | 120–525 | −6 to −19 |
| `pos_noise=0.01 m, vel_noise=0.30 m/s`, no smoothing | — | 38–150 | −5 to −13 |
| `pos_noise=0.01 m, vel_noise=0.30 m/s` + KF + EMA 0.25 | `h_nominal=0.41` | **up to 1500** (1 env hit timeout); 400–900 typical | +50 to +273 |

**Best noise-robust config** (confirmed working 2026-03-13):
```bash
python scripts/play_mirror_law.py \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --ball_pos_noise 0.01 --ball_vel_noise 0.3 \
    --cmd_smooth_alpha 0.25 --h_nominal 0.41 \
    --num_envs 4
```

**How the perception stack works**:
- `ball_pos_noise > 0` → position is noisy → KF smooths position
- `ball_vel_noise = 0` → GT velocity used directly (no noise, more accurate than KF)
- `ball_vel_noise > 0` → noisy velocity injected; KF velocity used (physics model smoother than raw noise)
- `cmd_smooth_alpha < 1` → EMA on roll/pitch/h_dot commands prevents step-to-step oscillation from reaching pi2

### Known Limitations

1. **Pi2 h_dot bandwidth**: Body oscillates at ~0.125 Hz vs ball bounce at ~4 Hz.
   Pi2 cannot synchronise its upward stroke to impact timing. Juggling is sustained
   via near-elastic restitution (0.99) rather than active energy injection.
2. **Kinematic paddle coupling**: Compliant legs create phase lag between commanded
   and actual paddle height. The 0.50 m impact window and 2× h_dot amplification
   compensate for this.
3. **Restitution mismatch**: `MirrorLawTorsoActionCfg.restitution` and the scene
   physics `restitution` must be kept in sync manually.

---

## File Map

```
scripts/
  play_mirror_law.py          Mirror-law play + noise testing
  rsl_rl/train_torso_tracking.py   Pi2 training
  rsl_rl/train.py             Generic training launcher

source/go1_ball_balance/go1_ball_balance/tasks/
  torso_tracking/
    mirror_law_action.py        Pi1 mirror-law controller
    ball_kalman_filter.py       Batched Kalman filter for ball state
    action_term.py              TorsoCommandAction (pi2 wrapper)
    agents/rsl_rl_ppo_cfg.py    PPO hyperparameters
  ball_juggle_hier/
    ball_juggle_mirror_env_cfg.py   Full env config (ball + paddle + robot)
  ball_balance/mdp/
    events.py                   Ball reset functions (vel_z_mean)

logs/rsl_rl/go1_torso_tracking/
  2026-03-13_03-02-24/
    model_best.pt               Best checkpoint (use this, not model_19999.pt)
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Ball sits on paddle, no bounce | `restitution` too low or `linear_damping` too high | Set `restitution=0.99`, `linear_damping=0.01` |
| Ball bounces but decays in ~4 s | `restitution` < 0.99 | Increase restitution |
| Robot too passive, barely tilts | Using `model_19999.pt` (over-trained) | Use `model_best.pt` instead |
| FileNotFoundError `'...'` | Literal `...` passed as checkpoint path | Pass actual path to `.pt` file |
| Kalman filter giving wrong estimates | KF not initialized before first step | Fixed: `step()` now auto-inits from measurement |
| Play script crashes on import | Missing PYTHONPATH for rsl_rl | Prepend `PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH` |
