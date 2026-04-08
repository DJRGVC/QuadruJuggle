# Perception Pipeline Handoff — EKF Mode

**From**: perception agent (`agent/perception`)
**To**: policy agent (`agent/policy`)
**Date**: 2026-04-08

## IMPORTANT: No EKF During Training

**Recommendation: train pi1 with `mode="d435i"` (raw noise), NOT `mode="ekf"`.**

NIS diagnostic (iter_021) showed the EKF is **30× worse** than raw noise during
training: EKF RMSE = 130mm vs raw RMSE = 4.4mm, NIS = 966 (target 3.0).

**Root cause (iter_024)**: the EKF's ballistic dynamics model is wrong during
**paddle contact** — it predicts gravity-driven freefall but the paddle's normal
force holds the ball. This ~9.81 m/s² unmodeled acceleration causes NIS explosion
regardless of coordinate frame. World-frame EKF (iter_023) eliminates pseudo-forces
but NIS=970 persists (same pattern as body-frame NIS=966).

**Fix**: `q_vel=7.0` (was 0.30) honestly represents the ~10 m/s² contact-force
uncertainty. NIS sweep results:

| q_vel | NIS | EKF pos mm | Raw pos mm | In band? |
|-------|------|------------|------------|----------|
| 0.30 | 970 | 131 | 4.4 | No |
| 1.0 | 95 | 33 | 4.3 | No |
| 3.0 | 13.5 | 10 | 4.3 | No |
| 5.0 | 6.3 | 6.2 | 4.3 | Marginal |
| 7.0 | ~3.5 | ~5.0 | 4.3 | Yes |
| 10.0 | 2.6 | 4.5 | 4.4 | Yes |

**Key insight**: with q_vel high enough for consistency, the EKF provides **no
position improvement** over raw D435i. Its value is: (1) velocity estimation,
(2) prediction during dropout frames, (3) consistent uncertainty quantification.

For training, the `d435i` mode with `noise_scale` curriculum is the right approach.
EKF is for **hardware deployment only** (velocity estimation + dropout bridging).

## What's Available

The perception pipeline has three modes:
- **`mode="oracle"`** — pass-through ground truth (default)
- **`mode="d435i"`** — D435i-style structured noise (RECOMMENDED for training)
- **`mode="ekf"`** — noise → EKF filter (for hardware deployment only)

Key features of `d435i` mode:
- Depth-dependent position noise (D435i stereo baseline model)
- Frame dropout (2% default)
- `noise_scale` curriculum support (0.0→1.0 ramp)
- Velocity noise from finite-differenced positions (30Hz camera model)

Key features of `ekf` mode (deployment):
- Stateful noise model with hold-last-value on dropout
- **World-frame EKF** (`world_frame=True`): ballistic dynamics in world frame,
  body↔world transforms via robot IMU orientation. Avoids pseudo-force artifacts.
- Body-frame EKF (legacy, `world_frame=False`): body-frame dynamics with
  acceleration compensation. Not recommended (NIS=1025).
- Latency buffer (configurable observation delay)
- ANEES/NIS consistency diagnostic

## How to Enable EKF Mode

### Option 1: Patch at runtime (like test_ekf_integration.py)

```python
from go1_ball_balance.perception import (
    BallObsNoiseCfg,
    ball_pos_perceived,
    ball_vel_perceived,
    reset_perception_pipeline,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

noise_cfg = BallObsNoiseCfg(mode="ekf")

# Replace ball obs terms
env_cfg.observations.policy.ball_pos = ObsTerm(
    func=ball_pos_perceived,
    params={
        "ball_cfg": SceneEntityCfg("ball"),
        "robot_cfg": SceneEntityCfg("robot"),
        "paddle_offset_b": _PADDLE_OFFSET_B,
        "noise_cfg": noise_cfg,
    },
)
env_cfg.observations.policy.ball_vel = ObsTerm(
    func=ball_vel_perceived,
    params={
        "ball_cfg": SceneEntityCfg("ball"),
        "robot_cfg": SceneEntityCfg("robot"),
        "noise_cfg": noise_cfg,
    },
)

# CRITICAL: add EKF reset event
env_cfg.events.reset_perception = EventTerm(
    func=reset_perception_pipeline,
    mode="reset",
    params={
        "ball_cfg": SceneEntityCfg("ball"),
        "robot_cfg": SceneEntityCfg("robot"),
        "paddle_offset_b": _PADDLE_OFFSET_B,
    },
)
```

### Option 2: In env_cfg.py directly

Same as above but in the `ObservationsCfg` and `EventCfg` classes.

## Worktree Isolation Note

Since the `go1_ball_balance` package is installed in editable mode, Python may
resolve imports from the policy agent's worktree (where the editable install
points). To ensure you get the latest perception code from `agent/perception`,
add this before importing `go1_ball_balance`:

```python
import os, sys
_SRC = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance",
))
sys.path.insert(0, _SRC)
```

Or, simpler: **cherry-pick / merge the perception files from `agent/perception`**
into your branch. The relevant files are:
- `source/go1_ball_balance/go1_ball_balance/perception/__init__.py`
- `source/go1_ball_balance/go1_ball_balance/perception/ball_obs_spec.py`
- `source/go1_ball_balance/go1_ball_balance/perception/ball_ekf.py`
- `source/go1_ball_balance/go1_ball_balance/perception/noise_model.py`

## Tunable Parameters

### BallObsNoiseCfg (top-level)
| Param | Default | Notes |
|---|---|---|
| `mode` | `"oracle"` | `"oracle"`, `"d435i"`, or `"ekf"` |
| `policy_dt` | `0.02` | Policy step period (50Hz); used by EKF predict |

### D435iNoiseModelCfg (noise_cfg.noise_model_cfg)
| Param | Default | Notes |
|---|---|---|
| `sigma_xy_base` | 0.002 | 2mm XY noise std |
| `sigma_z_base` | 0.003 | 3mm Z noise std |
| `sigma_z_per_metre` | 0.002 | +2mm/m depth-dependent Z |
| `dropout_prob` | 0.02 | 2% frame dropout |
| `latency_steps` | 1 | Observation delay |

### BallEKFConfig (noise_cfg.ekf_cfg)

**Tuned in iter_024** — q_vel set by contact-force uncertainty, not CWNA:

| Param | Default | Notes |
|---|---|---|
| `q_pos` | 0.003 | Position process noise std (m/√s) |
| `q_vel` | 7.0 | Velocity process noise std ((m/s)/√s) — covers ~10 m/s² contact force |
| `r_xy` | 0.002 | Measurement noise std, XY (m) — matches D435i sigma_xy_base |
| `r_z` | 0.004 | Measurement noise std, Z base (m) — 3mm + 2mm/m × 0.5m |
| `r_z_per_metre` | 0.002 | Additional Z noise std per metre of distance |
| `adaptive_r` | True | If True, r_z varies with estimated ball height |
| `drag_coeff` | 0.112 | Quadratic drag (ping-pong ball) |

**ANEES diagnostic**: `ekf.mean_nis` should be ~3.0 (mean of χ²(3)). Band [0.35, 7.81].
Access via `pipeline.diagnostics["mean_nis"]` when diagnostics enabled.

## Integration Test Results (iter_010)

- **4096 envs, 50 iterations** — no NaN, no crashes, no dimension mismatches
- Mean episode length: 21→123 (normal Stage A learning curve)
- Terminations: 99.95% ball_below, 0.05% ball_off, 0% timeout
- Rewards accumulating normally

## Required Changes to train_juggle_hier.py (policy agent)

The current `train_juggle_hier.py` on `agent/policy` needs two changes to support EKF mode:

1. **Add `"ekf"` to `--noise-mode` choices**:
   ```python
   choices=["oracle", "d435i", "ekf"],
   ```

2. **Add reset event when mode is "ekf"**:
   ```python
   if args_cli.noise_mode == "ekf":
       from go1_ball_balance.perception.ball_obs_spec import reset_perception_pipeline
       from isaaclab.managers import EventTermCfg as EventTerm
       from isaaclab.managers import SceneEntityCfg
       env_cfg.events.reset_perception = EventTerm(
           func=reset_perception_pipeline,
           mode="reset",
           params={
               "ball_cfg": SceneEntityCfg("ball"),
               "robot_cfg": SceneEntityCfg("robot"),
               "paddle_offset_b": _PADDLE_OFFSET_B,
           },
       )
   ```

Without the reset event, the EKF state will carry over between episodes and diverge.

## Noise Curriculum API (iter_013)

The `noise_scale` field on `BallObsNoiseCfg` lets you ramp D435i noise gradually
across curriculum stages. This matches Section 4.3 of `noise_curriculum_plan.md`.

### Static config (set once before training)

```python
noise_cfg = BallObsNoiseCfg(mode="d435i", noise_scale=0.25)  # 25% of full D435i noise
```

Scale multiplies all noise sigmas and dropout probability proportionally.
Latency is unchanged (always 1 step).

| `noise_scale` | XY σ (mm) | Z σ base (mm) | dropout % |
|---|---|---|---|
| 0.25 | 0.5 | 0.75 | 0.5% |
| 0.50 | 1.0 | 1.5 | 1.0% |
| 0.75 | 1.5 | 2.25 | 1.5% |
| 1.00 | 2.0 | 3.0 | 2.0% |

### Runtime curriculum update (during training)

```python
from go1_ball_balance.perception import update_perception_noise_scale

# In your curriculum callback (e.g., after stage transition):
update_perception_noise_scale(env, 0.5)  # ramp to 50%
```

This updates the live `PerceptionPipeline` instance on `env` — no env restart
needed. Works with both `d435i` and `ekf` modes.

### EKF mode with noise_scale

```python
noise_cfg = BallObsNoiseCfg(mode="ekf", noise_scale=0.5)
```

The EKF's measurement noise adapts to the scaled sensor noise. At `noise_scale=0`,
observations are noiseless (but still pass through the EKF dynamics model).

### World-Frame EKF (Recommended for Deployment)

```python
noise_cfg = BallObsNoiseCfg(mode="ekf", world_frame=True)
```

This runs the EKF in world frame. Robot orientation (`root_quat_w`) and position
(`root_pos_w`) are used to transform camera-frame measurements to world frame before
EKF update, and EKF outputs back to body frame for the policy. On real hardware,
the IMU provides the orientation — same architecture.

## Pipeline Status: FEATURE-COMPLETE for Sim Training (iter_025)

The sim perception pipeline (`mode="d435i"`) is the production-ready interface
for pi1 training. All sim-side work is complete:
- D435i-structured noise model with depth-dependent σ, dropout, latency
- Noise curriculum support (noise_scale 0.0→1.0)
- World-frame EKF validated (for deployment only — NIS consistent at q_vel=7.0)
- Comprehensive diagnostics (RMSE, NIS, detection rate)

**Known sim→real noise gaps** (to be addressed before hardware deployment):
- `sigma_z_per_metre`: sim=2mm/m, real≈5mm/m (2.5× too low)
- `dropout_prob`: sim=2%, real≈8–40% for white ball on D435i (4–20× too low)
- Fix: update noise_model.py before final pi1 training round

## Hardware Deployment Plan

See `docs/hardware_pipeline_architecture.md` for the full spec:
- D435i depth-only stream at 848×480 @ 90Hz (no ROS2 in hot path)
- YOLOv8n+P2 ball detection (TRT FP16, 3–4ms on Orin NX)
- World-frame EKF for velocity estimation + dropout bridging
- Same `(ball_pos_b, ball_vel_b)` output interface — pi1 needs no changes

Hardware implementation is blocked on physical Go1 + D435i access.
