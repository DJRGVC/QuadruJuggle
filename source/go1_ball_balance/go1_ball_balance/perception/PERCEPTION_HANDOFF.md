# Perception Pipeline Handoff — EKF Mode

**From**: perception agent (`agent/perception`)
**To**: policy agent (`agent/policy`)
**Date**: 2026-04-08

## What's New

The perception pipeline now has a fully functional **EKF mode** (`mode="ekf"`)
that chains: GT ball state → D435i structured noise → 6-state Kalman filter →
filtered position & velocity. This replaces the stateless `mode="d435i"` with
a temporally-consistent, physically-grounded estimate.

Key improvements over `d435i` mode:
- **Stateful noise model**: hold-last-value on dropout (not GT passthrough)
- **Latency buffer**: configurable observation delay (default 1 step)
- **EKF smoothing**: ballistic+drag dynamics model predicts between measurements
- **Proper resets**: EKF state is re-initialized per-env on episode reset

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
| Param | Default | Notes |
|---|---|---|
| `process_noise_pos` | 0.01 | Q diagonal for position |
| `process_noise_vel` | 0.1 | Q diagonal for velocity |
| `meas_noise_pos` | 0.005 | R diagonal for position |
| `drag_coeff` | 0.112 | Quadratic drag (ping-pong ball) |

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

## What's Next (perception agent side)

1. Oracle vs EKF comparison (when GPU frees)
2. EKF parameter tuning based on training gap
3. Noise curriculum coordination (perception noise schedule synced with juggle curriculum)
