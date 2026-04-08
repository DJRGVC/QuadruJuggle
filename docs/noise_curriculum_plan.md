# Noise Curriculum Plan: Pi1 Retraining with Perception Noise

**Author**: policy agent | **Date**: 2026-04-08 | **Status**: Draft

## 1. Background

The perception agent has built a D435i noise model (`ball_obs_spec.py`) with three
modes: oracle (GT passthrough), d435i (structured sensor noise), and ekf (full
noise → Kalman filter pipeline). The policy agent's job is to retrain pi1 to be
robust to this noise.

### Baseline comparison (500 iters, 12288 envs, 41D pi2)

| Metric              | Oracle (iter_003) | D435i (iter_005) | Delta     |
|---------------------|-------------------|------------------|-----------|
| mean_episode_length | 1500 (maxed)      | 1439             | -4.1%     |
| timeout %           | 98.9%             | 85.2%            | -13.7 pp  |
| ball_apex_height    | 2.92              | 3.04             | +4.1%     |
| ball_below %        | 1.1%              | 14.8%            | +13.7 pp  |
| mean_reward         | 78.5              | 72.2             | -8.0%     |
| noise_std           | 0.337             | 0.548            | +63%      |
| Curriculum reached  | Stage D           | Stage D          | Same      |

**Key finding**: D435i noise causes moderate degradation (~8% reward, ~14pp more
ball drops) but does NOT prevent curriculum advancement. Both oracle and d435i
hit the **same local optimum** at Stage D — the bottleneck is reward/architecture,
not perception noise.

## 2. The Stage D Plateau Problem

Both oracle and d435i get stuck at Stage D (target=0.20m, σ=0.08m):
- Ball stays near the paddle surface (~3-4cm average height above paddle)
- Apex reward plateaus at ~3/5.0 needed for advancement
- Policy converges to a safe "balance, don't bounce" strategy

This must be addressed BEFORE noise scheduling matters. Possible fixes:
1. **Increase apex reward weight** (currently 25.0) to make juggling more rewarding than survival
2. **Add a velocity bonus** for upward ball velocity (rewarding the "toss" action)
3. **Reduce alive weight** from 1.0 to make survival less dominant
4. **Warm-start from existing checkpoint** with 1000+ iterations to let the policy explore more
5. **Lower the apex threshold** from 5.0 to 3.5 to let curriculum advance (the policy may learn juggling at harder stages where survival alone isn't enough)

These are OUTSIDE the noise curriculum scope but are prerequisites for meaningful
noise-robustness work. Without breaking through Stage D, noise scheduling is
premature.

## 3. Noise Scheduling Design

Once the Stage D plateau is resolved, noise scheduling proceeds in three phases:

### Phase 1: Oracle Warmup (Stages A-C)
- **Noise mode**: oracle (GT ball state)
- **Rationale**: Early stages need clean signal to learn basic ball manipulation.
  Adding noise here would slow learning without benefit — the policy needs to
  discover that bouncing the ball is possible before worrying about noise.

### Phase 2: Gradual Noise Introduction (Stages D-F)
- **Noise mode**: d435i with scaled noise intensity
- **Implementation**: Add a `noise_scale` multiplier to `D435iNoiseParams`:
  - Stage D: noise_scale = 0.25 (25% of full d435i noise)
  - Stage E: noise_scale = 0.50
  - Stage F: noise_scale = 0.75
- **Scale affects**:
  - `sigma_xy_base *= noise_scale`
  - `sigma_z_base *= noise_scale`
  - `dropout_prob *= noise_scale`
  - `latency_steps` stays at 0 until Phase 3
- **Advancement criteria**: Same as current (timeout ≥ 75% AND apex ≥ 5.0)
- **Rationale**: Gradual noise avoids catastrophic forgetting when transitioning
  from oracle to noisy observations. The policy learns to be robust incrementally.

### Phase 3: Full Noise + EKF (Stages G+)
- **Noise mode**: ekf (full D435i noise → EKF filtered state)
- **noise_scale = 1.0** (full d435i parameters)
- **latency_steps = 1** (one-step observation delay, realistic for 30Hz camera)
- **Rationale**: By Stage G the policy has solid ball manipulation skills and can
  handle the full perception pipeline. The EKF provides filtered estimates that
  should be less noisy than raw d435i.

### Combined Curriculum Interaction

The noise curriculum runs **orthogonally** to the juggling curriculum — they share
the same stage counter but control different parameters:

```
Stage  Target(m)  σ(m)    Noise Mode      noise_scale  latency
──────────────────────────────────────────────────────────────
A      0.05       0.025   oracle          0.0          0
B      0.10       0.040   oracle          0.0          0
C      0.15       0.060   oracle          0.0          0
D      0.20       0.080   d435i           0.25         0
E      0.25       0.100   d435i           0.50         0
F      0.30       0.120   d435i           0.75         0
G      0.36       0.144   ekf             1.0          1
H      0.42       0.168   ekf             1.0          1
I+     [range]    varies  ekf             1.0          1
```

### Alternative: Noise-First Curriculum

If the Stage D plateau cannot be broken, an alternative is to train noise
robustness at the CURRENT performance level (Stage D, target=0.20m):

1. Train oracle to convergence at Stage D (done: iter_003)
2. Fine-tune with d435i noise_scale=0.25, same Stage D params
3. Increase noise_scale to 0.50, 0.75, 1.0
4. Switch to ekf mode
5. Only then try to advance past Stage D with full noise

This approach sacrifices curriculum progression speed for noise robustness but
guarantees the policy is noise-robust at every skill level it achieves.

## 4. Implementation Plan

### 4.1 Modify `_BJ_STAGES` in `train_juggle_hier.py`
Add a 6th column `noise_config` to each stage tuple:
```python
_BJ_STAGES = [
    # tgt_min  tgt_max  σ_ratio  xy_std  vel_xy_std  noise_cfg
    (0.05,     0.05,    2.0,     0.020,  0.00,       ("oracle", 0.0)),   # A
    (0.10,     0.10,    2.5,     0.022,  0.00,       ("oracle", 0.0)),   # B
    ...
    (0.20,     0.20,    2.5,     0.028,  0.00,       ("d435i",  0.25)),  # D
]
```

### 4.2 Modify `_bj_set_params()` to update noise config
The observation terms need their `BallObsNoiseCfg` updated at each stage
transition. This requires accessing the obs manager's term configs.

### 4.3 Add `noise_scale` parameter to `D435iNoiseParams`
The perception agent's `ball_obs_spec.py` already supports `sigma_xy_base` etc.
as configurable parameters. Adding a `noise_scale` multiplier is straightforward.

### 4.4 Eval infrastructure (`compare_pi1.py`)
Before deploying the noise curriculum, need systematic evaluation:
- Run a checkpoint against a fixed eval protocol (N episodes, deterministic)
- Compare oracle vs d435i vs ekf at each checkpoint
- Track metrics: survival rate, apex height achieved, ball drops

## 5. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Stage D plateau blocks all progress | High | Fix plateau first (reward tuning, longer training) |
| Noise introduction causes catastrophic forgetting | Medium | Gradual scaling, blend transitions |
| EKF mode not yet implemented in perception | Medium | Start with d435i-only curriculum; wait for perception |
| noise_scale parameter changes obs distribution discontinuously | Low | Blend noise_scale over _BJ_TRANSITION iterations |

## 6. Immediate Next Steps

1. **Break Stage D plateau** — try warm-starting from iter_003 checkpoint for 1000 more iters
2. **Build compare_pi1.py** — eval infrastructure for systematic comparison
3. **Implement noise_scale in ball_obs_spec.py** — coordinate with perception agent
4. **Run noise-first curriculum** at Stage D if plateau persists
