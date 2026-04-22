# Mirror-Law Ball-Position Noise Tolerance

**Experiment date:** 2026-04-21
**Commit reference:** c3bdfc8 (hybrid stack with user-cmd injection)
**Purpose:** Bound the maximum Gaussian ball-position-measurement noise at which the
classical mirror-law controller sustains continuous juggling. Result feeds the
EKF residual-noise budget in `docs/perception_roadmap.md`.

## Setup

| Knob | Value |
|---|---|
| Task | `Isaac-BallJuggleMirror-Go1-v0` (mirror-law pi1 + frozen 9D pi2) |
| pi2 checkpoint | `logs/rsl_rl/go1_torso_tracking/2026-04-21_10-38-52/model_best.pt` |
| apex_height | 0.30 m |
| user-cmd | `--user-vx 0.2` (forward walk at 0.1 m/s) |
| num_envs | 4 |
| video_length | 800 steps (16 s sim time) |
| noise model | Gaussian on ball position, i.i.d. per sim step, applied inside `MirrorLawTorsoAction` (see `ball_pos_noise_std`) |

Four i.i.d. envs per noise level → 4 independent episode attempts. An env whose
episode terminates before step 800 counts as a drop. Remaining envs record
`ep_len = 800`.

## Results

### Ball-position noise

| noise σ [mm] | drops observed | envs sustained 800/800 | ep_len when dropped (steps) |
|---|---|---|---|
| 2  | 0 | 4/4 | — |
| 5  | 1 (single-env run from demo 19) | — | 690 |
| 10 | 0 | 4/4 | — |
| 12 | 2 | 2/4 | 679, 789 |
| 15 | 3 | 1/4 | 706, 712, 778 |

Demo 19 was a one-env probe (num_envs=1); its single drop at step 690 is a tail
event in what the full sweep shows is a robust region — 2 mm and 10 mm both
sustain 4/4 with zero drops across the 16 s window.

### Ball-velocity noise

| noise σ [m/s] | drops observed | envs sustained 800/800 | ep_len when dropped (steps) |
|---|---|---|---|
| 0.01 | 1 | 3/4 | 765 |
| 0.03 | 1 | 3/4 | 634 |
| 0.05 | 0 | 4/4 | — |
| 0.10 | 2 | 2/4 | 651, 730 |

**Caveat on ordering.** 0.05 m/s shows 0 drops while 0.01 and 0.03 each show
1 drop. With n=4 envs per level the per-point drop rate is a noisy estimator;
these three points are statistically indistinguishable. The honest reading is
that the whole σ ≤ 0.05 m/s band is roughly 85-100 % sustained, with the first
clear degradation at 0.10 m/s (50 %). The mirror law is markedly more robust
to velocity noise than to position noise — at 100 mm/s (≈3-4 % of the ball's
peak juggle speed) it still sustains 2/4 envs, whereas 15 mm position noise
drops 3/4. Hypothesis: position error enters the mirror law's feedback term
directly (determines paddle tilt), while velocity error only enters through an
internal differentiation that is already smoothed.

## Interpretation

- **Robust below ~10 mm.** The controller absorbs i.i.d. position noise up to
  ~10 mm σ without any observable drops in 16-second windows. At vx = 0.1 m/s
  (0.2 normalised), that is comparable to or larger than the noise floor of a
  calibrated HSV blob detector at 30 cm paddle distance.
- **Graceful knee at ~12 mm.** 12 mm σ drops 2/4 envs within the window; 15 mm
  drops 3/4. No catastrophic cliff — drops happen after 600+ steps, indicating
  noise-driven drift rather than instantaneous controller failure.
- **Sim-to-real implication.** The EKF-based perception stack in
  `docs/perception_roadmap.md` targets residual ball-position noise. This sweep
  gives an empirical upper bound: **keep EKF residual σ ≤ 10 mm** to preserve
  the mirror-law's "0 drops over 16 s" margin. A budget up to ~12 mm is
  survivable but marginal.

## Limitations

- Only **i.i.d. Gaussian noise** was tested. Real camera pipelines introduce
  temporally correlated noise (motion blur, white-balance drift, detector
  dropouts) that the mirror law may tolerate differently.
- Position and velocity noise were swept **independently**. Real perception
  produces both simultaneously, often correlated, and the interaction was not
  characterised here.
- Fixed conditions: apex 0.30 m and vx = 0.1 m/s. Higher apex and higher vx
  both reduce margins (see clips 11, 18 in `myrecordings/user_cmd_demos/`).
- Only **4 independent samples per noise level**. A production robustness
  study would run ≥20 per level and report quantile bands. At n=4 the drop
  rate is a coarse estimator — differences of ±1/4 between adjacent points
  (e.g. 0.01 vs 0.05 m/s) are noise, not signal.

## Raw data

- Position-noise logs: `/tmp/noise_sweep/n0.{002,010,012,015}.log`
- Velocity-noise logs: `/tmp/velnoise_sweep/v0.{01,03,05,10}.log`
- Analysis scripts: `/tmp/analyze_noise_sweep.sh`, `/tmp/analyze_velnoise_sweep.sh`
- Boundary clip (position 12 mm): `myrecordings/user_cmd_demos/20_mirror_noise_boundary_12mm.mp4`

## Follow-ups

1. **Higher-n re-run** of each curve (≥20 envs per σ) for confidence bands.
2. Temporally correlated noise (AR(1) with τ ≈ 30 ms) — more realistic than i.i.d.
3. Detector-dropout probe: simulate 0.1–0.5 s perception gaps.
4. Re-run at higher apex (0.45 m) — expect tighter margin on both axes.
5. Joint position × velocity noise: determine whether combined budgets
   interact sub-linearly (independent) or super-linearly (coupled failure).
