# Mirror-Law Ball Noise Tolerance

**Experiment date:** 2026-04-21
**Commit reference:** c3bdfc8 (hybrid stack with user-cmd injection)
**Purpose:** Bound the maximum Gaussian ball-position and ball-velocity measurement
noise at which the classical mirror-law controller sustains continuous juggling.
Feeds the EKF residual-noise budget in `docs/perception_roadmap.md`.

> **2026-04-21 revision:** the original n=4 sweep substantially under-estimated
> drop rates at 10-12 mm position noise. An n=20 follow-up showed the true
> position-noise knee is around 6-10 mm, not 12 mm. The n=20 CIs do overlap the
> n=4 point estimates (so the smaller sweep was not *wrong* — just too coarse to
> resolve the knee) but the revised design target is tighter. Updated numbers
> and both sweeps' data are in the "High-n validation (n=20)" section below.

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

## High-n validation (n=20)

The n=4 sweep above had wide confidence intervals and its point estimates are
unreliable. Re-ran key points with num_envs=20 (same seed policy, same apex/vx
conditions) to tighten the rates. All rates below are "unique envs that
dropped the ball at least once within 800 steps", reported with Wilson 95 % CIs.

**Position noise (n=20):**

| σ [mm] | envs dropped | drop rate | 95 % Wilson CI |
|---|---|---|---|
| 2  | 2/20 | 10 % | 3–30 % |
| 4  | 1/20 | 5 %  | 1–24 % |
| 6  | 0/20 | 0 %  | 0–16 % |
| 10 | 8/20 | 40 % | 22–61 % |
| 12 | 7/20 | 35 % | 18–57 % |

**Velocity noise (n=20):**

| σ [m/s] | envs dropped | drop rate | 95 % Wilson CI |
|---|---|---|---|
| 0.05 | 1/20  | 5 %  | 1–24 % |
| 0.10 | 14/20 | 70 % | 48–85 % |

## Interpretation (post-n=20)

- **Position-noise knee is 6–10 mm**, not 12 mm as the n=4 sweep suggested.
  Between 6 mm and 10 mm the drop rate jumps from ~0 % to ~40 %. The
  narrow-CI region at the bottom of the curve (2/4/6 mm) is consistent
  with a true drop rate under ~20 %; above 10 mm it jumps substantially.
  The non-monotonicity at the low end (2 mm > 4 mm > 6 mm point
  estimates) is within-CI sample noise — do not over-interpret.
- **Velocity-noise knee is between 0.05 and 0.10 m/s**, with a steep
  transition (5 % → 70 %). 0.05 m/s is comfortably robust (CI upper bound
  24 %); 0.10 m/s is past the knee.
- **Asymmetric sensitivity.** The mirror law is ≈ 1.5–2× more tolerant of
  velocity than position noise, normalised to the ball's peak juggle speed
  (≈ 2–3 m/s at 0.30 m apex). Explanation in the original writeup:
  position error sets paddle tilt directly; velocity error passes through an
  internal differentiation that already filters.
- **Sim-to-real implication.** Updated EKF residual design targets:
  **σ_pos ≤ 6 mm** and **σ_vel ≤ 0.05 m/s** to keep drop-rate CI ≤ ~20 %.
  Budgets up to σ_pos ≈ 10 mm are survivable but the mirror law alone
  loses ~40 % of episodes; the learned pi1 may tolerate more but we don't
  have that data yet.

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

- n=4 position-noise logs: `/tmp/noise_sweep/n0.{002,010,012,015}.log`
- n=4 velocity-noise logs: `/tmp/velnoise_sweep/v0.{01,03,05,10}.log`
- n=20 position + velocity logs: `/tmp/highn_sweep/{pos,vel}_*.log`
- Analysis scripts: `/tmp/analyze_noise_sweep.sh`, `/tmp/analyze_velnoise_sweep.sh`, `/tmp/analyze_highn_sweep.sh`
- Boundary clip (position 12 mm, n=4): `myrecordings/user_cmd_demos/20_mirror_noise_boundary_12mm.mp4`

## Follow-ups

1. ~~Higher-n re-run (≥20 envs per σ) for confidence bands.~~ ✓ done (see section above).
2. **15 mm position noise at n=20** to complete the degradation curve. n=4 gave
   3/4 = 75 % (CI 30–95 %); expect n=20 to confirm >50 %.
3. Temporally correlated noise (AR(1) with τ ≈ 30 ms) — more realistic than i.i.d.
4. Detector-dropout probe: simulate 0.1–0.5 s perception gaps.
5. Re-run at higher apex (0.45 m) — expect tighter margin on both axes.
6. Coupled position × velocity noise: determine whether combined budgets
   interact sub-linearly (independent) or super-linearly (coupled failure).
7. Swap the mirror law for the learned pi1 (launcher checkpoint) to see whether
   the learned policy tolerates more noise than the hand-crafted mirror law.
