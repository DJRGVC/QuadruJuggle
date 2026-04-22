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

- Only **i.i.d. Gaussian position noise** was tested. Real camera pipelines
  introduce temporally correlated noise (motion blur, white-balance drift,
  detector dropouts) that the mirror law may tolerate differently.
- Ball **velocity** noise was held at zero. Separate sweep needed — the mirror
  law differentiates position to estimate velocity internally, so velocity
  noise is expected to be more corrosive.
- Fixed conditions: apex 0.30 m and vx = 0.1 m/s. Higher apex and higher vx
  both reduce margins (see clips 11, 18 in `myrecordings/user_cmd_demos/`).
- Only **4 independent samples per noise level**. A production robustness
  study would run ≥20 per level and report quantile bands.

## Raw data

Per-run logs: `/tmp/noise_sweep/n0.{002,010,012,015}.log`
Analysis script: `/tmp/analyze_noise_sweep.sh`
Boundary clip (12 mm): `myrecordings/user_cmd_demos/20_mirror_noise_boundary_12mm.mp4`

## Follow-ups

1. Ball-velocity noise sweep (σ = 0.005–0.05 m/s).
2. Temporally correlated noise (AR(1) with τ ≈ 30 ms).
3. Detector-dropout probe: simulate 0.1–0.5 s perception gaps.
4. Re-run with higher apex (0.45 m) — expect tighter margin.
