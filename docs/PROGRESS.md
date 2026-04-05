# QuadruJuggle — Development Progress

Chronological record of the control architecture evolution.
Each version builds on the previous; all code is preserved.

---

## V1 — Mirror Law Controller
**Goal:** Analytic height-customizable juggling, no RL beyond pi2.

**How it works:**
- Computes paddle tilt geometrically from ball state: `n = normalize(v_out - v_in)`
- Target apex height sets required `v_out_z = sqrt(2g * h_target)`
- Pi2 (frozen torso-tracking policy) executes the 6D torso command

**Key files:**
- `scripts/play_mirror_law.py`
- `source/.../torso_tracking/mirror_law_action.py`
- `source/.../ball_juggle_hier/ball_juggle_mirror_env_cfg.py`

**Bug fixed:** h_dot was saturating to max for all apex heights — 2× tracking-lag
boost was applied before clamping, making every target command identical.

**Result:** Works but cannot sustain height. Pi2 h_dot tracking efficiency
(~34–50%) means the ball loses energy each bounce. Energy floor ~0.22 m.
Below that the ball stalls. Height customization fails because open-loop
geometry cannot compensate for real energy losses.

**Demo:**
```bash
# Shows limitation: ball decays to zero at low apex
python scripts/play_mirror_law.py \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --apex_height 0.15 --num_envs 2

# Best achievable with mirror law alone
python scripts/play_mirror_law.py \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --apex_height 0.30 --num_envs 2
```

---

## V2 — Learned RL Pi1 (Hierarchical)
**Goal:** Replace analytic mirror law with an RL policy that can sustain juggling
and respond to variable apex height commands.

**How it works:**
- Pi1 (RL, 46D obs → 6D torso cmd) trained with randomized apex height rewards
- Pi2 frozen — pi1 learns to command pi2 to juggle
- Apex height randomized per episode (0.10–0.60 m) so pi1 learns height awareness

**Key files:**
- `scripts/play_pi1.py`
- `scripts/rsl_rl/train_pi1.py`
- `source/.../ball_juggle_hier/ball_juggle_pi1_env_cfg.py`
- `source/.../ball_juggle_hier/pi1_learned_action.py`
- Best checkpoint: `logs/rsl_rl/go1_ball_juggle_hier/2026-03-22_19-46-52/model_best.pt`

**Bug fixed:** `randomize_apex_height` event was overwriting `--apex_height` flag
on every episode reset, so the play-time override did nothing.

**Result:** Pi1 sustains juggling robustly but **overshoots target by 0.2–0.3 m**
and cannot regulate height precisely. It builds energy aggressively with no
closed-loop height feedback — trained to juggle forever, not to hit a target.

**Demo:**
```bash
python scripts/play_pi1.py \
    --pi1_checkpoint logs/rsl_rl/go1_ball_juggle_hier/2026-03-22_19-46-52/model_best.pt \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --apex_height 0.30 --num_envs 4
```

---

## V3 — Hybrid Controller (Pi1 + Mirror Law)
**Goal:** Combine pi1 (energy building) and mirror law (height regulation) in
one controller. Pi1 runs until the bounce apex is near target, then mirror law
takes over. Falls back to pi1 if apex decays.

**How it works:**
```
Pi1 (energy builder) ──► Mirror Law (height sustainer)
        ▲                          │
        └──── fallback if apex ────┘
              drops below threshold
```
- Per-env boolean mask selects which 6D command goes to pi2 each step
- Switch condition: `|last_bounce_apex - target| < 0.08 m` AND stable for ≥100 steps
- Fallback condition: `last_bounce_apex < 0.50 * target`

**Key files:**
- `scripts/play_hybrid.py`         ← main hybrid controller
- `scripts/sweep_hybrid.sh`        ← parameter sweep (runs headless)

**Changes made:**
- Added `--apex_schedule` for cycling through heights in one session
- Added `--max_steps` + `SWEEP_RESULT` metrics for benchmarking
- Fixed switch from relative threshold to absolute ±0.08 m window (prevents
  premature handoff when pi1 overshoots)

**Result:** Effective apex range **0.22–0.38 m**. Ball oscillates ±0.08 m around
target during mirror law phase rather than converging precisely. Mean episode
reward ~63 at 0.30 m target. Episodes consistently reach max length (1500 steps).

| Apex target | Behavior |
|---|---|
| < 0.22 m | Ball stalls — pi2 energy floor |
| 0.22–0.25 m | Marginal, decays frequently |
| 0.25–0.35 m | Best range, stable cycling |
| 0.35–0.38 m | Riskier, pi1 overshoots more |

**Demo:**
```bash
# Best stable config
python scripts/play_hybrid.py \
    --pi1_checkpoint logs/rsl_rl/go1_ball_juggle_hier/2026-03-22_19-46-52/model_best.pt \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --apex_height 0.30 --switch_threshold 0.80 --fallback_threshold 0.50 --num_envs 4

# Height contrast: low
python scripts/play_hybrid.py \
    --pi1_checkpoint logs/rsl_rl/go1_ball_juggle_hier/2026-03-22_19-46-52/model_best.pt \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --apex_height 0.25 --switch_threshold 0.80 --fallback_threshold 0.50 --num_envs 4

# Height contrast: high
python scripts/play_hybrid.py \
    --pi1_checkpoint logs/rsl_rl/go1_ball_juggle_hier/2026-03-22_19-46-52/model_best.pt \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --apex_height 0.38 --switch_threshold 0.80 --fallback_threshold 0.50 --num_envs 4
```

**Limitation:** Pi1 was not designed to work with mirror law — it overshoots the
target and the hybrid patches this by rebuilding repeatedly. Height oscillates
rather than converging. Open-loop mirror law cannot sense apex error.

---

## V4 — Launcher Pi1 (In Progress)
**Goal:** Redesign pi1 as a **precision launcher** purpose-built to hand off to
mirror law. Pi1's only job: get the ball to exactly `target ± 0.05 m` in as
few bounces as possible, then stop. Mirror law sustains indefinitely.

**Key design changes vs V3 pi1:**

| | V3 Pi1 | V4 Launcher Pi1 |
|---|---|---|
| Job | Juggle forever | Reach target apex, then hand off |
| Apex reward std | 0.10 m (loose) | 0.03 m (tight) |
| Overshoot | No penalty | Strong penalty if apex > target + 0.05 m |
| Success condition | Survive 1500 steps | Clean bounce in [target-0.05, target+0.05] |
| Episode end | Timeout | Terminal reward on successful handoff |

**Key files (to be created):**
- `source/.../ball_juggle_hier/ball_juggle_launcher_env_cfg.py`
- `scripts/rsl_rl/train_launcher.py`
- `scripts/play_launcher_hybrid.py`

**Status:** Design phase — see reward config sketch in env cfg file.
