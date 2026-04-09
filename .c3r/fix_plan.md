# fix_plan.md — experiment queue for policy
#
# Active tasks and forward-looking work only. Completed tasks archived.

## ACTIVE — Fix noise mode + oracle vs d435i comparison

- [x] Redesign curriculum: 16 stages → 6 stages (A-F), max target 0.50m.
- [x] Run fresh oracle training with 6-stage curriculum (iters 22-23, all stages reached).
- [x] BUG FIX: iter 22-23 ran with mode="oracle" due to missing --noise-mode d435i flag.
      Curriculum noise_scale changes were no-ops. Added warning to train script.
- [x] **Run fresh training WITH --noise-mode d435i** from scratch (Stage A).
      DONE: reached Stage E (5/6), apex=1.50, outperforms oracle at Stage E.
      Checkpoint: logs/.../2026-04-08_21-16-05/model_best.pt
- [x] Continue d435i training to Stage F — DONE (log dir 2026-04-08_22-51-56, step 3049)
- [x] D435i vs oracle comparison at Stage F — DONE: d435i +59% apex, +34% noise_std
- [x] Cross-eval: noise-trained checkpoint under oracle obs vs d435i obs — DONE iter 28
      **iter 28 numbers were WRONG** — eval had partial-episode bug. Fixed iter 29.
- [x] Fix eval_juggle_hier.py partial-episode bug — iter 29
      Bug: after flush, step_counts=0 but episode_length_buf mid-episode → short ep_len + false timeouts.
      Fix: track `fresh_env` flag, only record episodes that started after a collection-loop reset.
- [x] Re-run cross-eval with fixed eval — CORRECTED RESULTS (iter 29):
      Oracle→Oracle: 100% timeout, 1500 steps everywhere (perfect stability)
      Oracle→D435i:  60-94% timeout, 1100-1450 mean_len (moderate noise degradation)
      D435i→D435i:   0-73% timeout (0% at easy targets, 73% at hard), 300-1219 mean_len
      D435i→Oracle:  0-83% timeout, 442-1345 mean_len (better than d435i→d435i)
      KEY: d435i model overshoots easy targets, performs reasonably at hard ones
- [ ] Capture play.py video of d435i checkpoint for Quarto (Daniel requested videos)
- [x] Diagnose d435i target overshoot at easy targets (0.10-0.20m)
      ROOT CAUSE: d435i model only trained under noise at target=0.50 (Stages E-F).
      Never saw targets 0.10-0.40 with noise. Apex reward HIGHER than oracle at all targets
      (10.33 vs 7.79 at 0.10m) — pure energy modulation failure.
- [x] Stage G continuation training — completed (early-stopped at 1531 stage iters)
- [x] Re-eval Stage G checkpoint — model_early_stop.pt: 80% timeout at 0.10-0.30m, 48-63% at 0.40-0.50m
- [x] **BUG FIX: ES metric used total episode return** — non-monotonic for multi-target
      training (easy targets → short episodes → low total return). Fixed to per-step reward.
      Updated _ES_MIN_DELTA from 0.5 → 0.002.
- [ ] **NEXT: Retrain Stage G with fixed ES metric** — resume from Stage F checkpoint
      (model_best.pt from 2026-04-08_22-51-56). Should converge further now that ES
      won't kill training prematurely. Target: improve 0.40-0.50m while maintaining 0.10-0.30m.
- [ ] Capture play.py video of d435i checkpoint for Quarto (Daniel requested videos)

## NEXT — Validate juggling behavior

- [ ] Validate: does policy actively bounce ball at 0.50m target? (apex reward + visual)

## USER COMMAND INTERFACE (Daniel request 2026-04-08)

- [x] Implement WASD velocity + P/L height controls in play_teleop.py
- [ ] Test teleop with Daniel (needs visual verification)
- [ ] Research: how to compose user height commands with curriculum target during training

## COMPLETED
# iters 001-003: oracle baseline (41D pi2, 12288 envs, Stage D, timeout=98.9%)
# iters 004-005: perception noise integrated, d435i comparison (~8% degradation)
# iters 006-008: noise curriculum, Stage D plateau broken, noise outperforms oracle
# iters 009-014: root cause chain → ball_release_velocity_reward SUSTAINS JUGGLING
# iters 015-016: curriculum threshold 0.75→0.30, training plateau at apex≈10.7
# iter_017: compaction
# iter_018: plateau diagnosis, active GPU run analysis
# iter_019-020: curriculum redesign 16→6 stages (Daniel approved)
# iter_021: teleop interface
# iter_022-023: 6-stage oracle training — all stages reached (but accidentally oracle-only!)
