# fix_plan.md — experiment queue for policy
#
# Active tasks and forward-looking work only. Completed tasks archived.

## ACTIVE — Simplified 6-stage curriculum (0.50m target)

- [x] Redesign curriculum: 16 stages → 6 stages (A-F), max target 0.50m.
      Literature-aligned (Rudin 4, Zhuang 4, ROGER 4). Daniel approved.
- [x] Run fresh training with new 6-stage curriculum from scratch (Stage A).
      DONE: iter 22 trained A→D (stuck). iter 23 fixed σ_ratio 3.5→2.5, ALL 6 STAGES REACHED.
- [x] If Stage A-C reached in first run: continue with resume to push through D-F
      DONE: D→E→F in 1500 iters (78 min). Checkpoint: 2026-04-08_19-19-41/model_best.pt
- [ ] Validate: does policy actively bounce ball at 0.50m target? (apex reward + visual)
- [ ] Oracle vs d435i comparison once Stage E/F reached

## NEXT — Noise robustness validation

- [ ] Cross-eval: noise-trained vs oracle-trained, tested with both obs types
- [ ] Full run to Stage F with d435i noise — final result

## USER COMMAND INTERFACE (Daniel request 2026-04-08)

- [x] Implement WASD velocity + P/L height controls in play_teleop.py
- [ ] Test teleop with Daniel (needs visual verification)
- [ ] Research: how to compose user height commands with curriculum target during training

## INFRASTRUCTURE

- [ ] Capture play.py video to confirm juggling visually

## COMPLETED
# iters 001-003: oracle baseline (41D pi2, 12288 envs, Stage D, timeout=98.9%)
# iters 004-005: perception noise integrated, d435i comparison (~8% degradation)
# iters 006-008: noise curriculum, Stage D plateau broken, noise outperforms oracle
# iters 009-014: root cause chain → ball_release_velocity_reward SUSTAINS JUGGLING
# iters 015-016: curriculum threshold 0.75→0.30, training plateau at apex≈10.7
# iter_017: compaction
# iter_018: plateau diagnosis, active GPU run analysis
# iter_019: curriculum redesign 16→6 stages, 0.50m target (Daniel approved)
