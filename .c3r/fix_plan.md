# fix_plan.md — experiment queue for policy
#
# Active tasks and forward-looking work only. Completed tasks archived.

## ACTIVE — Curriculum advancement for juggling

- [ ] iter_015: Lower _BJ_THRESHOLD from 0.75 to 0.50 (or remove timeout criterion entirely).
      Active juggling has inherently lower timeout% (~63%) vs passive balance (~98%).
      Use apex_reward as primary advancement criterion. Resume from model_4249.pt.
      See git show agent/lit-review:docs/lit_review_curriculum_advancement_criteria.md
- [ ] If curriculum advances: monitor noise_std — if >1.3, reduce entropy_coef
- [ ] After Stage B+: check if ball_below rate improves with curriculum pressure

## NEXT — Noise robustness validation

- [ ] Once juggling advances past Stage A: oracle vs d435i comparison on active juggling
- [ ] Full noise-curriculum run to Stage G with working juggling reward
- [ ] Cross-eval matrix: noise-trained vs oracle-trained, tested with both obs types

## INFRASTRUCTURE

- [ ] Run play.py with model_best.pt or model_4249.pt — capture video to confirm juggling visually

## COMPLETED (archived in RESEARCH_LOG_ARCHIVE.md)
# iters 001-003: oracle baseline established (41D pi2, 12288 envs, Stage D, timeout=98.9%)
# iters 004-005: perception noise integrated, d435i comparison done (~8% degradation)
# iters 006-008: noise curriculum implemented, Stage D plateau broken, noise outperforms oracle
# iters 009-010: root cause chain (sigma_ratio + alive dominance) identified
# iter_011: ball_low_penalty breaks balance (apex 13.7 peak) but policy collapses back
# iter_012: compaction
# iter_013: ball_low=-2.0 + curriculum sustain-during-blend bugfix; -2.0 causes death spiral
# iter_014: ball_release_velocity_reward (+3.0) SUSTAINS JUGGLING — apex 9.7 stable, no collapse
