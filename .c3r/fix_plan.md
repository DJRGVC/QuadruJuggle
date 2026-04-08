# fix_plan.md — experiment queue for policy
#
# Active tasks and forward-looking work only. Completed tasks archived.

## ACTIVE — Break balance-not-bounce local optimum

- [ ] iter_014: Add ball_release_velocity_reward (+3.0 weight) to reward upward ball velocity at
      paddle separation. Combined with ball_low=-1.0, this should sustain juggling through curriculum
      steps without the death spiral that -2.0 caused. Warm-start from iter_013b model_best.pt.
      See git show agent/lit-review:docs/lit_review_passive_optimum_anti_balance.md
- [ ] If release_vel insufficient: gate apex_height_reward on is_airborne (PBRS Φ=0 at contact)
- [ ] If still stuck: try ball_airborne_fraction reward (fraction of timesteps ball is above threshold)

## NEXT — Noise robustness validation

- [ ] Once juggling works: re-run oracle vs d435i comparison at Stage F+ to measure noise degradation on active juggling (not just passive balance)
- [ ] Full noise-curriculum run to Stage G with working juggling reward
- [ ] Evaluate noise-trained policy with oracle obs and vice versa (cross-eval matrix)

## INFRASTRUCTURE

- [ ] Run play.py with peak model from iter_013b (model_best.pt) — capture video to confirm juggling visually

## COMPLETED (archived in RESEARCH_LOG_ARCHIVE.md)
# iters 001-003: oracle baseline established (41D pi2, 12288 envs, Stage D, timeout=98.9%)
# iters 004-005: perception noise integrated, d435i comparison done (~8% degradation)
# iters 006-008: noise curriculum implemented, Stage D plateau broken, noise outperforms oracle
# iters 009-010: root cause chain (sigma_ratio + alive dominance) identified
# iter_011: ball_low_penalty breaks balance (apex 13.7 peak) but policy collapses back
# iter_012: compaction
# iter_013: ball_low=-2.0 + curriculum sustain-during-blend bugfix; -2.0 causes death spiral
