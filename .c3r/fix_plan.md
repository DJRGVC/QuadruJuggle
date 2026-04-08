# fix_plan.md — experiment queue for policy
#
# Active tasks and forward-looking work only. Completed tasks archived.

## ACTIVE — Break balance-not-bounce local optimum

- [ ] iter_012: Increase ball_low_penalty weight from -1.0 to -2.0 (lit-review says passive earns 0/step at -1.0; needs net -1/step → weight -2.0). Warm-start from iter_011 model_best.pt (peak juggling at iter ~200).
- [ ] If -2.0 insufficient: add ball_release_velocity_reward=+3.0 at paddle-ball separation (DribbleBot + JuggleRL pattern). See git show agent/lit-review:docs/lit_review_passive_optimum_anti_balance.md
- [ ] If still stuck: gate apex_height_reward on is_airborne (PBRS Φ=0 at contact) — architectural fix

## NEXT — Noise robustness validation

- [ ] Once juggling works: re-run oracle vs d435i comparison at Stage F+ to measure noise degradation on active juggling (not just passive balance)
- [ ] Full noise-curriculum run to Stage G with working juggling reward
- [ ] Evaluate noise-trained policy with oracle obs and vice versa (cross-eval matrix)

## INFRASTRUCTURE

- [ ] iter_011 follow-up: compare model_best.pt (peak juggling) vs model_450.pt (passive) via compare_pi1.py
- [ ] Run play.py with model_best.pt from iter_011 peak — capture video to confirm juggling visually

## COMPLETED (archived in RESEARCH_LOG_ARCHIVE.md)
# iters 001-003: oracle baseline established (41D pi2, 12288 envs, Stage D, timeout=98.9%)
# iters 004-005: perception noise integrated, d435i comparison done (~8% degradation)
# iters 006-008: noise curriculum implemented, Stage D plateau broken, noise outperforms oracle
# iters 009-010: root cause chain (sigma_ratio + alive dominance) identified
# iter_011: ball_low_penalty breaks balance (apex 13.7 peak) but policy collapses back
