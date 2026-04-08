# fix_plan.md — experiment queue for policy
#
# Replace this preamble with 3-5 concrete starting tasks (one per line,
# as a markdown bullet). Lines starting with # are kept as comments.
# Agents read the TOP of this file at the start of every iteration.
#
# Example format:
#   - [ ] Task one — one full sentence, no line breaks
#   - [ ] Task two
#
# Save and exit your editor when done. Empty file = agent picks its own direction.

- [x] Run the current pi1+pi2 ball juggle baseline with oracle ball observations for 500 iterations to capture a clean reference metric (mean episode length, paddle-catch rate, final target height reached); log to RESEARCH_LOG.md with the exact command and tbdump output
  # DONE iter_001: Stage D reached, mean_len=1470, timeout=92.4%, apex=3.11, checkpoint logs/rsl_rl/go1_ball_juggle_hier/2026-04-07_19-11-34/model_best.pt
- [x] Read agent/perception:source/go1_ball_balance/perception/ball_obs_spec.py via git show as soon as perception publishes the stub; integrate the function as an observation term in the pi1 env, falling back to oracle behavior while the stub is zero-noise
  # DONE iter_004: Integrated ball_obs_spec.py into pi1 env config. --noise-mode oracle|d435i CLI flag added. Both modes smoke-tested OK.
- [x] Run full d435i noise training (500 iters, 12288 envs) and compare to oracle baseline (iter_003: timeout=98.9%, apex=2.92, mean_len=1500)
  # DONE iter_005: d435i causes ~8% reward degradation, same Stage D plateau. D435i+wandb rerun in progress.
- [x] Set up a degradation-comparison infrastructure: a script scripts/rsl_rl/compare_pi1.py that runs two checkpoints against each other on a fixed eval protocol and outputs relative metrics to stdout; commit to your branch
  # DONE iter_007a: compare_pi1.py created + --noise-mode added to eval_juggle_hier.py
- [x] Review the existing sigma curriculum in train_juggle_hier.py stages A-G from memory and propose how noise scheduling would interact with it; write the design to docs/noise_curriculum_plan.md without implementing yet
  # DONE iter_006: wrote docs/noise_curriculum_plan.md — 3-phase approach (oracle→scaled d435i→ekf). Stage D plateau is the blocker.
- [ ] BLOCKER: Break Stage D apex plateau — warm-start from iter_003 oracle checkpoint for 1000+ more iters to see if longer training helps
- [ ] After Stage D fix: implement noise_scale parameter in ball_obs_spec.py and integrate into _BJ_STAGES curriculum
- [ ] Add wandb logging to train_juggle_hier.py (entity: d-grant-uc-berkeley) + video recording for play visualization
- [x] BUG: action_term.py builds 41D pi2 obs (missing last_action=12D) but pi2 checkpoints from 2026-03-12_14-31-45 onward have 53D input; fix action_term.py to include last joint targets then retrain pi2 and pi1 from scratch
  # DONE iter_002: action_term.py now auto-detects pi2 input dim (41 or 53) and conditionally appends last_action. 53D pi2 baseline ran but underperformed 41D pi2 (Stage C vs Stage D at 500 iters).
  # iter_003: Confirmed — 41D pi2 + 12288 envs gives best oracle baseline (mean_len=1500 maxed, timeout=98.9%). Oracle checkpoint: logs/rsl_rl/go1_ball_juggle_hier/2026-04-07_20-18-34/model_best.pt

