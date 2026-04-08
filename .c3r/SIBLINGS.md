# SIBLINGS — auto-regenerated at the start of each iteration

This file is your snapshot of what every OTHER agent in this project
has been doing. Use it at the top of every iteration (after reading
INBOX.md) to stay coordinated.

**To actually read a file from a sibling's branch** (without
checking it out — they're on separate branches to avoid conflicts):
```
git show agent/<sibling-name>:path/to/file
```

**To see what a sibling has changed since you last looked:**
```
git log agent/<sibling-name> --since='1 hour ago' --name-status
git diff HEAD agent/<sibling-name> -- path/
```

**To push a handoff file to siblings:** commit it on your own branch
and reference it in your Discord thread or in your next log entry.
Siblings will see it in their next SIBLINGS.md refresh.

---

## policy
- **role**: generic
- **focus**: retrain pi1 with noise-injected ball observations from the perception pipeline, validate degradation versus oracle baseline and restore performance via curriculum and noise scheduling.
- **status**: running · iter #4 · ctx 0%
- **last iter**: 1h ago

### Recent commits on `agent/policy`
```
9ca60cc sync perception pipeline from perception branch: EKF mode + noise_scale curriculum
f2feadd update RESEARCH_LOG iter_008: corrected to 1199 iters (apex_rew=1.87), queue iter_009
69c9cd1 process inbox: c3r spawn note from perception; update RESEARCH_LOG iter_008 state
c67e0d3 iter_008: noise-curriculum training — d435i ramp 0→75% reaches Stage F, outperforms oracle
ac2fa7f update fix_plan: mark noise_scale done, add noise-curriculum training task
```
### Files modified on `agent/policy` (relative to `c3r/QuadruJuggle`)
```
.c3r/INBOX.md
.c3r/INBOX_ARCHIVE.md
.c3r/PROMPT.md
.c3r/RESEARCH_LOG.md
.c3r/SIBLINGS.md
.c3r/agent.conf
.c3r/env.sh
.c3r/fix_plan.md
.claude/settings.json
.gitignore
docs/noise_curriculum_plan.md
scripts/rsl_rl/compare_pi1.py
scripts/rsl_rl/eval_juggle_hier.py
scripts/rsl_rl/train_juggle_hier.py
source/go1_ball_balance/go1_ball_balance/perception/__init__.py
source/go1_ball_balance/go1_ball_balance/perception/ball_ekf.py
source/go1_ball_balance/go1_ball_balance/perception/ball_obs_spec.py
source/go1_ball_balance/go1_ball_balance/perception/noise_model.py
source/go1_ball_balance/go1_ball_balance/tasks/ball_juggle_hier/ball_juggle_hier_env_cfg.py
source/go1_ball_balance/go1_ball_balance/tasks/torso_tracking/action_term.py
```
### Read one with:
```
git show agent/policy:.c3r/INBOX.md
git show agent/policy:.c3r/INBOX_ARCHIVE.md
git show agent/policy:.c3r/PROMPT.md
git show agent/policy:.c3r/RESEARCH_LOG.md
git show agent/policy:.c3r/SIBLINGS.md
```

## lit-review
- **role**: generic
- **focus**: survey perception-for-manipulation papers (ETH noise injection, teacher-student distillation, event cameras, learned state estimators) and critique our D435i+EKF approach
- **status**: running · iter #15 · ctx 0%
- **last iter**: 1m ago
- **parent**: perception (this is a sub-agent)

### Recent commits on `agent/lit-review`
```
0118b69 iter_015: contact dynamics DR — PhysX solver validation + combine_mode precedence bug → docs/lit_review_contact_dynamics_dr.md
0a69cbd iter_014: distributional latency injection survey → docs/lit_review_latency_injection.md
47940e0 iter_013: learned state estimators survey → docs/lit_review_learned_state_estimators.md
b74c1cb iter_012: event camera latency survey → docs/lit_review_event_cameras.md
96602fd iter_011: teacher-student vs noise injection survey → docs/lit_review_teacher_student_vs_noise_injection.md
```
### Files modified on `agent/lit-review` (relative to `c3r/QuadruJuggle`)
```
.c3r/INBOX.md
.c3r/INBOX_ARCHIVE.md
.c3r/PROMPT.md
.c3r/RESEARCH_LOG.md
.c3r/SIBLINGS.md
.c3r/agent.conf
.c3r/env.sh
.c3r/fix_plan.md
.claude/settings.json
docs/lit_review_active_throwing_rewards.md
docs/lit_review_actuator_dynamics.md
docs/lit_review_ball_spin.md
docs/lit_review_bounce_dynamics.md
docs/lit_review_contact_dynamics_dr.md
docs/lit_review_ekf_lag_vs_raw_noise.md
docs/lit_review_ekf_tuning.md
docs/lit_review_event_cameras.md
docs/lit_review_integrated_sim2real.md
docs/lit_review_latency_injection.md
docs/lit_review_learned_state_estimators.md
docs/lit_review_noise_curriculum.md
docs/lit_review_noise_outperforms_oracle.md
docs/lit_review_perception.md
docs/lit_review_teacher_student_vs_noise_injection.md
```
### Read one with:
```
git show agent/lit-review:.c3r/INBOX.md
git show agent/lit-review:.c3r/INBOX_ARCHIVE.md
git show agent/lit-review:.c3r/PROMPT.md
git show agent/lit-review:.c3r/RESEARCH_LOG.md
git show agent/lit-review:.c3r/SIBLINGS.md
```

