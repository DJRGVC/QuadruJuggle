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
- **status**: running · iter #8 · ctx 0%
- **last iter**: 39m ago

### Recent commits on `agent/policy`
```
c5f3d4e iter_015: curriculum threshold 0.75→0.30 — CURRICULUM ADVANCES, juggling sustained
f77d50e iter_014: ball_release_velocity_reward SUSTAINS JUGGLING (apex 9.7 stable, 1500 iters)
dc7bb6f iter_014: add ball_release_velocity_reward (+3.0)
eefbf31 iter_013: ball_low -2.0 death spiral; fix curriculum sustain-during-blend bug
8607975 Log iter_013 early results (Stage D, apex=0.35 stable, weight=-2.0 working)
```
### Files modified on `agent/policy` (relative to `c3r/QuadruJuggle`)
```
.c3r/INBOX.md
.c3r/INBOX_ARCHIVE.md
.c3r/PROMPT.md
.c3r/RESEARCH_LOG.md
.c3r/RESEARCH_LOG_ARCHIVE.md
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
source/go1_ball_balance/go1_ball_balance/tasks/ball_juggle/mdp/rewards.py
source/go1_ball_balance/go1_ball_balance/tasks/ball_juggle_hier/ball_juggle_hier_env_cfg.py
source/go1_ball_balance/go1_ball_balance/tasks/torso_tracking/action_term.py
```
### Read one with:
```
git show agent/policy:.c3r/INBOX.md
git show agent/policy:.c3r/INBOX_ARCHIVE.md
git show agent/policy:.c3r/PROMPT.md
git show agent/policy:.c3r/RESEARCH_LOG.md
git show agent/policy:.c3r/RESEARCH_LOG_ARCHIVE.md
```

## lit-review
- **role**: generic
- **focus**: survey perception-for-manipulation papers (ETH noise injection, teacher-student distillation, event cameras, learned state estimators) and critique our D435i+EKF approach
- **status**: stopped · iter #28 · ctx 0%
- **last iter**: 5h ago
- **parent**: perception (this is a sub-agent)

### Recent commits on `agent/lit-review`
```
177405f iter_030: Go1 actuator sim-to-real gap survey — action delay 20ms is #1 gap; PD+DR > ActuatorNet; DR table + DelayedPDActuator rec → docs/lit_review_actuator_dynamics_sim2real.md
8030cca iter_029: reward phasing survey — stage-aware annealing; ball_low_penalty must taper by Stage C; release_vel keep until Stage D → docs/lit_review_reward_phasing.md
eb9d385 iter_028: policy collapse/forgetting survey — degenerate attractor root cause; early_termination -200→-1200; adaptive entropy, L2 anchor → docs/lit_review_policy_collapse_forgetting.md
d11eb83 iter_027: ping-pong CoR survey — COR=0.85 correct at Stage G; bounce_threshold_velocity=0.5→0.10 fix; DR ranges validated → docs/lit_review_ping_pong_cor.md
390f0f8 iter_026: curriculum advancement criteria survey — timeout_rate wrong for juggling; replace 75%→30% floor + apex_reward threshold → docs/lit_review_curriculum_advancement_criteria.md
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
docs/hardware_deployment_checklist.md
docs/lit_review_active_throwing_rewards.md
docs/lit_review_actuator_dynamics.md
docs/lit_review_actuator_dynamics_sim2real.md
docs/lit_review_ball_spin.md
docs/lit_review_bounce_dynamics.md
docs/lit_review_contact_dynamics_dr.md
docs/lit_review_curriculum_advancement_criteria.md
docs/lit_review_d435i_ros2_integration.md
docs/lit_review_ekf_dropout_fallback.md
docs/lit_review_ekf_lag_vs_raw_noise.md
docs/lit_review_ekf_tuning.md
docs/lit_review_event_cameras.md
docs/lit_review_imu_aided_ekf_platform_motion.md
docs/lit_review_integrated_sim2real.md
docs/lit_review_jetson_orin_profiling.md
docs/lit_review_latency_injection.md
docs/lit_review_learned_state_estimators.md
docs/lit_review_noise_curriculum.md
docs/lit_review_noise_outperforms_oracle.md
docs/lit_review_passive_optimum_anti_balance.md
... and 9 more
```
### Read one with:
```
git show agent/lit-review:.c3r/INBOX.md
git show agent/lit-review:.c3r/INBOX_ARCHIVE.md
git show agent/lit-review:.c3r/PROMPT.md
git show agent/lit-review:.c3r/RESEARCH_LOG.md
git show agent/lit-review:.c3r/SIBLINGS.md
```

## vel-cmd-survey
- **role**: generic
- **focus**: Survey 2023-2026 papers on user-defined velocity command input for legged robots doing manipulation while walking. Our Go1 has hierarchical pi2 torso tracker (8D: h,hdot,roll,pitch,omega_r,omega_p,vx,vy) and pi1 ball planner. Propose 3 methods for adding user velocity commands. Track references. Report to parent perception every iter via INBOX. Complete in 5 iters MAX then stop.
- **status**: stopped · iter #5 · ctx 0%
- **last iter**: 3h ago
- **parent**: perception (this is a sub-agent)

### Recent commits on `agent/vel-cmd-survey`
```
8ebc5a2 iter_005: final handoff — survey complete, M1 implemented by perception, M2 recommended next
f7c845b iter_004: Final proposal — vel_cmd_proposal_final.md (exec summary, comparison, risks, phased impl order)
516a817 iter_003: Methods 2 (Residual) and 3 (Multi-Mode) specs — ResidualMixer + ModeConditionedInput + reward changes
38522c3 iter_002: Method 1 (Direct Override) implementation spec — UserVelocityInput + CommandMixer
f78f121 iter_001: velocity command survey — 13 papers, 3 methods proposed
```
### Files modified on `agent/vel-cmd-survey` (relative to `c3r/QuadruJuggle`)
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
docs/lit_review_vel_cmd_loco_manipulation.md
docs/vel_cmd_method1_spec.md
docs/vel_cmd_method2_3_spec.md
docs/vel_cmd_proposal_final.md
```
### Read one with:
```
git show agent/vel-cmd-survey:.c3r/INBOX.md
git show agent/vel-cmd-survey:.c3r/INBOX_ARCHIVE.md
git show agent/vel-cmd-survey:.c3r/PROMPT.md
git show agent/vel-cmd-survey:.c3r/RESEARCH_LOG.md
git show agent/vel-cmd-survey:.c3r/SIBLINGS.md
```

## report-writer
- **role**: generic
- **focus**: Create a nicely formatted project report (HTML or Markdown with embedded images/graphs) covering the QuadruJuggle project: perception pipeline, policy training, architecture, and results. Max 10 iterations.
- **status**: running · iter #5 · ctx 0%
- **last iter**: 2m ago
- **parent**: perception (this is a sub-agent)

### Recent commits on `agent/report-writer`
```
0b6df27 iter_005: velocity commands section — Method 1/2/3 comparison table + architecture diagram
556195e iter_004: lessons learned section — 6 before/after cards (crouching, entropy, Gaussian, joints, noise, termination)
f547f3c iter_003: scene constants card + ball_release callout + pi2 curriculum table A→H
25c3245 iter_002: dual-line training chart (apex+timeout%) + NIS validation table from GPU iter_049
d8ed014 iter_001: initial HTML project report — architecture, training, perception, results, sim2real
```
### Files modified on `agent/report-writer` (relative to `c3r/QuadruJuggle`)
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
report.html
```
### Read one with:
```
git show agent/report-writer:.c3r/INBOX.md
git show agent/report-writer:.c3r/INBOX_ARCHIVE.md
git show agent/report-writer:.c3r/PROMPT.md
git show agent/report-writer:.c3r/RESEARCH_LOG.md
git show agent/report-writer:.c3r/SIBLINGS.md
```

