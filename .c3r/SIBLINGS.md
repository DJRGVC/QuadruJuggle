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

## perception
- **role**: generic
- **focus**: implement an onboard D435i camera to ball detector to 6-DOF EKF pipeline following ETH-style architecture, producing noisy ball observations usable by pi1 in Isaac Lab sim.
- **status**: running · iter #41 · ctx 0%
- **last iter**: 8m ago

### Recent commits on `agent/perception`
```
9314aaa iter_048: gate rejection stats in pipeline diagnostics + NIS diagnostic tool (2/2 new tests, 229/229 total)
aae1548 iter_047: chi-squared NIS gating in BallEKF (19/19 new tests, 227/227 total)
bbbedec iter_046: threaded RealPerceptionPipeline + integration tests (17/17 new tests, 208/208 total)
e9e14c3 iter_045: Hough circle fallback detector + from_yaml (15/15 new tests, 191/191 total)
174f57b iter_044: compaction (summarized iters 029-035)
```
### Files modified on `agent/perception` (relative to `c3r/QuadruJuggle`)
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
docs/hardware_pipeline_architecture.md
docs/lit_review_ekf_lag_vs_raw_noise.md
docs/lit_review_ekf_tuning.md
docs/perception_roadmap.md
docs/sim_to_real_plan.md
scripts/perception/compare_perception_modes.py
scripts/perception/debug_d435i_capture.py
scripts/perception/nis_diagnostic.py
scripts/perception/nis_sweep.py
scripts/perception/test_ballistic_trajectory.py
scripts/perception/test_contact_aware_ekf.py
scripts/perception/test_ekf_integration.py
scripts/perception/test_hough_detector.py
scripts/perception/test_imu_aided_ekf.py
scripts/perception/test_latency_injection.py
scripts/perception/test_mock_pipeline.py
scripts/perception/test_nis_gating.py
scripts/perception/test_pipeline_config.py
scripts/perception/test_real_utils.py
scripts/perception/test_spin_estimation.py
... and 26 more
```
### Read one with:
```
git show agent/perception:.c3r/INBOX.md
git show agent/perception:.c3r/INBOX_ARCHIVE.md
git show agent/perception:.c3r/PROMPT.md
git show agent/perception:.c3r/RESEARCH_LOG.md
git show agent/perception:.c3r/RESEARCH_LOG_ARCHIVE.md
```

## lit-review
- **role**: generic
- **focus**: survey perception-for-manipulation papers (ETH noise injection, teacher-student distillation, event cameras, learned state estimators) and critique our D435i+EKF approach
- **status**: stopped · iter #28 · ctx 0%
- **last iter**: 3h ago
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
- **last iter**: 58m ago
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

