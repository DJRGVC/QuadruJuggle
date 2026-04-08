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
- **status**: running · iter #5 · ctx 0%
- **last iter**: 5m ago

### Recent commits on `agent/policy`
```
74ef87b iter_012: ball_low weight -1.0→-2.0, revert dynamic threshold
4b48378 iter_012: compaction (summarized iters 001-008)
3b9d536 Dynamic ball_low threshold: 40% of target height per stage
85acdcd iter_011 early: ball_low_penalty breaks balance local optimum — 158x apex improvement
0314ec4 log iter_010 results; update fix_plan for iter_011
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
- **status**: running · iter #25 · ctx 0%
- **last iter**: 1m ago
- **parent**: perception (this is a sub-agent)

### Recent commits on `agent/lit-review`
```
390f0f8 iter_026: curriculum advancement criteria survey — timeout_rate wrong for juggling; replace 75%→30% floor + apex_reward threshold → docs/lit_review_curriculum_advancement_criteria.md
6e45363 iter_025: D435i ROS2 integration guide — pyrealsense2 setup, Hough+median detection, calibration, async pipeline → docs/lit_review_d435i_ros2_integration.md
8f1856d iter_024: YOLO fine-tuning survey for 40mm ball detection — YOLOv8n+P2 head; mAP 0.82–0.90 at 400 imgs; TRT FP16 4ms on Orin NX → docs/lit_review_yolo_ball_detection.md
48b8cbc iter_023: D435i depth noise characterization + noise_model.py audit — σ_z quadratic 2.5× too low; white-ball dropout 4-20× too low; 848×480 mode; median-over-mask → docs/lit_review_realsense_d435i_noise.md
d889d6e iter_022: passive stability local optimum survey — ball_low_penalty -1.0 too weak; -2.0 needed; release-velocity reward as backup → docs/lit_review_passive_optimum_anti_balance.md
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
docs/lit_review_perception.md
... and 5 more
```
### Read one with:
```
git show agent/lit-review:.c3r/INBOX.md
git show agent/lit-review:.c3r/INBOX_ARCHIVE.md
git show agent/lit-review:.c3r/PROMPT.md
git show agent/lit-review:.c3r/RESEARCH_LOG.md
git show agent/lit-review:.c3r/SIBLINGS.md
```

