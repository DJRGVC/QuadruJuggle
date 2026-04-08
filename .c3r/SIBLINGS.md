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

## SIBLINGS — peers you do NOT manage (other agents' work)

## perception
- **role**: generic
- **focus**: implement an onboard D435i camera to ball detector to 6-DOF EKF pipeline following ETH-style architecture, producing noisy ball observations usable by pi1 in Isaac Lab sim.
- **status**: idle · iter #54 · ctx 51%

### Recent commits on `agent/perception`
```
5c384e5 Iteration 61: sweep_q_vel.py bisection + auto-save
9608058 Iteration 60: compaction (summarized iters 056-059)
864409d Iteration 59: compaction (summarized iters 049-055)
8113692 Iteration 58: sweep script bugfixes + child cleanup
6674acd Iteration 57: post-contact P inflation + q_vel sweep script
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
docs/project_report.html
docs/sim_to_real_plan.md
scripts/perception/compare_perception_modes.py
scripts/perception/debug_d435i_capture.py
scripts/perception/eval_perception_live.py
scripts/perception/nis_diagnostic.py
scripts/perception/nis_sweep.py
scripts/perception/sweep_q_vel.py
scripts/perception/test_ballistic_trajectory.py
scripts/perception/test_contact_aware_ekf.py
scripts/perception/test_ekf_integration.py
scripts/perception/test_hough_detector.py
scripts/perception/test_imu_aided_ekf.py
scripts/perception/test_latency_injection.py
scripts/perception/test_mock_pipeline.py
scripts/perception/test_nis_gating.py
... and 30 more
```
### Read one with:
```
git show agent/perception:.c3r/INBOX.md
git show agent/perception:.c3r/INBOX_ARCHIVE.md
git show agent/perception:.c3r/PROMPT.md
git show agent/perception:.c3r/RESEARCH_LOG.md
git show agent/perception:.c3r/RESEARCH_LOG_ARCHIVE.md
```

