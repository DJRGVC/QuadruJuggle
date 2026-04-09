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
- **status**: running · iter #99 · ctx 51%

### Recent commits on `agent/perception`
```
39c8504 Iteration 112: parameterized perception eval script
1c0df1b Iteration 111: height-dependent velocity noise fix for d435i mode
9f46e49 Iteration 110: phase timeline visualization + Quarto update
cf1ea1d Iteration 109: camera scheduling — skip detection during contact phase
51e7c84 Iteration 108: ball phase tracker for flight-window detection
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
_quarto.yml
agents/index.qmd
agents/perception.qmd
docs/EKF_CONTACT_AWARE_NOTE.md
docs/hardware_pipeline_architecture.md
docs/lit_review_ekf_lag_vs_raw_noise.md
docs/lit_review_ekf_tuning.md
docs/perception_roadmap.md
docs/project_report.html
docs/sim_to_real_plan.md
experiments.qmd
experiments/perception/2026-04-09_adaptive_rxy_sweep.qmd
experiments/perception/2026-04-09_camera_pipeline_validation.qmd
experiments/perception/2026-04-09_q_vel_sweep.qmd
experiments/perception/2026-04-10_oracle_vs_d435i_eval.qmd
images/README.md
images/perception/.gitkeep
images/perception/demo_camera_ekf_summary.png
images/perception/ekf_vs_raw_by_height.png
images/perception/gpu_demo_depth_iter087.png
... and 96 more
```
### Read one with:
```
git show agent/perception:.c3r/INBOX.md
git show agent/perception:.c3r/INBOX_ARCHIVE.md
git show agent/perception:.c3r/PROMPT.md
git show agent/perception:.c3r/RESEARCH_LOG.md
git show agent/perception:.c3r/RESEARCH_LOG_ARCHIVE.md
```

