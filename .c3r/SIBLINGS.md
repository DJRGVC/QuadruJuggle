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
- **status**: running · iter #130 · ctx 51%

### Recent commits on `agent/perception`
```
2c6bd05 Iteration 144: Pipeline latency tracking for hardware bring-up
758991e Iteration 143: Extrinsics to_yaml() round-trip + Quarto update
71fc36c Iteration 142: Implement CameraCalibrator.from_checkerboard()
b0d4b31 Iteration 141: Implement D435iCamera pyrealsense2 wrapper
8b888a9 Iteration 140: Re-validate EKF error decomposition with linear drag
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
experiments/perception/2026-04-09_stage_g_bottleneck_analysis.qmd
experiments/perception/2026-04-10_noise_gap_prediction.qmd
experiments/perception/2026-04-10_oracle_vs_d435i_eval.qmd
experiments/perception/2026-04-10_perception_gap_decomposition.qmd
images/README.md
images/perception/.gitkeep
... and 150 more
```
### Read one with:
```
git show agent/perception:.c3r/INBOX.md
git show agent/perception:.c3r/INBOX_ARCHIVE.md
git show agent/perception:.c3r/PROMPT.md
git show agent/perception:.c3r/RESEARCH_LOG.md
git show agent/perception:.c3r/RESEARCH_LOG_ARCHIVE.md
```

## fix-keyboard-vision-control-ui
- **role**: fix-it
- **focus**: great. so what is left for me being able to have keyboard-based control of a vis
- **status**: running · iter #0 · ctx 0%

### Recent commits on `agent/fix-keyboard-vision-control-ui`
```
aab3845 torso tracking edit
9c4793a pi_2 with v trackign works
3841f54 working on pi_1 + pi_2
085e457 frank-proofed requirements.txt
7256612 added req + install explainer
```
### Files modified on `agent/fix-keyboard-vision-control-ui` (relative to `c3r/QuadruJuggle`)
_(none)_

