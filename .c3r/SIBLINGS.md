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
- **status**: running · iter #65 · ctx 51%

### Recent commits on `agent/perception`
```
75d0294 Iteration 78: fix demo_camera_ekf.py frame consistency — use cam.data.quat_w_ros
62631fc Iteration 77: convention fix ros→world + sim ball detector
7250413 Iteration 76: camera quaternion convention fix — 70° tilt with correct ros convention
51db80e Iteration 76: camera FOV fix — 45° → 75° tilt, debug capture with ball impulse
90ffdce Iteration 75: compaction (summarized iters 060-073)
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
experiments/perception/2026-04-09_q_vel_sweep.qmd
images/README.md
images/perception/.gitkeep
images/perception/q_vel_sweep_combined.png
images/perception/q_vel_sweep_iter_069.png
images/policy/.gitkeep
images/shared/.gitkeep
index.qmd
references.qmd
... and 51 more
```
### Read one with:
```
git show agent/perception:.c3r/INBOX.md
git show agent/perception:.c3r/INBOX_ARCHIVE.md
git show agent/perception:.c3r/PROMPT.md
git show agent/perception:.c3r/RESEARCH_LOG.md
git show agent/perception:.c3r/RESEARCH_LOG_ARCHIVE.md
```

## testing-dashboard
- **role**: generic
- **focus**: Build a Quarto testing page for the WASD+PL control + perception pipeline. Deadline: 2026-04-09 15:00 UTC. Page documents how to run play_teleop.py with WASD+PL controls, camera setup, height viz, velocity display. Decide by iter 3: live widget vs step-by-step guide for local Linux with Isaac Lab. Read sibling branches for model paths.
- **status**: running · iter #5 · ctx 0%
- **parent**: perception (this is a sub-agent)

### Recent commits on `agent/testing-dashboard`
```
693223b Iteration 5: quality polish — quick-start block, live status table, sibling check-in
f818a16 Iteration 4: startup walkthrough + verified checkpoint paths + --noise-mode in Step 2
1bea42d Iteration 3: perception pipeline docs — D435i→EKF section with noise model, camera geometry, --noise-mode guide
473cb76 Iteration 2: 8D interface reference + expanded architecture diagram
bfbb4fc Iteration 1: core Quarto page — WASD+PL teleop guide + references
```
### Files modified on `agent/testing-dashboard` (relative to `c3r/QuadruJuggle`)
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
agents/testing-dashboard.qmd
experiments/testing-dashboard/.gitkeep
images/testing-dashboard/.gitkeep
references/testing-dashboard.qmd
```
### Read one with:
```
git show agent/testing-dashboard:.c3r/INBOX.md
git show agent/testing-dashboard:.c3r/INBOX_ARCHIVE.md
git show agent/testing-dashboard:.c3r/PROMPT.md
git show agent/testing-dashboard:.c3r/RESEARCH_LOG.md
git show agent/testing-dashboard:.c3r/SIBLINGS.md
```

