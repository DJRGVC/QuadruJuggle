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
- **status**: running · iter #64 · ctx 51%

### Recent commits on `agent/perception`
```
7250413 Iteration 76: camera quaternion convention fix — 70° tilt with correct ros convention
51db80e Iteration 76: camera FOV fix — 45° → 75° tilt, debug capture with ball impulse
90ffdce Iteration 75: compaction (summarized iters 060-073)
9015cfc Iteration 74: adaptive R_xy sweep validates NIS calibration — all flight NIS 3.3-3.8
a3243d5 Iteration 73: adaptive R_xy — fix root cause of all NIS < 3.0
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
... and 48 more
```
### Read one with:
```
git show agent/perception:.c3r/INBOX.md
git show agent/perception:.c3r/INBOX_ARCHIVE.md
git show agent/perception:.c3r/PROMPT.md
git show agent/perception:.c3r/RESEARCH_LOG.md
git show agent/perception:.c3r/RESEARCH_LOG_ARCHIVE.md
```

