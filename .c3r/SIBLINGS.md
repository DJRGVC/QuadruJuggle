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
- **status**: paused · iter #59 · ctx 51%

### Recent commits on `agent/perception`
```
fe70ae9 Iteration 69: GPU q_vel sweep — EKF over-conservative, all flight NIS < 3.0
50b1cef Iteration 68: fix numpy→tensor warnings in tests
64cf3f9 Iteration 67: fix diagnostics bug in compare_perception_modes.py
8e55f7b Iteration 66: Quarto report + sweep status check
02ecb22 Iteration 65: fix sweep diagnostics (restore pipeline recreation)
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
images/README.md
images/perception/.gitkeep
images/policy/.gitkeep
images/shared/.gitkeep
index.qmd
scripts/perception/apply_sweep_results.py
scripts/perception/compare_perception_modes.py
scripts/perception/conftest.py
scripts/perception/debug_d435i_capture.py
scripts/perception/eval_perception_live.py
... and 42 more
```
### Read one with:
```
git show agent/perception:.c3r/INBOX.md
git show agent/perception:.c3r/INBOX_ARCHIVE.md
git show agent/perception:.c3r/PROMPT.md
git show agent/perception:.c3r/RESEARCH_LOG.md
git show agent/perception:.c3r/RESEARCH_LOG_ARCHIVE.md
```

