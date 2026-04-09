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
- **status**: running · iter #73 · ctx 51%

### Recent commits on `agent/perception`
```
eeb9d53 Iteration 86: GPU demo queued + timeout/sentinel improvements
26ac01f Iteration 85: bounce demo mode + combined GPU runner
2b3bc0c Iteration 84: pixel projection tests + camera convention derivation
4ad66d0 Iteration 83: Quarto update + video workflow + child cleanup
be3b5c5 Iteration 82: sim pipeline integration tests — 8 new tests (294/294 pass)
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
experiments/perception/2026-04-09_q_vel_sweep.qmd
images/README.md
images/perception/.gitkeep
images/perception/q_vel_sweep_combined.png
images/perception/q_vel_sweep_iter_069.png
images/policy/.gitkeep
images/shared/.gitkeep
index.qmd
... and 58 more
```
### Read one with:
```
git show agent/perception:.c3r/INBOX.md
git show agent/perception:.c3r/INBOX_ARCHIVE.md
git show agent/perception:.c3r/PROMPT.md
git show agent/perception:.c3r/RESEARCH_LOG.md
git show agent/perception:.c3r/RESEARCH_LOG_ARCHIVE.md
```

## checkin-087
- **role**: generic
- **focus**: Check perception and policy progress, update testing-dashboard status table
- **status**: running · iter #2 · ctx 0%
- **parent**: perception (this is a sub-agent)

### Recent commits on `agent/checkin-087`
```
d4ac9cb Iteration 2: policy Stage F (step 2879, apex=1.41) + dashboard update + role clarification
50cb036 Iteration 1: initial status check + testing-dashboard.md
aab3845 torso tracking edit
9c4793a pi_2 with v trackign works
3841f54 working on pi_1 + pi_2
```
### Files modified on `agent/checkin-087` (relative to `c3r/QuadruJuggle`)
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
docs/testing-dashboard.md
```
### Read one with:
```
git show agent/checkin-087:.c3r/INBOX.md
git show agent/checkin-087:.c3r/INBOX_ARCHIVE.md
git show agent/checkin-087:.c3r/PROMPT.md
git show agent/checkin-087:.c3r/RESEARCH_LOG.md
git show agent/checkin-087:.c3r/SIBLINGS.md
```

