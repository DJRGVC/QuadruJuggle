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

## YOUR CHILDREN — agents YOU spawned and YOU must manage

These are sub-agents you spawned (directly or transitively).
**YOU are responsible for killing them when their task is done,
they get stuck, or they exceed their useful budget.** Each child
also has a hard iteration cap and will self-kill at MAX_ITERATIONS,
but that's a safety net — proactive management is your job.

- **checkin-087** (generic, parent=perception) — status=idle, iter=#4, last=4m ago
  Focus: Check perception and policy progress, update testing-dashboard status table

**Decision rules** (apply at the top of every iteration):
1. If a child's last RESEARCH_LOG entry says its task is done, kill it: `$C3R_BIN/c3r kill <name>`
2. If a child has been stale (no iter for >2 hours), kill it.
3. If a child's fail_streak ≥ 3 in state.json, investigate or kill it.
4. Otherwise, leave it running and check again next iteration.

---

## SIBLINGS — peers you do NOT manage (other agents' work)

## policy
- **role**: generic
- **focus**: retrain pi1 with noise-injected ball observations from the perception pipeline, validate degradation versus oracle baseline and restore performance via curriculum and noise scheduling.
- **status**: running · iter #16 · ctx 81%
- **last iter**: 10m ago

### Recent commits on `agent/policy`
```
cce1b18 Iteration 27: d435i reaches Stage F — full comparison (apex +59% vs oracle)
a7c6d74 Iteration 26: d435i vs oracle comparison — noise outperforms oracle at Stage E (+176% apex)
56ee92e Iteration 25: sync perception noise model (Ahn 2019 calibration) + prepare d435i training
b99bfab Iteration 24: discovered iters 22-23 ran oracle (not d435i) — noise_mode flag was missing
bf3d8a8 Iteration 23: σ_ratio 3.5→2.5 breaks Stage D plateau — all 6 stages reached (A→F)
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
_quarto.yml
agents/index.qmd
agents/perception.qmd
agents/policy.qmd
docs/noise_curriculum_plan.md
experiments/policy/2026-04-09_d435i_vs_oracle_curriculum.qmd
images/README.md
images/perception/.gitkeep
images/policy/.gitkeep
images/policy/d435i_vs_oracle_stage_f.png
images/policy/iter_023_training_curves.png
images/shared/.gitkeep
index.qmd
references/policy.qmd
scripts/rsl_rl/compare_pi1.py
scripts/rsl_rl/eval_juggle_hier.py
scripts/rsl_rl/play_teleop.py
scripts/rsl_rl/train_juggle_hier.py
source/go1_ball_balance/go1_ball_balance/perception/__init__.py
... and 10 more
```
### Read one with:
```
git show agent/policy:.c3r/INBOX.md
git show agent/policy:.c3r/INBOX_ARCHIVE.md
git show agent/policy:.c3r/PROMPT.md
git show agent/policy:.c3r/RESEARCH_LOG.md
git show agent/policy:.c3r/RESEARCH_LOG_ARCHIVE.md
```

