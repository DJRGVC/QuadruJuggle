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

## policy
- **role**: generic
- **focus**: retrain pi1 with noise-injected ball observations from the perception pipeline, validate degradation versus oracle baseline and restore performance via curriculum and noise scheduling.
- **status**: running · iter #20 · ctx 81%

### Recent commits on `agent/policy`
```
e92a1df Iteration 31: Stage G training launched — mixed targets + height-dependent noise sync
67aef9d Iteration 30: cross-eval analysis + Stage G preparation
3b94cfd Iteration 29: fix eval partial-episode bug + corrected cross-eval
c1236f9 Iteration 28: cross-eval — d435i vs oracle under swapped obs modes
cce1b18 Iteration 27: d435i reaches Stage F — full comparison (apex +59% vs oracle)
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
images/policy/cross_eval_iter029.png
images/policy/d435i_vs_oracle_stage_f.png
images/policy/iter_023_training_curves.png
images/shared/.gitkeep
index.qmd
references/policy.qmd
scripts/parse_cross_eval.py
scripts/plot_cross_eval.py
scripts/rsl_rl/compare_pi1.py
scripts/rsl_rl/cross_eval.sh
... and 15 more
```
### Read one with:
```
git show agent/policy:.c3r/INBOX.md
git show agent/policy:.c3r/INBOX_ARCHIVE.md
git show agent/policy:.c3r/PROMPT.md
git show agent/policy:.c3r/RESEARCH_LOG.md
git show agent/policy:.c3r/RESEARCH_LOG_ARCHIVE.md
```

