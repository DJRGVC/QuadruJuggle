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
- **status**: idle · iter #10 · ctx 81%

### Recent commits on `agent/policy`
```
6e99ed8 Iteration 18: apex plateau diagnosis + Quarto report
206e928 iter_017: compaction (summarized iters 009-014)
e33e0d8 iter_016: continue training from step 5748 — apex plateau at 10.7 after curriculum advance
c5f3d4e iter_015: curriculum threshold 0.75→0.30 — CURRICULUM ADVANCES, juggling sustained
f77d50e iter_014: ball_release_velocity_reward SUSTAINS JUGGLING (apex 9.7 stable, 1500 iters)
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
images/README.md
images/perception/.gitkeep
images/policy/.gitkeep
images/shared/.gitkeep
index.qmd
scripts/rsl_rl/compare_pi1.py
scripts/rsl_rl/eval_juggle_hier.py
scripts/rsl_rl/train_juggle_hier.py
source/go1_ball_balance/go1_ball_balance/perception/__init__.py
source/go1_ball_balance/go1_ball_balance/perception/ball_ekf.py
source/go1_ball_balance/go1_ball_balance/perception/ball_obs_spec.py
source/go1_ball_balance/go1_ball_balance/perception/noise_model.py
source/go1_ball_balance/go1_ball_balance/tasks/ball_juggle/mdp/rewards.py
source/go1_ball_balance/go1_ball_balance/tasks/ball_juggle_hier/ball_juggle_hier_env_cfg.py
... and 5 more
```
### Read one with:
```
git show agent/policy:.c3r/INBOX.md
git show agent/policy:.c3r/INBOX_ARCHIVE.md
git show agent/policy:.c3r/PROMPT.md
git show agent/policy:.c3r/RESEARCH_LOG.md
git show agent/policy:.c3r/RESEARCH_LOG_ARCHIVE.md
```

