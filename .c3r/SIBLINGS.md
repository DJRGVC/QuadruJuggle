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

- **lit-review** (generic, parent=perception) — status=stopped, iter=#28, last=7h ago  ⚠ STALE — consider killing  (already stopped)
  Focus: survey perception-for-manipulation papers (ETH noise injection, teacher-student distillation, event cameras, learned state estimators) and critique our D435i+EKF approach
- **vel-cmd-survey** (generic, parent=perception) — status=stopped, iter=#6, last=7m ago  (already stopped)
  Focus: Survey 2023-2026 papers on user-defined velocity command input for legged robots doing manipulation while walking. Our Go1 has hierarchical pi2 torso tracker (8D: h,hdot,roll,pitch,omega_r,omega_p,vx,vy) and pi1 ball planner. Propose 3 methods for adding user velocity commands. Track references. Report to parent perception every iter via INBOX. Complete in 5 iters MAX then stop.
- **report-writer** (generic, parent=perception) — status=stopped, iter=#9, last=9m ago  (already stopped)
  Focus: Create a nicely formatted project report (HTML or Markdown with embedded images/graphs) covering the QuadruJuggle project: perception pipeline, policy training, architecture, and results. Max 10 iterations.

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
- **status**: running · iter #10 · ctx 81%
- **last iter**: 39m ago

### Recent commits on `agent/policy`
```
206e928 iter_017: compaction (summarized iters 009-014)
e33e0d8 iter_016: continue training from step 5748 — apex plateau at 10.7 after curriculum advance
c5f3d4e iter_015: curriculum threshold 0.75→0.30 — CURRICULUM ADVANCES, juggling sustained
f77d50e iter_014: ball_release_velocity_reward SUSTAINS JUGGLING (apex 9.7 stable, 1500 iters)
dc7bb6f iter_014: add ball_release_velocity_reward (+3.0)
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
docs/noise_curriculum_plan.md
scripts/rsl_rl/compare_pi1.py
scripts/rsl_rl/eval_juggle_hier.py
scripts/rsl_rl/train_juggle_hier.py
source/go1_ball_balance/go1_ball_balance/perception/__init__.py
source/go1_ball_balance/go1_ball_balance/perception/ball_ekf.py
source/go1_ball_balance/go1_ball_balance/perception/ball_obs_spec.py
source/go1_ball_balance/go1_ball_balance/perception/noise_model.py
source/go1_ball_balance/go1_ball_balance/tasks/ball_juggle/mdp/rewards.py
source/go1_ball_balance/go1_ball_balance/tasks/ball_juggle_hier/ball_juggle_hier_env_cfg.py
source/go1_ball_balance/go1_ball_balance/tasks/torso_tracking/action_term.py
```
### Read one with:
```
git show agent/policy:.c3r/INBOX.md
git show agent/policy:.c3r/INBOX_ARCHIVE.md
git show agent/policy:.c3r/PROMPT.md
git show agent/policy:.c3r/RESEARCH_LOG.md
git show agent/policy:.c3r/RESEARCH_LOG_ARCHIVE.md
```

