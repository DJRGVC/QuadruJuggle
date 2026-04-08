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

## policy
- **role**: generic
- **focus**: retrain pi1 with noise-injected ball observations from the perception pipeline, validate degradation versus oracle baseline and restore performance via curriculum and noise scheduling.
- **status**: running · iter #1 · ctx 0%
- **last iter**: 22m ago

### Recent commits on `agent/policy`
```
20cc01f iter_002: fix pi2 obs dim (41→53D auto-detect) + 500-iter oracle baseline
b0c014c iter_001: oracle pi1 baseline (500 iter, Stage D, mean_len=1470)
aab3845 torso tracking edit
9c4793a pi_2 with v trackign works
3841f54 working on pi_1 + pi_2
```
### Files modified on `agent/policy` (relative to `c3r/QuadruJuggle`)
```
.c3r/INBOX.md
.c3r/PROMPT.md
.c3r/RESEARCH_LOG.md
.c3r/SIBLINGS.md
.c3r/agent.conf
.c3r/env.sh
.c3r/fix_plan.md
.claude/settings.json
source/go1_ball_balance/go1_ball_balance/tasks/torso_tracking/action_term.py
```
### Read one with:
```
git show agent/policy:.c3r/INBOX.md
git show agent/policy:.c3r/PROMPT.md
git show agent/policy:.c3r/RESEARCH_LOG.md
git show agent/policy:.c3r/SIBLINGS.md
git show agent/policy:.c3r/agent.conf
```

