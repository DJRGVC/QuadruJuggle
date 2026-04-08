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

## perception
- **role**: generic
- **focus**: implement an onboard D435i camera to ball detector to 6-DOF EKF pipeline following ETH-style architecture, producing noisy ball observations usable by pi1 in Isaac Lab sim.
- **status**: running · iter #2 · ctx 0%
- **last iter**: 19s ago

### Recent commits on `agent/perception`
```
6aa2dc6 iter_002: Isaac Lab camera survey — CAMERA_CHOICE.md + perception package stub
747947f iter_001: D435i roadmap audit — update perception_roadmap.md for D435i sensor
aab3845 torso tracking edit
9c4793a pi_2 with v trackign works
3841f54 working on pi_1 + pi_2
```
### Files modified on `agent/perception` (relative to `c3r/QuadruJuggle`)
```
.c3r/INBOX.md
.c3r/PROMPT.md
.c3r/RESEARCH_LOG.md
.c3r/SIBLINGS.md
.c3r/agent.conf
.c3r/env.sh
.c3r/fix_plan.md
.claude/settings.json
docs/perception_roadmap.md
source/go1_ball_balance/go1_ball_balance/perception/CAMERA_CHOICE.md
source/go1_ball_balance/go1_ball_balance/perception/__init__.py
```
### Read one with:
```
git show agent/perception:.c3r/INBOX.md
git show agent/perception:.c3r/PROMPT.md
git show agent/perception:.c3r/RESEARCH_LOG.md
git show agent/perception:.c3r/SIBLINGS.md
git show agent/perception:.c3r/agent.conf
```

