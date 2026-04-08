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
- **status**: running · iter #4 · ctx 0%
- **last iter**: 1h ago

### Recent commits on `agent/perception`
```
67051ce iter_007: ball_ekf.py — batched 6-state Kalman filter with ballistic+drag dynamics
32d734c iter_006: REFERENCES.md — perception architecture references + fix gym.make in debug script
2c75ac7 iter_005b: fix debug_d435i_capture.py — instantiate env_cfg directly instead of Hydra lookup
a2d3927 iter_005: mount simulated D435i TiledCamera in PLAY scene + debug capture script
1919aab iter_004: sim_to_real_plan.md — update Track 3 + latency + risks for D435i rear-paddle mount
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
docs/sim_to_real_plan.md
scripts/perception/debug_d435i_capture.py
source/go1_ball_balance/go1_ball_balance/perception/CAMERA_CHOICE.md
source/go1_ball_balance/go1_ball_balance/perception/REFERENCES.md
source/go1_ball_balance/go1_ball_balance/perception/__init__.py
source/go1_ball_balance/go1_ball_balance/perception/ball_ekf.py
source/go1_ball_balance/go1_ball_balance/perception/ball_obs_spec.py
source/go1_ball_balance/go1_ball_balance/perception/debug/__init__.py
source/go1_ball_balance/go1_ball_balance/tasks/ball_juggle_hier/ball_juggle_hier_env_cfg.py
```
### Read one with:
```
git show agent/perception:.c3r/INBOX.md
git show agent/perception:.c3r/PROMPT.md
git show agent/perception:.c3r/RESEARCH_LOG.md
git show agent/perception:.c3r/SIBLINGS.md
git show agent/perception:.c3r/agent.conf
```

