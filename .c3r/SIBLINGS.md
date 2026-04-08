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
- **status**: running · iter #5 · ctx 0%
- **last iter**: 48m ago

### Recent commits on `agent/perception`
```
c2ac397 iter_010: EKF integration test — full pipeline verified in Isaac Lab (4096 envs, 50 iters)
fca21ed iter_009: noise_model.py + EKF mode wired into ball_obs_spec.py
487e69b iter_008: D435i TiledCamera verified — RGB+depth frames captured
67051ce iter_007: ball_ekf.py — batched 6-state Kalman filter with ballistic+drag dynamics
32d734c iter_006: REFERENCES.md — perception architecture references + fix gym.make in debug script
```
### Files modified on `agent/perception` (relative to `c3r/QuadruJuggle`)
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
docs/perception_roadmap.md
docs/sim_to_real_plan.md
scripts/perception/debug_d435i_capture.py
scripts/perception/test_ekf_integration.py
source/go1_ball_balance/go1_ball_balance/perception/CAMERA_CHOICE.md
source/go1_ball_balance/go1_ball_balance/perception/REFERENCES.md
source/go1_ball_balance/go1_ball_balance/perception/__init__.py
source/go1_ball_balance/go1_ball_balance/perception/ball_ekf.py
source/go1_ball_balance/go1_ball_balance/perception/ball_obs_spec.py
source/go1_ball_balance/go1_ball_balance/perception/debug/.gitignore
source/go1_ball_balance/go1_ball_balance/perception/debug/__init__.py
source/go1_ball_balance/go1_ball_balance/perception/noise_model.py
source/go1_ball_balance/go1_ball_balance/tasks/ball_juggle_hier/ball_juggle_hier_env_cfg.py
```
### Read one with:
```
git show agent/perception:.c3r/INBOX.md
git show agent/perception:.c3r/INBOX_ARCHIVE.md
git show agent/perception:.c3r/PROMPT.md
git show agent/perception:.c3r/RESEARCH_LOG.md
git show agent/perception:.c3r/SIBLINGS.md
```

