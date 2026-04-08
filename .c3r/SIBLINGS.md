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
- **status**: running · iter #9 · ctx 0%
- **last iter**: 1h ago

### Recent commits on `agent/perception`
```
6ea23b9 iter_014: body-frame gravity in EKF + HANDOFF noise_scale docs
fa2c767 iter_013: noise_scale curriculum support for BallObsNoiseCfg + update_perception_noise_scale()
94874d8 iter_012: perception diagnostics + compare_perception_modes.py
0482b39 iter_011: PERCEPTION_HANDOFF.md — EKF mode integration guide for policy agent
c2ac397 iter_010: EKF integration test — full pipeline verified in Isaac Lab (4096 envs, 50 iters)
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
scripts/perception/compare_perception_modes.py
scripts/perception/debug_d435i_capture.py
scripts/perception/test_ekf_integration.py
source/go1_ball_balance/go1_ball_balance/perception/CAMERA_CHOICE.md
source/go1_ball_balance/go1_ball_balance/perception/PERCEPTION_HANDOFF.md
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

