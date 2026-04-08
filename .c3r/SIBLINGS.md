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
- **status**: running · iter #11 · ctx 0%
- **last iter**: 8m ago

### Recent commits on `agent/perception`
```
c6b53dc iter_016: fix EKF vel-view covariance bug + 3-mode perception comparison
6d0e6f8 iter_015: subprocess isolation for compare script + lit-review subagent spawned
6ea23b9 iter_014: body-frame gravity in EKF + HANDOFF noise_scale docs
fa2c767 iter_013: noise_scale curriculum support for BallObsNoiseCfg + update_perception_noise_scale()
94874d8 iter_012: perception diagnostics + compare_perception_modes.py
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

## lit-review
- **role**: generic
- **focus**: survey perception-for-manipulation papers (ETH noise injection, teacher-student distillation, event cameras, learned state estimators) and critique our D435i+EKF approach
- **status**: running · iter #4 · ctx 0%
- **last iter**: 4m ago
- **parent**: perception (this is a sub-agent)

### Recent commits on `agent/lit-review`
```
5a4c9ab iter_004: ball spin survey → docs/lit_review_ball_spin.md (Q-inflation sufficient for Stage A-D; upgrade path at Stage E+)
16c8cf4 iter_003: EKF Q/R tuning + ANEES diagnostic → docs/lit_review_ekf_tuning.md
ac06c8c iter_002: noise curriculum + latency injection literature → docs/lit_review_noise_curriculum.md
cfa644b iter_001: literature survey — 8 papers + D435i+EKF critique → docs/lit_review_perception.md
aab3845 torso tracking edit
```
### Files modified on `agent/lit-review` (relative to `c3r/QuadruJuggle`)
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
docs/lit_review_ball_spin.md
docs/lit_review_ekf_tuning.md
docs/lit_review_noise_curriculum.md
docs/lit_review_perception.md
```
### Read one with:
```
git show agent/lit-review:.c3r/INBOX.md
git show agent/lit-review:.c3r/INBOX_ARCHIVE.md
git show agent/lit-review:.c3r/PROMPT.md
git show agent/lit-review:.c3r/RESEARCH_LOG.md
git show agent/lit-review:.c3r/SIBLINGS.md
```

