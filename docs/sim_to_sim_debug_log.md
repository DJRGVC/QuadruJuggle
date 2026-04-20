# Sim-to-Sim Transfer Debug Log
**Project**: QuadruJuggle — Isaac Lab → MuJoCo transfer for pi1 + pi2  
**Goal**: Run the trained pi2 (torso-tracking) and pi1 (juggling) policies in MuJoCo to enable fast, headless evaluation.

---

## Phase 1 — Initial Suspects

Three categories of mismatch were suspected from the start:

### 1. Joint Ordering Disorder
Isaac Lab uses **type-grouped** joint ordering (all hips → all thighs → all calves):
```
[FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
```
MuJoCo MJCF uses **leg-grouped** ordering (all joints of one leg together):
```
[FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
```
Without reindexing, joint targets sent to the actuator net are assigned to the wrong physical joints — the robot immediately falls.

**Fix**: Built `build_reindex(model)` in `mujoco_utils.py` that reads joint names from the XML at runtime and computes `REINDEX[mjcf_i] = isaac_i`. Applied consistently when constructing observations (`qpos[7:19][reindex]`, `qvel[6:18][reindex]`) and when sending targets to `data.ctrl`.

### 2. Actuator Net Mismatch
The actuator net (`unitree_go1.pt`) is a TorchScript MLP with 3-step position-error and velocity history. Two issues were identified:

- **Cold start**: History initialises to zero, so the first 3 policy steps see stale (zero) history — output torques are wrong. Fixed by running a warmup loop to fill the history.
- **Torque scale gap**: Isaac Lab PhysX produces a stronger physical response per Nm than MuJoCo. Compensated with `TORQUE_SCALE = 2.5` applied to all actuator net outputs before writing to `data.ctrl`.

### 3. Timestep / Decimation Mismatch
Isaac Lab trains with `sim.dt = 0.005 s`, `decimation = 4` → policy runs at **50 Hz**.  
In MuJoCo, the XML timestep must match (`<option timestep="0.005"/>`), and the outer loop must apply each policy action for exactly `DECIMATION = 4` physics steps before querying the policy again.

---

## Phase 2 — Evidence: Joint Step-Response Comparison

To verify the three fixes above, built `scripts/tests/compare_joint.py`:

- **Fixed base**: Isaac Lab uses `fix_root_link=True` on the robot ArticulationCfg; MuJoCo calls `pin_base()` (overwrites `qpos[0:7]` and zeros `qvel[0:6]`) before every step. Both spawn at z=1.0m above ground.
- **Same step input**: Both sims start from default joint positions then apply a fixed +0.3 rad step target to all joints simultaneously at `t=HOLD`.
- **12-joint overlay plot** (`tests_out/compare_joint.png`): REINDEX pairs each MJCF joint with its Isaac Lab counterpart correctly — calves match well, hips show expected divergence from multi-body coupling differences between PhysX and MuJoCo.

The comparison confirmed joint ordering and decimation are aligned. Remaining divergence (mainly hips) is expected physics-engine difference, not a bug.

---

## Phase 3 — Pi2 Still Falls in MuJoCo

With the above fixes in place, `test_pi2_mujoco.py` runs but the robot falls within the first 50 steps every time. The printout at step 0 revealed the root cause:

```
[warmup done] trunk_z=0.304
  step     0 | trunk_z=0.312m  tilt=0.3deg
    obs[15:27] jp_rel : [ 0.700  -1.497  0.031  -0.042  0.001  -2.366
                         -0.858   0.017  -0.068  -0.095   1.370   2.518]
    action             : [-2.731  0.233  1.193  0.703 -0.043  1.278 -5.855 -6.455  2.603  6.597  4.882  4.641]
                         |max|=6.597
    torques            : [-10.5  23.7  23.7 -23.7  12.7  23.7 -11.  -23.7  23.7 -14.5 -23.7  23.7]
```

The robot is *upright* at step 0 (tilt=0.3°) but joint positions are up to ±2.5 rad from default. **Pi2 sees this as a large error and immediately outputs saturating actions — the robot falls by step 50.**

**Root cause**: The warmup loop was calling `actuator.compute(default_mjcf, ...)`, but the actuator net (trained on hardware/PhysX) does not converge to `default_mjcf` as a fixed point in MuJoCo. The robot reached a different equilibrium (lower, wrong joint config).

**Fix** (`reset()` in `test_pi2_mujoco.py`):
1. Replaced actuator-net warmup with **plain PD control** targeting `default_mjcf`:  
   `ctrl = clip(KP * (default - qpos) - KD * qvel, ±23.7)`
2. Increased warmup from 200 → **500 steps** to let PD fully converge.
3. Changed spawn height from 0.35 → **0.42 m** (matching Isaac Lab's `UNITREE_GO1_CFG.init_state.pos`).
4. Added `actuator.reset()` *after* PD warmup so the actuator net history starts clean.
5. Prints `jp_err_max` at warmup end to verify convergence before the main loop.

Expected result: `trunk_z ≈ 0.37–0.38 m`, `jp_err_max < 0.01`, `jp_rel ≈ 0` at step 0 → pi2 outputs small actions → robot holds stance.

---

## Phase 4 — Pi2 Standalone Isaac Lab Test (Two Bugs Found and Fixed)

To isolate pi2's behaviour independently of the full gym env, built `scripts/tests/test_pi2_isaaclab.py` using `SimulationContext` + `InteractiveScene` directly — same pattern as `compare_joint.py`. This lets us inspect every obs term individually without the gym env as a black box.

Two bugs were found and fixed during this phase.

### Bug A — Warmup Left Joints 0.42 rad From Default

**What happened**: The warmup loop called `robot.set_joint_position_target(default_pos)` for 200 steps. Despite this, `jp_err_max = 0.4176` at warmup end — joints were still nearly half a radian off default. The robot appeared upright (tilt=0.3°) but crouched (trunk_z=0.267m). Pi2 saw this as a large joint error and immediately output saturating actions, flipping the robot by step 50.

**Why**: `set_joint_position_target()` works through the actuator net MLP (GO1_ACTUATOR_CFG). The actuator net was trained on hardware dynamics, not to act as a position controller in isolation. It converges slowly and settles at a different equilibrium than the exact default joint position.

Think of it like telling someone "hold this pose" but giving them instructions written for a different person's body — they'll do their best but end up somewhere close, not exact.

**Fix**: Called `robot.write_joint_state_to_sim(default_pos, default_vel)` before the warmup loop. This is the same call the gym env's `reset_robot_joints` event uses internally — it **directly writes** the joint angles into the physics engine, bypassing the actuator net entirely. Joints are at exactly default from frame 0. The subsequent warmup steps then just let the physics settle around that correct starting point.

After fix: `jp_err_max < 0.001` → pi2 sees near-zero `jp_rel` → outputs small actions → robot holds stance.

### Bug B — Render Called Every Physics Sub-step Instead of Every Policy Step

**What happened**: The decimation loop in the standalone test used:
```python
sim.step(render=True)   # renders on every one of the 4 sub-steps
```

The gym env's actual loop (read from `manager_based_rl_env.py` source) uses:
```python
sim.step(render=False)                                    # physics only
if step_counter % render_interval == 0 and is_rendering:
    sim.render()                                          # render once per 4 steps
```

**Why it matters**: Rendering doesn't change the physics, but calling it 4× more often than the gym env means the sim spends extra time on rendering between physics steps. More importantly, the pattern was structurally wrong relative to training — if the render call ever has side effects on the scene graph (USD sync, sensor updates), those would fire at 4× the expected rate.

**Fix**: Changed the loop to `sim.step(render=False)` every sub-step and `sim.render()` only on the last sub-step of each policy step, matching the gym env exactly.

### Validation — Golden Test Cases

To confirm the actor loading code is correct independently of the scene, built `scripts/tests/test_pi2_golden.py`:
- Runs the gym env for 10 fixed torso commands, records `(obs, action)` pairs at steady state
- Saves to `tests_out/pi2_golden.csv`

Then `scripts/tests/verify_pi2_golden.py` reads the CSV, feeds the recorded obs directly into the actor, and compares the output to the stored action. Result: all 10 cases match to within `5e-6` (pure float32 CSV rounding). This proves the actor loading function is identical to what the gym env uses.

**Note on a recording bug found during validation**: The first version of the golden collector recorded `obs` from *after* the final `env.step()` but `action` from *before* it — a one-timestep mismatch causing errors up to 2.75 in the "mixed" case. Fixed by recording both obs and action at the same timestep before stepping.

---

---

## Phase 5 — Pi2 MuJoCo: Three More Bugs

After the Phase 3 warmup fix, three additional issues were found and fixed.

### Bug A — Base Floats During PD Warmup

**What happened**: Even after hard-setting joints to default, running 500 PD warmup steps with the base floating let gravity pull the robot down. Joints drifted from default because the legs had to bear the body weight, which PD (KP=100) couldn't fully resist. `jp_err_max=0.237` after warmup.

**Fix**: Pin the base at every warmup step — after each `mj_step`, overwrite `data.qpos[0:3]`, `data.qpos[3:7]`, and `data.qvel[0:6]` with the spawn values. This mirrors Isaac Lab's `fix_root_link=True`. Result: `jp_err_max=0.000` after warmup.

### Bug B — Actuator Net History Cold Start After Hard-Set

**What happened**: After the PD warmup, we hard-set joints to exact default (`jp_err_max=0.000`) but then called `actuator.reset()`, zeroing the 3-step history buffer. The first 3 policy steps see stale (zero) history — the actuator net outputs wrong torques and the robot destabilises immediately after warmup.

**Fix**: After the hard-set, run a second brief warmup (20 steps) using the actuator net with the base still pinned, targeting the default position. This fills the history buffer with near-zero position errors — the correct state for a robot holding its default stance. Then hard-set joints once more and release the base.

### Bug C — Obs Reindex Direction Inverted (Root Cause of All MuJoCo Falls)

**What happened**: The observation construction used:
```python
jp_rel   = data.qpos[7:19][reindex] - DEFAULT_JOINT_POS_ISAAC   # wrong
jv_isaac = data.qvel[6:18][reindex]                              # wrong
```

NumPy fancy indexing `a[reindex]` is a **gather** operation: `result[i] = a[reindex[i]]`. Since `reindex[mjcf_i] = isaac_i`, this produces `result[i] = a[isaac_i]` — an Isaac→MJCF mapping applied in the wrong direction to data that's already in MJCF order. The result is a nonsensical permutation.

**Why it wasn't caught earlier**: The warmup diagnostic `jp_err_max` computed `np.abs(data.qpos[7:19] - default_mjcf).max()` in raw MJCF space — never applying `[reindex]` — so it read zero correctly. The bug only fired when the obs was constructed for the policy.

**Evidence**: At step 0 (joints perfectly at default), `jp_rel_max = 2.5` — exactly equal to `DEFAULT_JOINT_POS_ISAAC.max() - DEFAULT_JOINT_POS_ISAAC.min()`. Pi2 saw huge apparent joint errors and immediately output saturating actions.

**Fix**: Use NumPy **scatter** instead of gather for MJCF→Isaac conversion:
```python
jp_isaac = np.zeros(12); jp_isaac[reindex] = data.qpos[7:19]   # scatter: correct
jv_isaac = np.zeros(12); jv_isaac[reindex] = data.qvel[6:18]   # scatter: correct
jp_rel   = jp_isaac - DEFAULT_JOINT_POS_ISAAC
```

The direction of `[reindex]` depends on what you're doing:
| Operation | Direction | Correct form |
|---|---|---|
| MJCF values → Isaac order (obs) | MJCF→Isaac | `result[reindex] = a_mjcf` (scatter) |
| Isaac targets → MJCF order (ctrl) | Isaac→MJCF | `a_mjcf = target_isaac[reindex]` (gather) |

The target conversion `(DEFAULT_JOINT_POS_ISAAC + scale * action)[reindex]` was already correct. The actuator net internally uses scatter for its inputs and gather for its output, also correctly.

**Result after all three fixes**: `jp_err_max=0.000` after warmup, `jp_rel_max≈0` at step 0, `trunk_z≈0.411m`, `tilt≈0.6deg`, stable for 1000+ steps without falling.

---

## Current State

| Issue | Status |
|---|---|
| Joint reindexing (MJCF ↔ Isaac Lab) | ✅ Fixed — `build_reindex()` |
| Actuator net cold start | ✅ Fixed — PD warmup → `actuator.reset()` |
| Timestep / decimation alignment | ✅ Fixed — `dt=0.005`, `DECIMATION=4` |
| Pi2 working in Isaac Lab (gym env) | ✅ Confirmed |
| Pi2 warmup wrong joints (standalone IL) | ✅ Fixed — `write_joint_state_to_sim()` |
| Render cadence in standalone IL | ✅ Fixed — `render=False` + separate `sim.render()` |
| Actor loading verified bit-exact | ✅ Golden test: max error 5e-6 across 10 cases |
| MuJoCo warmup base floats | ✅ Fixed — pin base at every warmup step |
| MuJoCo actuator net cold start | ✅ Fixed — actuator net warmup phase after hard-set |
| MuJoCo obs reindex inverted | ✅ Fixed — scatter `result[reindex]=qpos` not gather `qpos[reindex]` |
| Pi2 stable in MuJoCo | ✅ Stable 1000+ steps: trunk_z=0.411m, tilt=0.6deg |
