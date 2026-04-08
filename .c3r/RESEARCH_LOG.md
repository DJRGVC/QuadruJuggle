# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## iter_001 — D435i roadmap audit  (2026-04-07T00:00:00Z)
Hypothesis: The existing perception_roadmap.md was written for a mono USB camera; updating it for the D435i will expose concrete differences in noise model parameters, detection pipeline, and EKF timing that will guide subsequent simulation work.
Change:     Updated docs/perception_roadmap.md to reflect D435i sensor throughout: Phase 3 camera selection (D435i chosen, ELP rejected with reasoning), Phase 4 detection (depth from D435i depth frame not apparent ball size), Phase 2 noise model (σ_base 3mm→1-2mm, σ_dist 5mm/m→2mm/m, latency ~1 policy step), EKF timing (60-120Hz→30-90Hz depth), deployment stack (D435i USB3 + predict-every-5ms EKF).
Command:    Edit docs/perception_roadmap.md (6 targeted edits, no code written)
Result:     Roadmap now accurately reflects D435i. Key implication: depth accuracy is ~3mm at 1m vs ~65mm/m for mono — noise model is ~10× better at long range, which matters for Stage E-G (0.6-1.0m apex). Dropout mode changes from "ball too small to detect" to "IR reflection off shiny surface".
Decision:   Next iter: task 2 — survey Isaac Lab camera sensor API (RayCasterCamera vs TiledCamera) and document choice for D435i semantics simulation.

---

## iter_002 — Isaac Lab camera sensor survey → CAMERA_CHOICE.md  (2026-04-07T00:30:00Z)
Hypothesis: Surveying RayCasterCamera vs TiledCamera vs base Camera will reveal a clear winner for D435i semantics, with a key architectural finding that the ETH noise-injection approach requires no camera at all during training.
Change:     Created `source/go1_ball_balance/go1_ball_balance/perception/__init__.py` (perception package stub) and `perception/CAMERA_CHOICE.md` (full sensor survey + decision doc).
Command:    Read isaaclab/sensors/ray_caster/ray_caster_camera.py, tiled_camera.py, camera.py, sensor_base_cfg.py; wrote CAMERA_CHOICE.md.
Result:     Key finding: RayCasterCamera DISQUALIFIED — only works on static meshes, cannot see the moving ball. TiledCamera is the correct debug option (RGB + depth, dynamic objects, parallel). But the training pipeline uses NO camera sensor — ETH architecture injects noise directly on GT state, batched across 12288 envs on GPU. TiledCamera config spec written for debug/play use (30Hz, 640×480, 45° upward mount).
Decision:   Next iter: task 3 — create `perception/ball_obs_spec.py` stub returning oracle ball state with explicit zero-noise flag, consumed by policy agent's pi1 training script.

---

## iter_003 — ball_obs_spec.py: ObsTerm-compatible perception interface  (2026-04-07T01:00:00Z)
Hypothesis: Creating ObsTerm-compatible wrapper functions with an explicit noise_cfg parameter will let the policy agent swap oracle→noisy observations without changing env_cfg wiring, just the noise_cfg dataclass.
Change:     Created `perception/ball_obs_spec.py` with `ball_pos_perceived()` and `ball_vel_perceived()` — drop-in replacements for `mdp.ball_pos_in_paddle_frame` / `mdp.ball_vel_in_paddle_frame`. Three modes: "oracle" (GT passthrough), "d435i" (structured depth-dependent noise), "ekf" (NotImplementedError stub). D435i noise model: XY σ=2mm base, Z σ=3mm+2mm/m, 2% dropout, velocity noise derived from finite-differenced position at 30Hz. Updated `__init__.py` to export public API (`BallObsNoiseCfg`, `D435iNoiseParams`, `ball_pos_perceived`, `ball_vel_perceived`).
Command:    AST parse + dataclass construction test (isaaclab not available outside sim env)
Result:     Module parses cleanly, dataclasses construct with defaults and custom overrides. Full sim integration test deferred to camera mount iteration (task 5). Key design: `BallObsNoiseCfg(mode="oracle")` is the default → zero behavior change when policy agent swaps in these functions.
Decision:   Next iter: task 4 — survey docs/sim_to_real_plan.md for camera mount assumptions and update for rear-paddle-mounted 45° upward D435i orientation.

---

## iter_004 — sim_to_real_plan.md D435i update  (2026-04-07T02:00:00Z)
Hypothesis: The sim_to_real_plan.md still references a generic "global shutter USB cam" with monocular depth and 120Hz; updating it for D435i will align the hardware preparation track (Track 3), latency budget, comms architecture, and risk assessment with the actual sensor choice.
Change:     13 targeted edits to docs/sim_to_real_plan.md: (1) perception gap table → "noisy D435i depth", (2) latency table → D435i 11ms readout at 90Hz, (3) Track 2 summary → "D435i stereo depth lookup", (4) Track 3A camera row → D435i rear-paddle-mounted 45° upward with mass/USB3 details, (5) added camera-paddle transform row, (6) updated cabling for USB3, (7) added mass → 170g assembly, (8) comms diagram → D435i 30-90Hz, (9) compute stack → librealsense2 SDK, (10) DR base mass note → D435i ~90g, (11) calendar → D435i-specific tasks, (12) risk: motion blur → D435i IR global shutter, (13) risk: detection → D435i stereo depth advantage. Added 45° mount rationale paragraph with FoV geometry analysis.
Command:    Read + Edit (no code, doc-only iteration)
Result:     sim_to_real_plan.md now consistent with perception_roadmap.md on D435i choice. Key additions: 45° mount geometry rationale (FoV spans 0m to >1m), camera-paddle rigid transform spec (~50mm behind, ~30mm below, 45° Y-rotation), assembly mass estimate (170g vs prior ~25g mono), librealsense2 in software stack.
Decision:   Next iter: task 5 — mount a simulated D435i in the Isaac Lab scene env_cfg and take a debug snapshot. This requires reading the existing scene setup and adding a TiledCamera sensor (per CAMERA_CHOICE.md decision).

---

## iter_005 — Mount simulated D435i TiledCamera in PLAY scene  (2026-04-07T19:15:00Z)
Hypothesis: Adding a TiledCameraCfg to a DEBUG scene subclass (used only in PLAY config) will let us render D435i-like RGB+depth frames without affecting training performance (training uses 12288 envs with no camera).
Change:     (1) Added `TiledCameraCfg` import and `BallJuggleHierSceneCfg_DEBUG` class to `ball_juggle_hier_env_cfg.py` — inherits full scene, adds `d435i` TiledCamera at `{ENV_REGEX_NS}/Robot/base/D435i` (30Hz, 640×480, 86° HFOV matching D435i depth module, 45° upward pitch, mounted 5cm behind + 8cm above trunk). (2) Updated `BallJuggleHierEnvCfg_PLAY` to use the DEBUG scene. (3) Created `scripts/perception/debug_d435i_capture.py` — standalone script that creates the PLAY env, steps 50-100 steps, then saves RGB + depth PNG + raw .npy from env 0. (4) Created `perception/debug/` directory for output frames.
Command:    AST parse test (OK). GPU smoke test queued but blocked by policy agent's 500-iter training run — GPU lock contention.
Result:     Code parses cleanly. Camera config: focal_length=11.24cm, horizontal_aperture=20.955cm → 86° HFOV. Quaternion (0.9239, -0.3827, 0, 0) = 45° pitch up in ROS convention. Visual verification deferred to next iter when GPU frees.
Decision:   Next iter: run the debug_d435i_capture.py smoke test (GPU should be free). If it works, save frame_000.png and move to task 6 (Forrai et al. reference survey). If TiledCamera fails at runtime, debug the sensor config.

---

## iter_006 — REFERENCES.md + fix gym.make + Forrai skip  (2026-04-07T19:45:00Z)
Hypothesis: Writing perception/REFERENCES.md documenting the ETH noise-injection approach (Ma et al. 2025) and related work will clarify architectural decisions for the EKF/noise pipeline and satisfy task 6.
Change:     (1) Fixed debug_d435i_capture.py — `gym.make()` needs `cfg=env_cfg` with an instantiated config object, not `cfg_entry_point` string. Now directly imports and instantiates `BallJuggleHierEnvCfg_PLAY()`. (2) Created `perception/REFERENCES.md` summarising Ma et al. (noise injection, primary ref), Ji et al. (DribbleBot, ball manipulation), Portela et al. (sigma curriculum), noted Forrai modality differences (event camera vs frame-based D435i). (3) Updated fix_plan with next 3 tasks: ball_ekf.py, noise_model.py, integration test. (4) Forrai paper not in repo — task 6 marked complete with documented skip reasoning.
Command:    No GPU workload (policy agent holding lock for 500-iter training). Debug capture still queued in background.
Result:     REFERENCES.md committed. gym.make fix committed. Debug capture smoke test deferred — background task will complete when GPU frees.
Decision:   Next iter: check if debug capture completed (background task). If not, begin ball_ekf.py implementation (pure PyTorch, no GPU needed for writing).

---

## iter_007 — ball_ekf.py: batched 6-state Kalman filter  (2026-04-07T20:00:00Z)
Hypothesis: A 6-state EKF with ballistic+drag dynamics can track ball position and velocity from noisy D435i measurements, batched across all envs on GPU.
Change:     Created `perception/ball_ekf.py` — `BallEKF` class with predict (ballistic+quadratic drag), update (measurement with dropout mask), step (combined), and reset (per-env). Uses Joseph-form covariance update for numerical stability. Drag coefficient computed from ping-pong ball aerodynamics (Cd=0.4, r=20mm, m=2.7g → c=0.112). Updated `__init__.py` to export `BallEKF` and `BallEKFConfig`.
Command:    CPU unit tests via direct module import (no Isaac Lab deps required).
Result:     All 5 tests pass: (1) gravity prediction z=-0.115 after 0.2s (analytic: -0.116), (2) measurement update blends correctly, (3) dropout → predict-only, (4) stationary convergence holds XY within 2cm, (5) free-fall tracking error 3.4cm at 1.0s. Debug camera capture still waiting for GPU lock (policy 500-iter training).
Decision:   Next iter: implement noise_model.py (structured D435i noise sampling as a standalone module) OR wire the EKF into ball_obs_spec.py's "ekf" mode. The latter is more impactful — it completes the full GT→noise→EKF→obs pipeline.

---

## iter_008 — D435i TiledCamera verified: RGB+depth frames captured  (2026-04-07T20:30:00Z)
Hypothesis: Fixing the camera prim path (trunk, not base), scene override (PYTHONPATH for worktree isolation), and pi2 checkpoint auto-detection will produce valid D435i frames.
Change:     (1) Fixed prim path: `Robot/base/D435i` → `Robot/trunk/D435i` (Go1 root link is `trunk`). (2) Fixed scene override: `@configclass` field overrides don't propagate in subclasses — moved to `__post_init__` assignment. (3) Added `sys.path.insert` for worktree isolation (editable install points to policy agent's worktree). (4) Added pi2 checkpoint auto-detection from `~/Research/QuadruJuggle/logs/`. (5) Fixed depth save bug (undefined `d_min` when no valid pixels).
Command:    `uv run --active python scripts/perception/debug_d435i_capture.py --task Isaac-BallJuggleHier-Go1-Play-v0 --num_envs 1 --headless --enable_cameras --steps 20`
Result:     Camera successfully instantiated in scene (`d435i` in entity list). RGB (640×480) and depth frames saved to `perception/debug/`. Images are grey/black because: (a) headless mode shows only dome light in upward view, (b) pi2 step fails (obs dim mismatch 41 vs 53 — the TiledCamera adds obs to pi2 that the frozen checkpoint doesn't expect). Key finding: adding a camera sensor changes the scene entity count, which may affect pi2 obs dimension if not isolated properly.
Decision:   Next iter: wire EKF into ball_obs_spec.py "ekf" mode — this is the main deliverable. The camera is a debug tool; the EKF pipeline is what pi1 training needs.

---

## iter_009 — noise_model.py + EKF mode wired into ball_obs_spec.py  (2026-04-08T03:45:00Z)
Hypothesis: A stateful D435iNoiseModel (latency buffer, hold-last-value dropout) combined with BallEKF, exposed via PerceptionPipeline class in ball_obs_spec.py's "ekf" mode, completes the GT→noise→EKF→obs pipeline.
Change:     (1) Created `perception/noise_model.py` — `D435iNoiseModel` class with depth-dependent noise, proper hold-last-value on dropout (not GT passthrough), configurable latency buffer (deque). (2) Added `PerceptionPipeline` class to `ball_obs_spec.py` — wraps D435iNoiseModel + BallEKF, lazy-initialized on `env._perception_pipeline`, idempotent step dedup via `env.common_step_counter`. (3) Implemented "ekf" mode in both `ball_pos_perceived` and `ball_vel_perceived`. (4) Added `reset_perception_pipeline()` EventTerm-compatible function for env reset. (5) Updated `__init__.py` exports.
Command:    AST parse (OK). CPU unit tests for noise_model: 4/4 pass (sampling stats, dropout hold, latency buffer, reset).
Result:     Full pipeline code complete. The "ekf" mode in ball_obs_spec.py now chains: GT pos → D435iNoiseModel.sample() → (noisy_pos, detected) → BallEKF.step() → filtered pos/vel. Drop-in usage: change `BallObsNoiseCfg(mode="oracle")` to `BallObsNoiseCfg(mode="ekf")` and add `reset_perception_pipeline` EventTerm. No sim test yet (requires GPU + Isaac Lab env).
Decision:   Next iter: integration test — modify ball_juggle_hier env_cfg to use mode="ekf", run 50-iter smoke test with GPU, verify mean_episode_length is reasonable (some degradation expected vs oracle).

---

## iter_010 — EKF integration test: full pipeline verified in Isaac Lab  (2026-04-08T03:50:00Z)
Hypothesis: The EKF perception pipeline (GT→D435iNoiseModel→BallEKF→obs) integrates correctly with ball_juggle_hier env, producing valid observations that allow PPO training to proceed without crashes.
Change:     Created `scripts/perception/test_ekf_integration.py` — standalone test script that patches env_cfg obs terms to use `ball_pos_perceived`/`ball_vel_perceived` with `BallObsNoiseCfg(mode="ekf")`, adds `reset_perception_pipeline` EventTerm, and runs PPO training with metric collection. Fixed `--headless` arg conflict with AppLauncher, and added sys.path worktree isolation (editable install points to policy agent's worktree).
Command:    (1) Smoke test: 4 envs, 2 iters, 30s timeout — PASS. (2) Full test: 4096 envs, 50 iters — PASS.
Result:     EKF mode works end-to-end in Isaac Lab. 50-iter metrics: mean_ep_len grew from 21→123 (learning curve normal for Stage A). Termination breakdown: 99.95% ball_below, 0.05% ball_off, 0% timeout. Rewards accumulating normally (alive=0.094, apex=0.632). No NaN, no dimension mismatch, no EKF divergence. Oracle comparison deferred — GPU queue blocked by policy agent's 500-iter training runs (2 jobs ahead in lock queue).
Decision:   Next iter: handoff to policy agent — write PERCEPTION_HANDOFF.md documenting how to enable EKF mode (config diff, reset event, sys.path note). Then pursue oracle vs EKF comparison when GPU frees.

---

## iter_011 — PERCEPTION_HANDOFF.md for policy agent  (2026-04-08T04:30:00Z)
Hypothesis: A clear handoff document will let the policy agent integrate EKF mode without needing to reverse-engineer the perception pipeline code.
Change:     Created `perception/PERCEPTION_HANDOFF.md` with: (1) runtime patching example code, (2) env_cfg.py integration example, (3) worktree isolation sys.path note, (4) full parameter table (D435iNoiseModelCfg, BallEKFConfig), (5) integration test results summary, (6) specific changes needed in policy agent's train_juggle_hier.py (add "ekf" to --noise-mode choices, add reset_perception_pipeline EventTerm). Processed inbox from Daniel ("you still running?").
Command:    No GPU workload — documentation-only iteration.
Result:     PERCEPTION_HANDOFF.md committed. Key finding from reviewing policy agent's code: their train_juggle_hier.py only supports oracle/d435i, and doesn't add the reset event for EKF mode. Both gaps documented in handoff.
Decision:   Next iter: oracle vs EKF comparison test. Run test_ekf_integration.py with --perception-mode oracle and --perception-mode ekf back-to-back (50 iters each, 4096 envs) to quantify the EKF filtering benefit vs raw d435i noise. Then begin EKF parameter tuning.

---

## iter_012 — Perception diagnostics + comparison script  (2026-04-07T21:00:00Z)
Hypothesis: Adding per-step estimation error tracking (EKF vs raw noise vs GT) to PerceptionPipeline will quantify the EKF filtering benefit and guide parameter tuning.
Change:     (1) Added `_PerceptionDiagnostics` class to `ball_obs_spec.py` — tracks running pos/vel RMSE for both EKF-filtered and raw noisy measurements. Computes `ekf_improvement_pct` (RMSE reduction). Enabled via `env._perception_diagnostics_enabled = True` before pipeline creation. (2) Updated `PerceptionPipeline.step()` to accept optional `gt_vel_b` for velocity error tracking. (3) Added `PerceptionPipeline.diagnostics` property — returns summary dict and resets accumulators. (4) Created `scripts/perception/compare_perception_modes.py` — runs oracle/d435i/ekf back-to-back with diagnostic logging every 10 iterations, saves comparison table + per-mode diagnostic JSON.
Command:    AST parse (both files OK). No GPU workload — policy agent holds lock.
Result:     Code ready for comparison test. Diagnostics collect: pos_rmse_ekf_mm, pos_rmse_raw_mm, vel_rmse_ekf_mps, detection_rate, ekf_improvement_pct. Comparison script will produce a table like the one policy agent generated (iter_005/006) but with the EKF column added.
Decision:   Next iter: run compare_perception_modes.py when GPU frees. If still blocked, begin noise curriculum support (parameterize noise scaling so policy can ramp d435i noise across stages — this is what their noise_curriculum_plan.md calls for).

---

## iter_013 — noise_scale curriculum support  (2026-04-08T05:15:00Z)
Hypothesis: Adding a `noise_scale` multiplier to BallObsNoiseCfg will let the policy agent ramp D435i noise gradually across curriculum stages (0.25→0.50→0.75→1.0), as specified in their noise_curriculum_plan.md Section 4.3.
Change:     (1) Added `noise_scale: float = 1.0` field to `BallObsNoiseCfg`. (2) Created `_scaled_d435i_params()` and `_scaled_noise_model_cfg()` helpers that multiply sigma/dropout by scale (latency unchanged). (3) Updated `_get_or_create_pipeline()` to apply scale at creation. (4) Updated d435i inline mode (`ball_pos_perceived`/`ball_vel_perceived`) to apply scale via `_scaled_d435i_params`. (5) Added `PerceptionPipeline.update_noise_scale()` for live curriculum updates (recovers unscaled base on first call). (6) Added `update_perception_noise_scale(env, scale)` standalone function for policy agent's curriculum callback. (7) Exported from `__init__.py`.
Command:    6 CPU unit tests: scaling at 0.5/0.25/1.0/0.0, identity at 1.0, round-trip update_noise_scale, scale=0.0 zeroes noise.
Result:     All 6 tests PASS. Key API for policy agent: `BallObsNoiseCfg(mode="d435i", noise_scale=0.25)` for static config, or `update_perception_noise_scale(env, 0.5)` for runtime curriculum transitions. Matches noise_curriculum_plan.md stages D-F design exactly.
Decision:   Next iter: run compare_perception_modes.py if GPU frees. Otherwise, update PERCEPTION_HANDOFF.md with noise_scale API docs for policy agent.

---

## iter_014 — Body-frame gravity in EKF + HANDOFF noise_scale docs  (2026-04-08T06:00:00Z)
Hypothesis: The EKF's hardcoded gravity=[0,0,-9.81] is wrong in the body frame when the robot tilts; at 5° tilt this introduces ~0.86 m/s² lateral prediction error. Passing the actual `projected_gravity_b * 9.81` will improve EKF accuracy during active balancing.
Change:     (1) `ball_ekf.py`: `predict()` and `step()` now accept optional `gravity_b: Tensor(N,3)` — backward-compatible (None→default downward). (2) `ball_obs_spec.py`: `PerceptionPipeline.step()` accepts `gravity_b`; `ball_pos_perceived` and `ball_vel_perceived` read `robot.data.projected_gravity_b * 9.81` and pass it through in EKF mode. (3) Updated `PERCEPTION_HANDOFF.md` with full `noise_scale` curriculum API docs (static config table, runtime `update_perception_noise_scale()` example, EKF+noise_scale interaction).
Command:    3 CPU unit tests: (1) backward compat predict without gravity_b, (2) tilted 10° gravity produces lateral displacement, (3) step() with explicit gravity. AST parse both files.
Result:     All 3 tests PASS. Isaac Lab `projected_gravity_b` confirmed as unit vector in body frame (articulation_data.py:790). GPU smoke test for compare_perception_modes.py still blocked behind policy agent's training run (~10 min elapsed when checked).
Decision:   Next iter: run the comparison test (oracle vs d435i vs ekf, 50 iters each). GPU should be free by then. The body-frame gravity fix will be validated as part of the EKF comparison.

---

## iter_015 — subprocess isolation for compare script + lit-review subagent spawned  (2026-04-08T06:38:00Z)
Hypothesis: The compare_perception_modes.py script hangs when switching modes because Isaac Lab can't cleanly recreate envs in a single process. Running each mode in a separate subprocess will fix this.
Change:     (1) Rewrote compare_perception_modes.py to use subprocess isolation — each mode runs in a separate `python ... --single-mode <mode>` process, results collected via JSON files. The orchestrator spawns subprocesses sequentially and combines results. (2) Killed stuck PID 134896 (old compare script hung after oracle completed, ~61 min wasted). (3) Spawned `lit-review` subagent per Daniel's request — sonnet model, parent=perception, task: survey perception-for-manipulation papers and critique our D435i+EKF approach. (4) Seeded lit-review fix_plan.md and INBOX with task description. (5) Notified policy agent via INBOX about c3r spawn usage (Daniel's request). (6) Processed 3 INBOX messages from Daniel (spawn lit-review, use c3r spawn, context status).
Command:    /home/daniel-grant/Research/c3r/c3r spawn /home/daniel-grant/Research/QuadruJuggle lit-review generic "..." --model sonnet --parent perception
Result:     lit-review spawned (thread 1491325907695177789), currently in error state from iter 0 (API 529 overload) — agent loop will auto-retry. Compare script AST-parses correctly. GPU still occupied by policy agent's 1200-iter d435i training (PID 161679, 12288 envs).
Decision:   Next iter: if GPU is free, run the subprocess-based comparison test. If lit-review has produced output, review it. If GPU still busy, look into whether we can run a reduced comparison (fewer envs, fewer iters) or work on EKF parameter analysis offline.

---

## iter_016 — EKF bug fix + oracle vs d435i vs ekf comparison  (2026-04-08T07:15:00Z)
Hypothesis: Fixing the EKF vel-view mutation bug and running a 3-mode comparison (oracle/d435i/ekf, 2048 envs × 50 iters) will quantify the perception pipeline's impact on early training.
Change:     (1) Fixed critical EKF covariance bug: `vel = self._x[:, 3:]` is a view that mutates after state update, causing `v_hat = vel/speed` to use new velocity with old speed (1e-8), inflating P[5,5] to 74M → singular S → NaN → PPO crash. Fix: `vel = self._x[:, 3:].clone()` + compute F BEFORE state mutation. (2) Replaced Joseph-form covariance update with standard form `P = (I-KH)@P` + symmetrization + eps regularization (simpler, avoids PSD violation from diagonal clamping). (3) Replaced `linalg.inv(S)` with `linalg.solve` + 1e-8 diagonal regularization. (4) Added NaN/Inf clamping on EKF output observations (safety net). (5) Fixed compare_perception_modes.py metric capture: reads `locs["rewbuffer"]`/`locs["lenbuffer"]` instead of parsing ep_infos keys. (6) Added state clamping: pos ±5m, vel ±20m/s.
Command:    `gpu_lock.sh uv run --active python scripts/perception/compare_perception_modes.py --pi2-checkpoint .../2026-03-12_09-04-32/model_best.pt --num_envs 2048 --max_iterations 50 --headless --modes oracle d435i ekf`
Result:     All three modes complete (no crashes). 50-iter comparison (2048 envs):
  | Mode   | ep_len_final10 | reward_final10 | timeout% |
  |--------|----------------|----------------|----------|
  | oracle | 294.0          | 22.0           | 0.0%     |
  | d435i  | 317.4          | 20.1           | 0.3%     |
  | ekf    | 278.6          | 19.3           | 0.0%     |
  Oracle leads on reward (+12% over EKF). D435i raw noise surprisingly leads on ep_len (noise as regularization?). Gaps are modest at 50 iters — real differentiation expected at 500+ iters.
Decision:   Core perception pipeline is feature-complete and stable. Next: update fix_plan (mark comparison done), check lit-review subagent, then pursue EKF parameter tuning or longer training comparison.

---

## iter_017 — EKF Q/R tuning (CWNA-derived) + time-varying R + ANEES diagnostic  (2026-04-08T10:30:00Z)
Hypothesis: CWNA-derived Q/R values (lit-review analysis) will make the EKF rely more on ballistic dynamics during camera gaps, and time-varying R will correctly weight close vs far measurements. ANEES provides automatic consistency checking.
Change:     (1) Updated BallEKFConfig defaults: q_pos 0.01→0.003, q_vel 1.0→0.15, r_xy 0.003→0.002, r_z 0.005→0.004+0.002/m. (2) Added `r_z_per_metre=0.002` and `adaptive_r=True` fields. (3) Made `update()` compute per-env R when adaptive_r=True: σ_z = r_z + r_z_per_metre × |z_est|. (4) Added ANEES/NIS diagnostic: `NIS_k = y^T S^{-1} y` computed during update for detected envs; exposed via `mean_nis` property and `reset_nis()` method. Target band: [0.35, 7.81] for 3D meas (95% χ²(3)). (5) Updated PerceptionPipeline.diagnostics to include mean_nis. (6) Updated PERCEPTION_HANDOFF.md with new param table. (7) Copied lit_review_ekf_tuning.md to this branch. (8) Killed lit-review subagent (attempted — c3r binary path issue).
Command:    6 CPU unit tests (new defaults, gravity prediction, time-varying R, NIS diagnostics, adaptive_r=False fallback, NIS counting). GPU comparison blocked by modal process holding lock.
Result:     All 6 tests PASS. GPU comparison deferred — another agent holds GPU lock for modal cloud run (no local GPU usage, just lock contention). Comparison to be run next iteration.
Decision:   Next iter: run 3-mode comparison (oracle/d435i/ekf, 2048 envs × 50 iters) with tuned Q/R. Compare mean_nis to [0.35, 7.81] consistency band. If NIS is outside band, adjust Q/R further.

---

## iter_018 — Fix pi2 obs dim (41→53) + 3-mode comparison with tuned EKF  (2026-04-08T08:00:00Z)
Hypothesis: The compare script crashes because the pi2 checkpoint (2026-03-12_17-16-01) was trained with 53D obs (including last_action) but action_term.py only provides 41D. Fixing this and running with tuned EKF params will show EKF filtering benefit.
Change:     (1) Fixed `action_term.py`: auto-detect pi2 input dimension from checkpoint first layer, conditionally include `_last_pi2_actions` (12D) when checkpoint expects 53D. Added `_last_pi2_actions` buffer, updated after each forward pass. (2) Added NIS logging to compare_perception_modes.py DIAG output. (3) Cleaned stale result JSONs before running.
Command:    `uv run --active python scripts/perception/compare_perception_modes.py --pi2-checkpoint .../2026-03-12_17-16-01/model_best.pt --num_envs 2048 --max_iterations 50 --headless --modes oracle d435i ekf`
Result:     All 3 modes complete (no crashes). 50-iter comparison (2048 envs):
  | Mode   | ep_len_final10 | reward_final10 | timeout% |
  |--------|----------------|----------------|----------|
  | oracle | 337.9          | 13.7           | 1.2%     |
  | d435i  | 225.1          | 10.5           | 1.4%     |
  | ekf    | 208.6          | 7.6            | 0.4%     |
  Oracle leads. EKF trails d435i (raw noise) by ~28% on reward — tuned Q/R may be over-smoothing early training. Diagnostics (NIS) not captured in subprocess output — needs piping fix. Note: iter_016 results are INVALID (ran with old 41D action_term against 53D checkpoint — unclear how it produced results; likely used a different checkpoint).
Decision:   The pi2 obs dim bug is the key fix this iteration. Next: (1) Run NIS diagnostic standalone to validate tuned Q/R values are in [0.35, 7.81] band. (2) If NIS is too low (over-smoothing), increase q_vel back toward 0.5. (3) Consider longer comparison (200+ iters) to see if EKF converges.

---

## iter_019 — NIS diagnostic script + compare fix  (2026-04-08T13:05:00Z)
Hypothesis: A standalone NIS diagnostic script (no training loop, just env stepping with random actions) will validate EKF Q/R values faster than the full compare script, and the compare script's diagnostic capture was broken by env wrapper indirection.
Change:     (1) Created `scripts/perception/nis_diagnostic.py` — runs env with EKF mode for N steps, logs NIS/RMSE/detection_rate every `log_interval` steps, classifies NIS against 95% χ²(3) band [0.35, 7.81], supports `--q_vel`/`--q_pos`/`--r_xy`/`--r_z` overrides for quick parameter sweeps. (2) Fixed `compare_perception_modes.py`: changed `env.unwrapped._perception_pipeline` to `base_env` (direct reference saved after `gym.make`), added `flush=True` to diagnostic prints for subprocess visibility, added warning print when pipeline is None on first diagnostic check.
Command:    AST parse both files (OK). Smoke test blocked — GPU held by policy agent's 1500-iter d435i training (PID 223622).
Result:     Both scripts pass AST validation. GPU unavailable — NIS diagnostic deferred to next iteration. Compare script fix addresses the missing diagnostic data from iter_018.
Decision:   Next iter: run NIS diagnostic (500 steps, 2048 envs, ~2 min GPU time). If GPU still busy, run with smaller num_envs (256) as fallback. Based on NIS result, adjust q_vel if needed.

---

## iter_020 — q_vel 0.15→0.30 (CWNA fix) + NIS sweep script  (2026-04-08T21:30:00Z)
Hypothesis: q_vel=0.15 is 7× below CWNA prescription (lit_review_ekf_lag_vs_raw_noise.md), causing 24cm position lag at 2 m/s and explaining why EKF (7.6) trails raw d435i (10.5) by 28% in iter_018. Increasing to q_vel=0.30 (CWNA midpoint for Stage E–F) should halve the lag.
Change:     (1) Updated BallEKFConfig.q_vel from 0.15 to 0.30 (comment updated with CWNA justification). (2) Created `scripts/perception/nis_sweep.py` — runs nis_diagnostic.py for q_vel=[0.15,0.25,0.30,0.35,0.50] sequentially, produces comparison table, saves JSON results. (3) Copied lit_review_ekf_lag_vs_raw_noise.md from lit-review branch.
Command:    AST parse both files (OK). GPU locked by policy agent (PID 223622, 12288 envs × 1500 iters, ~10 min elapsed). NIS sweep deferred.
Result:     Code changes ready; validation blocked on GPU. Lit-review iter_009 confirms diagnosis: D'Ambrosio 2024 (DeepMind TT) found latency > noise in sensitivity; CWNA prescribes q_vel ≈ 0.40; our 0.30 is conservative midpoint. Also flagged "train without EKF, deploy with EKF" as architecture option if EKF still trails raw noise after tuning.
Decision:   Next iter: run NIS sweep (256 envs × 300 steps × 5 q_vel values, ~10 min GPU total). Pick q_vel with mean NIS closest to 3.0. Then re-run 3-mode comparison to see if reward gap closes.

---

## iter_021 — NIS diagnostic reveals EKF is 30× worse than raw noise  (2026-04-08T22:05:00Z)
Hypothesis: Running NIS diagnostic with updated q_vel=0.30 (256 envs × 100 steps, random actions) will show NIS in the [0.35, 7.81] consistency band, validating the CWNA-derived tuning.
Change:     (1) Fixed stdout buffering in nis_diagnostic.py (all print→_print with flush=True). (2) Fixed diagnostics initialization: pipeline was created during gym.make() BEFORE _perception_diagnostics_enabled flag was set; fix: set flag then set pipeline=None to force recreation. (3) Removed debug prints from ball_obs_spec.py.
Command:    `PYTHONUNBUFFERED=1 uv run --active python scripts/perception/nis_diagnostic.py --pi2-checkpoint .../2026-03-12_17-16-01/model_best.pt --num_envs 256 --steps 100 --log_interval 10 --headless`
Result:     **NIS = 966** (target 3.0) — 320× above ideal. EKF RMSE = 130mm vs raw noise RMSE = 4.4mm — EKF is **30× worse**. NIS grows from 12 at step 10 to ~1200 at step 40 and plateaus there. Detection rate = 98% (healthy). Root cause: EKF uses ballistic dynamics (gravity+drag) in body frame, but with random/untrained actions, the robot's own acceleration creates massive unmodeled pseudo-forces in the body frame. The EKF's prediction is catastrophically wrong → covariance collapses → innovations explode.
  | Step | NIS    | EKF mm | Raw mm | Improvement |
  |------|--------|--------|--------|-------------|
  |   10 |   12.1 |  23.1  |  19.7  |    -17.6%   |
  |   50 | 1262.0 | 136.1  |   4.3  |  -3051.9%   |
  |  100 | 1155.2 | 129.9  |   4.4  |  -2852.4%   |
Decision:   **EKF is conclusively harmful during training.** Adopt "train without EKF, deploy with EKF" pattern per lit-review recommendation. Next iter: (1) Update PERCEPTION_HANDOFF.md to recommend raw d435i noise for training, EKF for hardware deployment only. (2) Add body-frame acceleration compensation to EKF for deployment use (pass robot body_lin_acc to predict). (3) NIS sweep is moot — the body-frame dynamics mismatch is the bottleneck, not Q/R tuning.

---

## iter_022 — Body-frame acceleration compensation + "no EKF for training" recommendation  (2026-04-08T22:45:00Z)
Hypothesis: Adding robot body-frame acceleration compensation to EKF predict() will fix the pseudo-force problem (NIS=966 in iter_021). The robot's acceleration in body frame creates unmodeled forces on the ball in the EKF's coordinate system; subtracting it from the dynamics model should bring NIS back toward 3.0.
Change:     (1) Added `robot_acc_b` parameter to `BallEKF.predict()` and `step()` — subtracts robot body-frame acceleration from ball dynamics: `a_ball = gravity_body + drag - robot_acc_body`. Backward-compatible (None → no compensation). (2) Added finite-difference acceleration estimation in `PerceptionPipeline.step()`: stores `_prev_robot_vel_b`, computes `(v_now - v_prev) / dt`, clamped to ±50 m/s². Clears velocity buffer on `reset()` to prevent spurious spikes. (3) Wired `robot.data.root_lin_vel_b` through `ball_pos_perceived`/`ball_vel_perceived` → pipeline.step(). (4) Updated PERCEPTION_HANDOFF.md: added "No EKF During Training" section with NIS=966 evidence, recommended d435i mode for training, documented acceleration compensation for deployment. (5) 4 CPU unit tests: backward compat, compensation changes prediction correctly, step() pass-through, zero-acc == no-acc.
Command:    4 CPU unit tests (torch only, no Isaac Lab). AST parse both files.
Result:     All 4 tests PASS. Compensation produces correct physics: 5 m/s² robot acc → 1mm position delta over 0.02s (matches 0.5 * a * dt²). GPU validation blocked (policy agent holds lock). PERCEPTION_HANDOFF.md now clearly states d435i for training, EKF for deployment only.
Decision:   Next iter: run NIS diagnostic with acceleration compensation enabled (256 envs × 100 steps). This is the critical validation — if NIS drops from 966 to near 3.0, the EKF is viable for deployment. If still high, the issue may be rotational pseudo-forces (Coriolis/centrifugal) which need separate modeling.

---

## iter_022b — NIS diagnostic with accel compensation: still broken (rotational pseudo-forces)  (2026-04-08T23:30:00Z)
Hypothesis: Body-frame linear acceleration compensation (robot_acc_b) will reduce NIS from 966 to near 3.0 by removing the dominant pseudo-force term.
Change:     (no code change — used iter_022 code to run GPU diagnostic)
Command:    `gpu_lock.sh uv run --active python scripts/perception/nis_diagnostic.py --pi2-checkpoint .../model_best.pt --num_envs 256 --steps 100 --log_interval 10 --headless`
Result:     **NIS = 1025** (was 966 without compensation). Linear acceleration compensation has negligible effect. EKF RMSE = 130mm vs raw = 4.4mm — identical to iter_021.
  Root cause analysis: the EKF operates in the body (paddle) frame — a ROTATING non-inertial frame. Linear acceleration is only one pseudo-force term. The dominant terms are ROTATIONAL:
  - Coriolis: -2ω×v (at ω=5 rad/s, v=1 m/s → 10 m/s²)
  - Centrifugal: -ω×(ω×r) (at ω=5 rad/s, r=0.1m → 2.5 m/s²)
  - Euler: -dω/dt×r (angular acceleration)
  With random actions, trunk ω can reach 5+ rad/s → rotational pseudo-forces dominate.
  Fundamental issue: ball follows ballistic dynamics in WORLD frame, not body frame. The EKF's dynamics model is structurally wrong for a rotating reference frame.
Decision:   **EKF in body-frame is fundamentally incompatible with training.** Two options for deployment:
  (A) Run EKF in world frame (requires camera-to-world transform from IMU — natural for real robot)
  (B) Add full non-inertial dynamics (Coriolis+centrifugal+Euler) — complex, error-prone
  Option A is cleaner. For training, d435i raw noise is confirmed as the right approach.
  Next iter: either (1) implement world-frame EKF option for deployment, or (2) consider this module done and focus on other perception tasks (real camera integration, detection pipeline).

---

## iter_023 — World-frame EKF implementation  (2026-04-08T23:55:00Z)
Hypothesis: Running EKF in world frame (where ballistic dynamics are correct) will eliminate the pseudo-force problem that caused NIS=966/1025 in body-frame mode. The lit-review agent (iter_016) identified the root cause as coordinate frame mismatch, not body-frame acceleration — Coriolis/centrifugal/Euler forces dominate in the rotating body frame. World-frame EKF is also the natural architecture for real hardware (camera + IMU → world transform).
Change:     (1) Added `world_frame=True` option to BallObsNoiseCfg. (2) Updated PerceptionPipeline to transform body-frame measurements → world frame before EKF update (using robot_quat_w + robot_pos_w + paddle_offset_b), and transform world-frame EKF outputs → body frame for policy consumption. (3) Added _body_to_world_pos/vel and _world_to_body_pos/vel helper methods. (4) Reset transforms init pos/vel to world frame when world_frame=True. (5) Updated ball_pos_perceived/ball_vel_perceived to pass robot pose in world-frame mode. (6) Added `--world-frame` flag to nis_diagnostic.py. (7) Updated PERCEPTION_HANDOFF.md with world-frame EKF docs. (8) 5 CPU unit tests: round-trip transform, stationary ball, tilted robot, backward compat, reset.
Command:    `python scripts/perception/test_world_frame_ekf.py` — 5/5 pass. GPU NIS diagnostic queued (blocked by policy agent training, PID 311640).
Result:     All 5 CPU tests PASS. Key results: body→world→body round-trip error = 1.86e-8, tilted robot (15° pitch) error = 22mm, backward compat OK, reset initialises world coords correctly. GPU validation deferred — policy agent holding lock (12288 env × 1500 iter training).
Decision:   Next iter: run `nis_diagnostic.py --world-frame` (256 envs × 100 steps) when GPU available. If NIS drops from 966→near 3.0, world-frame EKF is validated for deployment. Then run 3-mode comparison. If GPU still locked, run with smaller envs or wait.

---

## iter_024 — World-frame EKF NIS diagnostic: contact forces are the root cause, not coordinate frames  (2026-04-08T10:00:00Z)
Hypothesis: World-frame EKF (iter_023) will produce NIS ≈ 3.0 by eliminating pseudo-forces from rotating body frame.
Change:     (1) Ran NIS diagnostic with `--world-frame` flag (256 envs × 100 steps). (2) When NIS=970 (same as body-frame), ran q_vel sweep: 1.0, 3.0, 5.0, 10.0. (3) Updated q_vel default from 0.30 to 7.0 in BallEKFConfig. (4) Updated PERCEPTION_HANDOFF.md with root-cause analysis and NIS sweep table.
Command:    `gpu_lock.sh uv run --active python scripts/perception/nis_diagnostic.py --pi2-checkpoint .../model_best.pt --num_envs 256 --steps 100 --log_interval 10 --headless --world-frame [--q_vel X]`
Result:     **World-frame NIS = 970** (identical to body-frame 966). Root cause is NOT coordinate frames — it's **unmodeled contact normal force** (~9.81 m/s²) during paddle contact. EKF predicts freefall but ball sits on paddle.
  NIS sweep results:
  | q_vel | NIS | EKF mm | Raw mm | In band? |
  |-------|------|--------|--------|----------|
  | 0.30 | 970 | 131 | 4.4 | No |
  | 1.0 | 95 | 33 | 4.3 | No |
  | 3.0 | 13.5 | 10 | 4.3 | No |
  | 5.0 | 6.3 | 6.2 | 4.3 | Marginal |
  | 10.0 | 2.6 | 4.5 | 4.4 | Yes |
  At q_vel ≥ 5.0, EKF is consistent (NIS in band) but provides NO position improvement over raw D435i. EKF value at deployment: velocity estimation + dropout prediction only.
Decision:   Set q_vel=7.0 as default (NIS≈3.5, honest about contact uncertainty). Perception pipeline now feature-complete for sim training (d435i mode). Next iter: declare feature-complete and plan real hardware integration (D435i driver, YOLO detection, camera calibration).

---

## iter_025 — Feature-complete declaration + noise model calibration + hardware pipeline spec  (2026-04-08T11:00:00Z)
Hypothesis: The sim perception pipeline is feature-complete for training; the remaining high-impact sim-side improvement is calibrating the noise model to match real D435i characteristics (lit-review found sigma_z_per_metre 2.5× too low and dropout_prob 5× too low).
Change:     (1) Created `docs/hardware_pipeline_architecture.md` — comprehensive spec for real-time D435i→YOLO→EKF→pi1 pipeline on Orin NX, synthesizing 26 lit-review iterations. Defines file layout (perception/real/), component interfaces (D435iCamera, BallDetector, CameraCalibrator, RealPerceptionPipeline), timing budget (15ms of 20ms), and deployment sequence. (2) Updated noise model defaults: sigma_z_per_metre 0.002→0.005 (2.5× increase, matches D435i quadratic depth noise curve), dropout_prob 0.02→0.10 (5× increase, matches white ball IR reflectance dropout rate). Updated in both D435iNoiseParams (ball_obs_spec.py) and D435iNoiseModelCfg (noise_model.py). (3) Updated EKF r_z from 0.004→0.0055 and r_z_per_metre from 0.002→0.005 to match new measurement noise. (4) Updated PERCEPTION_HANDOFF.md: marked pipeline feature-complete, documented known sim→real noise gaps, added pointer to hardware spec. (5) Updated fix_plan.md: marked declaration task complete, added Phase 3 hardware integration task queue (9 items).
Command:    AST parse all 3 modified Python files — OK.
Result:     Sim perception pipeline declared feature-complete. Noise model now matches real D435i characteristics per lit-review audit. Hardware pipeline architecture fully specified. Policy agent should note the noise parameter changes — training with updated defaults will produce a more robust policy (higher dropout forces better ball_lost_flag handling, higher depth noise trains depth uncertainty tolerance).
Decision:   Next: create perception/real/ directory stubs per hardware_pipeline_architecture.md. These stubs define the exact interfaces the hardware deployment will use, ensuring the real pipeline is a drop-in replacement for the sim pipeline. Alternatively, if policy agent needs support (e.g., noise curriculum re-tuning with new defaults), prioritise that.
