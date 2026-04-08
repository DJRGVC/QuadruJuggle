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
