# RESEARCH_LOG_ARCHIVE.md

Verbatim archive of compacted iterations. Do not delete.

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
Change:     13 targeted edits to docs/sim_to_real_plan.md: perception gap table, latency table, Track 2/3 summaries, camera mount geometry, cabling, mass, comms diagram, compute stack, DR, calendar, risk items.
Command:    Read + Edit (no code, doc-only iteration)
Result:     sim_to_real_plan.md now consistent with perception_roadmap.md on D435i choice. Key additions: 45° mount geometry rationale (FoV spans 0m to >1m), camera-paddle rigid transform spec (~50mm behind, ~30mm below, 45° Y-rotation), assembly mass estimate (170g vs prior ~25g mono).
Decision:   Next iter: task 5 — mount a simulated D435i in the Isaac Lab scene env_cfg.

---

## iter_005 — Mount simulated D435i TiledCamera in PLAY scene  (2026-04-07T19:15:00Z)
Hypothesis: Adding a TiledCameraCfg to a DEBUG scene subclass (used only in PLAY config) will let us render D435i-like RGB+depth frames without affecting training performance.
Change:     Added `BallJuggleHierSceneCfg_DEBUG` class with `d435i` TiledCamera (30Hz, 640×480, 86° HFOV, 45° upward pitch, mounted 5cm behind + 8cm above trunk). Updated PLAY config. Created `debug_d435i_capture.py`.
Command:    AST parse test (OK). GPU smoke test blocked.
Result:     Code parses. Camera config verified. Visual verification deferred.
Decision:   Run debug capture next iter.

---

## iter_006 — REFERENCES.md + fix gym.make + Forrai skip  (2026-04-07T19:45:00Z)
Hypothesis: Writing perception/REFERENCES.md and fixing gym.make will enable debug camera capture.
Change:     Fixed gym.make (needs cfg=env_cfg instance, not string). Created REFERENCES.md. Forrai paper not in repo — skip documented.
Command:    No GPU.
Result:     REFERENCES.md committed. gym.make fix committed. Debug capture deferred.
Decision:   Begin ball_ekf.py implementation.

---

## iter_007 — ball_ekf.py: batched 6-state Kalman filter  (2026-04-07T20:00:00Z)
Hypothesis: A 6-state EKF with ballistic+drag dynamics can track ball position and velocity from noisy D435i measurements, batched across all envs on GPU.
Change:     Created `perception/ball_ekf.py` — `BallEKF` class with predict (ballistic+quadratic drag), update (measurement with dropout mask), step (combined), and reset (per-env). Joseph-form covariance. Drag from ping-pong aerodynamics (Cd=0.4, r=20mm, m=2.7g → c=0.112).
Command:    5 CPU unit tests.
Result:     5/5 pass. Gravity prediction z=-0.115 after 0.2s (analytic -0.116). Free-fall tracking error 3.4cm at 1.0s.
Decision:   Wire EKF into ball_obs_spec.py.

---

## iter_008 — D435i TiledCamera verified: RGB+depth frames captured  (2026-04-07T20:30:00Z)
Hypothesis: Fixing camera prim path and scene override will produce valid D435i frames.
Change:     Fixed prim path (trunk not base), scene override (__post_init__), sys.path isolation, pi2 checkpoint auto-detection, depth save bug.
Command:    `debug_d435i_capture.py --num_envs 1 --headless --enable_cameras --steps 20`
Result:     Camera instantiated. RGB+depth saved. Grey/black expected in headless. Key finding: TiledCamera changes scene entity count, may affect pi2 obs dimension.
Decision:   Wire EKF into ball_obs_spec.py — the main deliverable.

---

## iter_009 — noise_model.py + EKF mode wired into ball_obs_spec.py  (2026-04-08T03:45:00Z)
Hypothesis: A stateful D435iNoiseModel + BallEKF, exposed via PerceptionPipeline in ball_obs_spec.py "ekf" mode, completes the GT→noise→EKF→obs pipeline.
Change:     Created `noise_model.py` (D435iNoiseModel: depth-dependent noise, hold-last-value dropout, latency buffer). Added `PerceptionPipeline` to ball_obs_spec.py (wraps noise+EKF, lazy init, idempotent step dedup). Implemented "ekf" mode. Added `reset_perception_pipeline()`.
Command:    4 CPU tests for noise_model.
Result:     4/4 pass. Full pipeline code complete: GT→noise→EKF→obs chain works.
Decision:   Integration test with Isaac Lab.

---

## iter_010 — EKF integration test: full pipeline verified in Isaac Lab  (2026-04-08T03:50:00Z)
Hypothesis: EKF pipeline integrates correctly with ball_juggle_hier env.
Change:     Created test_ekf_integration.py. Fixed --headless arg conflict, sys.path isolation.
Command:    Smoke: 4 envs, 2 iters, 30s. Full: 4096 envs, 50 iters.
Result:     Both pass. mean_ep_len 21→123. No NaN, no dim mismatch, no EKF divergence.
Decision:   Write handoff doc, then oracle vs EKF comparison.

---

## iter_011 — PERCEPTION_HANDOFF.md for policy agent  (2026-04-08T04:30:00Z)
Hypothesis: Clear handoff document lets policy agent integrate EKF without reverse-engineering.
Change:     Created PERCEPTION_HANDOFF.md: runtime patching, env_cfg integration, worktree isolation, parameter tables, policy agent changes needed.
Command:    No GPU.
Result:     Handoff committed. Policy agent's train_juggle_hier.py only supports oracle/d435i — gaps documented.
Decision:   Run oracle vs EKF comparison.

---

## iter_012 — Perception diagnostics + comparison script  (2026-04-07T21:00:00Z)
Hypothesis: Per-step error tracking will quantify EKF filtering benefit.
Change:     Added _PerceptionDiagnostics to ball_obs_spec.py. Created compare_perception_modes.py.
Command:    AST parse. GPU blocked.
Result:     Code ready. Diagnostics: pos_rmse, vel_rmse, detection_rate, ekf_improvement_pct.
Decision:   Run comparison when GPU frees.

---

## iter_013 — noise_scale curriculum support  (2026-04-08T05:15:00Z)
Hypothesis: noise_scale multiplier enables gradual noise ramping (0.25→1.0) per policy's noise_curriculum_plan.md.
Change:     Added noise_scale to BallObsNoiseCfg, _scaled helpers, PerceptionPipeline.update_noise_scale(), standalone update function.
Command:    6 CPU unit tests.
Result:     6/6 pass. API: BallObsNoiseCfg(noise_scale=0.25) or update_perception_noise_scale(env, 0.5).
Decision:   Run comparison or update handoff.

---

## iter_014 — Body-frame gravity in EKF + HANDOFF noise_scale docs  (2026-04-08T06:00:00Z)
Hypothesis: Hardcoded gravity=[0,0,-9.81] is wrong when robot tilts (0.86 m/s² error at 5°). projected_gravity_b * 9.81 will fix.
Change:     EKF predict()/step() accept gravity_b. Pipeline passes projected_gravity_b * 9.81. Updated handoff with noise_scale API.
Command:    3 CPU tests.
Result:     3/3 pass. GPU blocked.
Decision:   Run comparison.

---

## iter_015 — subprocess isolation + lit-review subagent spawned  (2026-04-08T06:38:00Z)
Hypothesis: Subprocess isolation will fix compare script mode-switching hang.
Change:     Rewrote compare_perception_modes.py with subprocess isolation. Spawned lit-review subagent. Killed stuck PID 134896.
Command:    c3r spawn lit-review.
Result:     lit-review spawned. Compare script ready. GPU busy.
Decision:   Run comparison next iter.

---

## iter_016 — EKF vel-view bug fix + 3-mode comparison  (2026-04-08T07:15:00Z)
Hypothesis: Fixing EKF covariance bug (vel view mutation) and running comparison will quantify impact.
Change:     Fixed critical bug: vel view mutation→clone. Replaced Joseph-form with standard form + symmetrization. linalg.inv→solve. NaN clamping. State clamping ±5m/±20m/s. Fixed metric capture in compare.
Command:    3-mode comparison: oracle/d435i/ekf, 2048 envs × 50 iters.
Result:     oracle=22.0, d435i=20.1, ekf=19.3 reward. Gaps modest at 50 iters.
Decision:   Pipeline stable. EKF parameter tuning next.

---

## iter_017 — EKF Q/R tuning (CWNA-derived) + ANEES diagnostic  (2026-04-08T10:30:00Z)
Hypothesis: CWNA-derived Q/R + time-varying R + ANEES diagnostic will improve EKF consistency.
Change:     q_pos 0.01→0.003, q_vel 1.0→0.15, r_xy 0.003→0.002, time-varying r_z. Added NIS diagnostic.
Command:    6 CPU tests. GPU blocked.
Result:     6/6 pass.
Decision:   Run comparison with tuned params.

---

## iter_018 — Fix pi2 obs dim (41→53) + tuned comparison  (2026-04-08T08:00:00Z)
Hypothesis: pi2 checkpoint expects 53D obs (with last_action 12D). Fix action_term.py.
Change:     Auto-detect pi2 input dim from checkpoint. Add _last_pi2_actions buffer. NIS logging.
Command:    3-mode comparison, 2048 envs × 50 iters.
Result:     oracle=13.7, d435i=10.5, ekf=7.6. EKF 28% below d435i — over-smoothing (q_vel too low).
Decision:   NIS diagnostic to validate Q/R. Increase q_vel if NIS too low.
