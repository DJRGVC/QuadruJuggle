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

---

## iter_019 — NIS diagnostic script + compare fix  (2026-04-08T13:05:00Z)
Hypothesis: Standalone NIS diagnostic (no training, just env stepping) will validate EKF Q/R faster.
Change:     Created nis_diagnostic.py (NIS/RMSE/detection per step, --q_vel/--q_pos overrides). Fixed compare script base_env reference.
Command:    AST parse. GPU blocked.
Result:     Both scripts validated. GPU deferred.
Decision:   Run NIS diagnostic next iter.

---

## iter_020 — q_vel 0.15→0.30 (CWNA fix) + NIS sweep script  (2026-04-08T21:30:00Z)
Hypothesis: q_vel=0.15 is 7× below CWNA prescription, causing 24cm lag at 2 m/s.
Change:     q_vel 0.15→0.30. Created nis_sweep.py for parameter sweep. Copied lit_review_ekf_lag_vs_raw_noise.md.
Command:    AST parse. GPU blocked.
Result:     Code ready. Lit-review confirms: latency > noise in sensitivity (D'Ambrosio 2024).
Decision:   Run NIS sweep when GPU frees.

---

## iter_021 — NIS diagnostic reveals EKF is 30× worse than raw noise  (2026-04-08T22:05:00Z)
Hypothesis: NIS with q_vel=0.30 will be in [0.35, 7.81] consistency band.
Change:     Fixed stdout buffering, diagnostics init (pipeline recreated after flag set).
Command:    `nis_diagnostic.py --num_envs 256 --steps 100`
Result:     **NIS = 966** (target 3.0). EKF RMSE=130mm vs raw=4.4mm — 30× worse. Root cause: body-frame pseudo-forces from robot motion (not modeled).
Decision:   Adopt "train without EKF, deploy with EKF". Try body-frame accel compensation.

---

## iter_022 — Body-frame accel compensation + "no EKF for training"  (2026-04-08T22:45:00Z)
Hypothesis: Subtracting robot body-frame acceleration from EKF dynamics will fix pseudo-force problem.
Change:     Added robot_acc_b to EKF predict()/step() (finite-diff from root_lin_vel_b, ±50 m/s² clamp). Updated PERCEPTION_HANDOFF.md with "no EKF" recommendation.
Command:    4 CPU tests.
Result:     4/4 pass. GPU validation deferred.
Decision:   Run NIS with accel compensation.

---

## iter_022b — NIS with accel compensation: still broken  (2026-04-08T23:30:00Z)
Hypothesis: Linear accel compensation reduces NIS from 966 to ~3.0.
Change:     No code change — GPU diagnostic run.
Command:    `nis_diagnostic.py --num_envs 256 --steps 100`
Result:     **NIS = 1025** (worse). Linear accel negligible. Dominant: Coriolis (-2ω×v, ~10 m/s² at ω=5 rad/s), centrifugal, Euler forces. Body-frame EKF dynamics structurally wrong.
Decision:   Implement world-frame EKF (option A — cleaner than full non-inertial dynamics).

---

## iter_023 — World-frame EKF implementation  (2026-04-08T23:55:00Z)
Hypothesis: World-frame EKF eliminates pseudo-force problem (ballistic dynamics correct in world frame).
Change:     Added world_frame=True to BallObsNoiseCfg. Body→world measurement transform, world→body output. Helper methods. Reset in world coords. 5 CPU tests.
Command:    `test_world_frame_ekf.py` — 5/5 pass.
Result:     Round-trip error 1.86e-8, tilted robot 22mm error, backward compat OK.
Decision:   GPU NIS diagnostic with --world-frame.

---

## iter_024 — World-frame EKF NIS: contact forces are the root cause  (2026-04-08T10:00:00Z)
Hypothesis: World-frame EKF → NIS ≈ 3.0.
Change:     NIS diagnostic + q_vel sweep {0.30, 1.0, 3.0, 5.0, 10.0}. Updated q_vel default to 7.0.
Command:    `nis_diagnostic.py --world-frame` (256 envs × 100 steps)
Result:     **NIS=970** (same as body-frame). Root cause: unmodeled contact normal force during paddle contact. q_vel sweep: at q_vel≥5.0 NIS in band but EKF = raw noise accuracy. EKF value: velocity estimation + dropout bridging only.
Decision:   q_vel=7.0 default. Declare feature-complete. Shift to real hardware integration.

---

## iter_025 — Feature-complete + noise calibration + hardware spec  (2026-04-08T11:00:00Z)
Hypothesis: Calibrate noise model to real D435i characteristics per lit-review audit.
Change:     Created docs/hardware_pipeline_architecture.md. Updated noise: sigma_z_per_metre 2→5mm/m, dropout_prob 2→10%. Updated EKF r_z/r_z_per_metre. Marked feature-complete.
Command:    AST parse.
Result:     Sim pipeline feature-complete. Hardware spec written. Noise model matches D435i.
Decision:   Create perception/real/ stubs.

---

## iter_026 — Real hardware pipeline stubs  (2026-04-08T12:15:00Z)
Hypothesis: Interface stubs enable parallel component development when hardware arrives.
Change:     Created perception/real/ (6 files): __init__.py, config.py, camera.py, detector.py, calibration.py, pipeline.py. Pure-math methods implemented; hardware methods raise NotImplementedError.
Command:    AST parse all 6.
Result:     All parse. Interfaces match hardware_pipeline_architecture.md.
Decision:   Write unit tests for utility methods.

---

## iter_027 — Unit tests for real hardware utils + from_known_mount  (2026-04-08T13:30:00Z)
Hypothesis: Utility methods (deproject, transform_to_body, median_depth) are correct; from_known_mount is pure RPY→rotation math.
Change:     Implemented from_known_mount (XYZ Euler). Created test_real_utils.py with 17 tests.
Command:    `test_real_utils.py`
Result:     17/17 pass. Rotation matrices orthogonal, det=1.0, pitch mapping correct.
Decision:   MockCamera for integration testing without hardware.

---

## iter_028 — MockCamera + MockDetector for hardware-free integration testing  (2026-04-08T14:45:00Z)
Hypothesis: Mock implementations enable end-to-end testing of real pipeline chain without hardware.
Change:     Created perception/real/mock.py (MockCamera + MockDetector). Created test_mock_pipeline.py (15 tests: camera, detector, full chain including EKF convergence, velocity tracking, dropout).
Command:    `test_mock_pipeline.py`
Result:     **15/15 pass** (0.048s). EKF converges <10mm in 10 measurements. Velocity tracking within bounds. Dropout drift bounded.
Decision:   All remaining fix_plan items hardware-blocked. Check if policy needs support or find other sim-side work.

---

## iter_029 — compaction (summarized iters 001-018)  (2026-04-08T16:00:00Z)
Hypothesis: N/A — compaction iteration.
Change:     Archived iters 001-018 verbatim to RESEARCH_LOG_ARCHIVE.md. Wrote compacted summary
            covering docs/roadmap (001-004), camera+EKF core (005-008), full pipeline+handoff
            (009-011), diagnostics+curriculum (012-015), comparison+tuning (016-018). Kept
            iters 019-028 verbatim. Pruned fix_plan.md.
Command:    No GPU commands.
Result:     Log shrunk from 313->~115 lines. Archive preserved 18 verbatim entries.
Decision:   Next: contact-aware EKF or policy support.

---

## iter_030 — Contact-aware EKF: adaptive process noise during paddle contact  (2026-04-08T17:15:00Z)
Hypothesis: Inflating q_vel during contact (ball Z < 25mm) lets EKF use low q_vel=0.40 for
            free-flight smoothing while trusting measurements during contact (q_vel=50.0),
            fixing the NIS=970 root cause without permanently degrading accuracy.
Change:     BallEKFConfig: q_vel 7.0->0.40 (free-flight CWNA), added contact_aware=True,
            q_vel_contact=50.0, contact_z_threshold=0.025m. predict() builds per-env Q
            matrix based on ball Z position. Added --no-contact-aware flag to nis_diagnostic.py.
Command:    test_contact_aware_ekf.py (7 tests), test_world_frame_ekf.py (6), test_mock_pipeline.py (15).
Result:     **28/28 tests pass.** Free-flight P growth ~6.4e-5 (low noise -> good smoothing).
            Contact P growth >0.5 (high noise -> trust measurements).
Decision:   GPU NIS validation next.

---

## iter_031 — GPU NIS validation: contact-aware EKF 860x improvement  (2026-04-08T18:35:00Z)
Hypothesis: Contact-aware EKF will achieve NIS~3.0 during free-flight.
Change:     No code changes — diagnostic validation.
Command:    `nis_diagnostic.py --num_envs 256 --steps 200 --headless`
Result:     **Contact-aware ON**: NIS=0.78, 10/10 intervals in 95% band. EKF RMSE=5.4mm.
            **Contact-aware OFF**: NIS=671, 0/10 in band. 860x improvement.
Decision:   Ballistic trajectory testing next.

---

## iter_032 — Ballistic trajectory tests (13/13 pass)  (2026-04-08T19:30:00Z)
Hypothesis: EKF with contact-aware mode correctly tracks parabolic arcs across curriculum stages.
Change:     Created test_ballistic_trajectory.py with 13 tests (Stages A/D/G arcs, noisy/dropout,
            contact transitions, multi-bounce, off-axis launch).
Command:    `python scripts/perception/test_ballistic_trajectory.py`
Result:     **13/13 pass.** Stage G pos RMSE <20mm, multi-bounce NIS bounded, 20% dropout <30mm.
Decision:   Latency injection testing next.

---

## iter_033 — Latency injection tests (16/16 pass)  (2026-04-08T20:30:00Z)
Hypothesis: D435i latency buffer correctly delays observations by N steps, EKF degrades gracefully.
Change:     Created test_latency_injection.py with 16 tests (buffer correctness, RMSE bounds,
            monotonic degradation, combined dropout+latency, multi-env independence).
Command:    `python scripts/perception/test_latency_injection.py`
Result:     **16/16 pass.** Pos RMSE: 0-lat <15mm, 1-lat <40mm, 2-lat <80mm, 3-lat <120mm.
            Vel RMSE at 3-frame: <2 m/s (usable). 57 total tests pass.
Decision:   Ahn 2019-calibrated noise model next.

---

## iter_034 — Ahn 2019-calibrated noise model (74/74 pass)  (2026-04-08T22:00:00Z)
Hypothesis: Physics-based noise (sigma_xy proportional to z, sigma_z proportional to z^2, distance-dependent
            dropout) better matches real D435i than constant/linear model.
Change:     Rewrote D435iNoiseParams: sigma_xy=0.0025*z (1mm floor), sigma_z=1mm+0.005*z^2,
            dropout=20%+30%*(1-exp(-(z-0.5)/0.8)). EKF R matched at z=0.5m nominal.
Command:    `pytest scripts/perception/test_*.py`
Result:     **74/74 pass.** Noise tighter at close range, realistically worse at high altitude.
Decision:   GPU NIS re-validation.

---

## iter_035 — GPU NIS re-validation with calibrated noise + kill lit-review  (2026-04-08T14:12:00Z)
Hypothesis: Calibrated noise still achieves in-band NIS.
Change:     No code changes — diagnostic. Killed lit-review subagent per Daniel's request.
Command:    `nis_diagnostic.py --num_envs 256 --steps 200 --log_interval 20 --headless`
Result:     **NIS=1.598, 10/10 in-band.** EKF RMSE=9.1mm, Raw=7.4mm, detection ~80%.
            vs iter_031: NIS 0.78->1.60 (less over-conservative), detection 90->80% (realistic).
Decision:   All Phase 4 sim-side tasks complete. Phase 5 (IMU-aided EKF, spin) nice-to-have.
            Policy agent at iter_014, still on reward shaping. Next: check policy needs or
            propose Phase 5 work.

---

## iter_036 — compaction (summarized iters 019-028)  (2026-04-08T15:00:00Z)
Hypothesis: N/A — compaction iteration.
Change:     Archived iters 019-028 verbatim to RESEARCH_LOG_ARCHIVE.md. Expanded compacted
            summary. Pruned fix_plan.md.
Command:    No GPU commands.
Result:     Log shrunk from 310->~120 lines. Archive now has 28 verbatim entries (001-028).
Decision:   Next: IMU-aided EKF (Phase 5).

---

## iter_037 — IMU-aided EKF: Coriolis + centrifugal corrections (16/16 tests pass)  (2026-04-08T16:30:00Z)
Hypothesis: Adding Coriolis (-2ω×v) and centrifugal (-ω×(ω×r)) pseudo-force corrections
            using robot angular velocity will make body-frame EKF physically correct under
            platform rotation, improving tracking accuracy without requiring world-frame mode.
Change:     Added `robot_ang_vel_b` parameter to `BallEKF.predict()` and `.step()`.
            When provided, computes Coriolis + centrifugal accelerations and adds them to the
            prediction dynamics. Linearised Jacobian F updated. Added `_batch_skew()` helper.
            Pipeline passes `robot.data.root_ang_vel_b` through to EKF in body-frame mode.
Command:    `python scripts/perception/test_imu_aided_ekf.py -v` (16 tests)
            Full suite: 90/90 tests pass.
Result:     **16/16 new tests pass.** Coriolis/centrifugal magnitudes exact. Multi-step tracking
            under 1.5 rad/s rotation: pos RMSE <10mm, vel RMSE <0.5 m/s. 90/90 total.
Decision:   GPU NIS validation of body-frame+IMU vs world-frame next.

---

## iter_038 — 9D spin estimation: Magnus effect EKF extension (25/25 tests, 109/109 total)  (2026-04-08T17:45:00Z)
Hypothesis: Extending EKF to 9D [pos, vel, spin] with Magnus force dynamics allows spin
            estimation from trajectory curvature.
Change:     Added optional `enable_spin` mode to BallEKF. Magnus force: a_M = Cm*(spin×vel).
            Spin decay: exponential with Stokes viscous torque. Contact-aware: q_spin inflated.
            All matrices properly sized as D×D where D=6 or 9. 25 tests across 6 classes.
Command:    `pytest scripts/perception/test_spin_estimation.py -v` → 25/25, full suite 109/109.
Result:     **25/25 pass.** Magnus direction correct. Cm*ω*v = 0.894 m/s² matches. EKF estimates
            spin=40 rad/s from curvature alone. Zero regressions.
Decision:   Wire enable_spin through BallObsNoiseCfg pipeline.

---

## iter_039 — Wire enable_spin + spawn vel-cmd-survey subagent (12/12 new tests, 127/127 total)  (2026-04-08T15:00:00Z)
Hypothesis: BallObsNoiseCfg.enable_spin should propagate to EKF via PerceptionPipeline.
Change:     Added enable_spin to BallObsNoiseCfg + PerceptionPipeline. Created test_pipeline_config.py
            (12 tests). Spawned vel-cmd-survey subagent (sonnet, max 5 iters) per Daniel's request.
Command:    `pytest scripts/perception/test_*.py` → 127/127 pass.
Result:     **12/12 new tests pass.** Config propagation verified. vel-cmd-survey spawned.
Decision:   Monitor vel-cmd-survey. GPU NIS when available.

---

## iter_040 — enable_imu flag + NIS diagnostic flags (5 new tests, 132/132 total)  (2026-04-08T08:10:00Z)
Hypothesis: Adding enable_imu toggle to BallObsNoiseCfg allows ablation of IMU corrections.
Change:     Added enable_imu to BallObsNoiseCfg (default True). --no-imu and --enable-spin
            flags on nis_diagnostic.py. 5 tests added.
Command:    `pytest scripts/perception/test_*.py` → 132/132 pass.
Result:     **5/5 new pass.** GPU NIS blocked by policy agent.
Decision:   Velocity command work (CPU-only) while GPU blocked.

---

## iter_041 — Velocity command modules: UserVelocityInput + CommandMixer (21/21 tests, 153/153 total)  (2026-04-08T09:30:00Z)
Hypothesis: Method 1 (Direct Override) gives users joystick/keyboard vx/vy control during play.
Change:     Created vel_cmd/ package: user_velocity_input.py (threaded input), command_mixer.py
            (override/blend/passthrough modes). 21 tests in 8 classes.
Command:    `pytest scripts/perception/test_vel_cmd.py -v` → 21/21; full suite 153/153.
Result:     **21/21 pass.** All blend modes verified. 153/153 total.
Decision:   Create play_teleop.py integration script.

---

## iter_042 — play_teleop.py integration script (7/7 new tests, 160/160 total)  (2026-04-08T11:00:00Z)
Hypothesis: Standalone play_teleop.py wires UserVelocityInput + CommandMixer into play loop.
Change:     Created scripts/rsl_rl/play_teleop.py with backend selection, telemetry, video support.
            7 integration tests (TestTeleopFlow class).
Command:    `pytest scripts/perception/test_vel_cmd.py -v` → 28/28; full suite 160/160.
Result:     **7/7 pass.** 160/160 total. Script ready for use.
Decision:   Check vel-cmd-survey subagent. Kill if done.

---

## iter_043 — ResidualMixer (Method 2) + kill vel-cmd-survey (16/16 new tests, 176/176 total)  (2026-04-08T16:30:00Z)
Hypothesis: ResidualMixer (Method 2) provides production velocity command architecture.
Change:     Killed vel-cmd-survey subagent. Created vel_cmd/residual_mixer.py: ResidualMixer +
            ResidualMixerCfg. 16 tests in 5 classes.
Command:    `pytest scripts/perception/test_vel_cmd.py -v` → 44/44; full suite 176/176.
Result:     **16/16 pass.** Residual adds pi1+user (vs override which discards pi1). 176/176 total.
Decision:   Policy agent handoff for Method 2 requirements. GPU NIS when available.

---

## iter_044 — compaction (summarized iters 029-035)  (2026-04-08T18:00:00Z)
Hypothesis: N/A — compaction iteration.
Change:     Archived iters 029-035 verbatim to RESEARCH_LOG_ARCHIVE.md. Rewrote compacted summary
            covering all phases through iter_035 (roadmap, EKF core, pipeline, NIS debugging,
            contact-aware fix, Ahn 2019 calibration). Kept iters 036-043 verbatim. Pruned fix_plan.
Command:    No GPU commands.
Result:     Log shrunk from 323->~155 lines. Archive now has 35 verbatim entries (001-035).
            Fix plan consolidated to 11 forward-looking tasks across 3 phases.
Decision:   Next iter: write handoff note to policy agent INBOX about Method 2 requirements
            (obs 40->42D, vel_tracking reward, hot-start from Stage G). GPU NIS validation
            (IMU on/off, 9D spin) when GPU available.

---

## iter_045 — Hough circle fallback detector + from_yaml (15/15 new tests, 191/191 total)  (2026-04-08T19:15:00Z)
Hypothesis: Hough circle detection on depth frames provides robust ball detection
            when YOLO is unavailable or low-confidence, with <10mm error at 0.3-1.0m.
Change:     Implemented `BallDetector._detect_hough()` using cv2.HoughCircles on
            normalised 8-bit depth images. Circle scoring by radius-ratio vs expected
            ball size at detected depth. Implemented `BallDetector.detect()` fallback
            chain (YOLO→Hough→low-conf YOLO). Also implemented
            `CameraCalibrator.from_yaml()` (YAML extrinsics loader with validation).
            Wrote policy agent handoff note about Method 2 velocity commands.
Command:    `uv run --active python scripts/perception/test_hough_detector.py -v` → 15/15
            Full suite: 191/191 pass (174 pytest + 17 real_utils).
Result:     **15/15 new tests pass.** Hough detects ball at 30cm/50cm/1m with <10mm error.
            Works with 2mm depth noise. Empty-frame correctly returns None. Bbox/confidence
            valid. from_yaml loads identity + non-trivial extrinsics, validates shape.
            Wrote handoff to policy INBOX re: Method 2 (obs 40→42D, vel_tracking reward).
Decision:   GPU NIS with IMU on/off when GPU available. Else: implement more mock-testable
            real pipeline pieces (e.g. threaded pipeline integration with MockCamera+Hough).

---

## iter_046 — Threaded RealPerceptionPipeline + integration tests (17/17 new tests, 208/208 total)  (2026-04-08T20:30:00Z)
Hypothesis: A threaded RealPerceptionPipeline (camera+detector on acq thread, EKF on main thread)
            can be fully tested with MockCamera + MockDetector, validating the real-time architecture.
Change:     Replaced pipeline.py stubs with working implementation: acquisition thread runs
            camera.get_frame() + detector.detect() and pushes _Measurement to a lock-guarded deque;
            get_observation() drains queue, runs EKF predict+update, transforms to body frame.
            Dependency injection: accepts any camera/detector matching the interface (MockCamera,
            MockDetector, BallDetector w/ Hough, or future real D435i). Added _quat_to_rotmat(),
            PipelineObservation (extended with timestamp + ekf_pos_w/ekf_vel_w debug fields),
            reset_ekf(), stats property. 17 integration tests across 8 test classes:
            lifecycle, convergence, dropout, body-frame transform, extrinsics, EKF reset, Hough.
Command:    `uv run --active python scripts/perception/test_threaded_pipeline.py -v` → 17/17
            Full suite (12 test files): 208/208 pass.
Result:     **17/17 new pass.** EKF converges to <5cm error on stationary ball within 200ms.
            Body-frame transform correct under rotation and translation. Dropout→ball_lost works.
            Hough-on-MockCamera end-to-end: <10cm error. Zero regressions.
Decision:   GPU NIS IMU on/off comparison next if GPU available. Else: NIS gating in pipeline
            (reject wild measurements via chi-squared test before EKF update).

---

## iter_047 — Chi-squared NIS gating in BallEKF (19/19 new tests, 227/227 total)  (2026-04-08T22:00:00Z)
Hypothesis: Per-env chi-squared NIS gating in BallEKF.update() rejects outlier measurements
            (detector glitches, multi-ball confusion) before they corrupt the state estimate.
Change:     Added NIS gating to BallEKF.update(): computes NIS for all envs, rejects measurements
            where NIS > nis_gate_threshold (default 11.345 = chi-squared 3DOF 99th percentile).
            Per-env warm-up: gating skipped for first nis_gate_warmup=50 updates per env after
            reset, since velocity takes many position-only observations to converge (position P
            drops in 1 step but velocity P needs ~50). Config: nis_gate_enabled (default True),
            nis_gate_threshold, nis_gate_warmup. Diagnostics: gate_rejection_rate,
            gate_rejection_count, reset_gate_stats(). _update_count per env, reset on env reset.
Command:    `uv run --active python scripts/perception/test_nis_gating.py -v` → 19/19
            Full suite (13 test files): 227/227 pass.
Result:     **19/19 new tests pass.** Zero regressions across all 12 pre-existing test files.
Decision:   GPU NIS IMU on/off when available. Else: NIS diagnostic logging.

---

## iter_048 — Gate rejection stats in pipeline diagnostics + NIS diagnostic tool (2/2 new tests, 229/229 total)  (2026-04-08T23:15:00Z)
Hypothesis: Surfacing NIS gate rejection counters in PerceptionPipeline.diagnostics and
            nis_diagnostic.py enables tuning gate threshold via the GPU diagnostic tool.
Change:     (1) PerceptionPipeline.diagnostics now includes gate_rejected, gate_total,
            gate_rejection_rate. (2) nis_diagnostic.py: added Gate% and Gated columns.
            (3) 2 new tests in test_world_frame_ekf.py.
Command:    All 13 test files: 229/229 pass (188 pytest + 41 manual).
Result:     **2/2 new tests pass.** 229/229 total. Zero regressions.
Decision:   GPU NIS IMU on/off comparison next if GPU available.
