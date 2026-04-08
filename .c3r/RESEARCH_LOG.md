# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## Compacted summary (through iter_035)

**Iters 001-004 (docs/roadmap):** Updated perception_roadmap.md and sim_to_real_plan.md for D435i
(stereo depth, not monocular). Surveyed Isaac Lab camera APIs — TiledCamera for debug only;
training uses ETH noise-injection on GT state (no camera sensor needed). Created ball_obs_spec.py
with 3 modes: oracle (GT passthrough), d435i (structured depth noise), ekf (EKF-filtered).

**Iters 005-008 (camera + EKF core):** Mounted simulated D435i TiledCamera in PLAY scene
(trunk/D435i, 30Hz, 640x480, 86deg HFOV, 45deg pitch). Verified RGB+depth capture in headless.
Implemented ball_ekf.py: batched 6-state EKF (pos+vel), ballistic+quadratic-drag dynamics,
Joseph-form covariance update.

**Iters 009-011 (full pipeline + handoff):** Created noise_model.py (depth-dependent sigma,
hold-last-value dropout, latency buffer) and PerceptionPipeline class (chains GT->noise->EKF->obs).
Integration tested in Isaac Lab: 4096 envs x 50 iters, mean_ep_len 21->123. Wrote
PERCEPTION_HANDOFF.md for policy agent.

**Iters 012-015 (diagnostics + curriculum + lit-review):** Added _PerceptionDiagnostics
(RMSE, NIS). Added noise_scale curriculum support (0.0-1.0 multiplier). Added body-frame
gravity correction to EKF. Spawned lit-review subagent (sonnet). Fixed compare script.

**Iters 016-018 (comparison + EKF tuning):** Fixed critical EKF vel-view mutation bug ->
covariance explosion -> NaN. 3-mode comparison (2048 envs x 50 iters): oracle=13.7,
d435i=10.5, ekf=7.6 reward. EKF 28% below raw d435i — over-smoothing from q_vel=0.15.
Fixed pi2 obs dim mismatch (41->53, auto-detect from checkpoint).

**Iters 019-024 (NIS debugging — body-frame EKF fundamentally broken):** NIS=966 with
q_vel=0.30 (target 3.0). Root cause: contact normal forces during paddle contact are the
dominant unmodeled acceleration, not Coriolis/centrifugal frame effects. q_vel sweep: at
q_vel>=5.0 NIS in-band but EKF = raw noise accuracy. Set q_vel=7.0.

**Iters 025-028 (real hardware pipeline + feature-complete):** Sim pipeline declared
feature-complete. Created hardware pipeline stubs (camera, detector, calibration). MockCamera +
MockDetector for hardware-free testing. 32/32 real+mock tests pass.

**Iters 029-035 (contact-aware EKF + calibrated noise):** Contact-aware EKF (q_vel=0.40
free-flight, q_vel=50.0 during contact) achieved 860x NIS improvement (0.78 vs 671).
GPU-validated at NIS=1.60 with Ahn 2019-calibrated noise model (sigma_xy∝z, sigma_z∝z²,
distance-dependent dropout). Ballistic trajectory tests (13/13), latency injection tests
(16/16) all pass. Killed lit-review subagent after 29 iters.

**Key architectural findings through iter_035:**
- Body-frame EKF structurally wrong due to unmodeled contact forces -> contact-aware mode required
- Contact-aware EKF: inflate process noise during paddle contact -> NIS from 966 to 0.78
- Ahn 2019 noise: sigma_xy=0.0025*z, sigma_z=1mm+0.005*z², dropout 20-50% with distance
- Raw d435i noise outperforms EKF for training (noise acts as regularization)
- EKF value: velocity estimation + dropout bridging during free-flight only
- TiledCamera adds obs to pi2 scene -> must isolate to DEBUG scene subclass only
- Policy agent consumes perception via BallObsNoiseCfg(mode="d435i") — drop-in swap

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
