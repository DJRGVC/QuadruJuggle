# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## Compacted summary (through iter_043)

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
(16/16) all pass.

**Iters 036 (compaction):** Archived iters 019-028.

**Iters 037-038 (IMU + spin extensions):** IMU-aided EKF adds Coriolis/centrifugal corrections
using robot angular velocity (16/16 tests). 9D spin estimation extends state to [pos, vel, spin]
with Magnus force dynamics (25/25 tests). Both optional, toggled via enable_imu/enable_spin flags.

**Iters 039-040 (config wiring):** Wired enable_spin and enable_imu through BallObsNoiseCfg +
PerceptionPipeline. Added --no-imu and --enable-spin flags to nis_diagnostic.py. Spawned
vel-cmd-survey subagent.

**Iters 041-043 (velocity commands):** Created vel_cmd/ package: UserVelocityInput (threaded
keyboard/joystick), CommandMixer (override/blend/passthrough), ResidualMixer (Method 2 —
pi1+user additive). play_teleop.py wires it all together. Killed vel-cmd-survey subagent
after it completed 5 iters. 176/176 tests total.

**Key architectural findings through iter_043:**
- Body-frame EKF structurally wrong due to unmodeled contact forces → contact-aware mode required
- Contact-aware EKF: inflate process noise during paddle contact → NIS from 966 to 0.78
- Ahn 2019 noise: sigma_xy=0.0025*z, sigma_z=1mm+0.005*z², dropout 20-50% with distance
- Raw d435i noise outperforms EKF for training (noise acts as regularization)
- EKF value: velocity estimation + dropout bridging during free-flight only
- IMU/spin features: non-harmful in sim (random actions → low ω), ready for real hardware
- Velocity commands: Method 1 (override) implemented; Method 2 (residual) needs pi1 retraining

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

---

## iter_049 — GPU NIS validation: IMU ON vs OFF vs 9D Spin comparison  (2026-04-08T16:10:00Z)
Hypothesis: IMU-aided Coriolis/centrifugal corrections and 9D spin estimation will measurably
            improve EKF NIS consistency compared to baseline (no IMU).
Change:     Ran nis_diagnostic.py three times on GPU with 2048 envs × 500 steps, random actions.
Result:     **All three modes functionally identical:**
            | Mode      | NIS   | In-band | EKF mm | Raw mm | Gate rej |
            |-----------|-------|---------|--------|--------|----------|
            | IMU ON    | 0.445 | 2/10    | 6.09   | 3.91   | 0/656k   |
            | IMU OFF   | 0.444 | 2/10    | 5.98   | 3.87   | 0/657k   |
            | 9D Spin   | 0.442 | 2/10    | 5.99   | 3.85   | 2/657k   |
            IMU corrections negligible with random actions (low ω). EKF over-conservative
            (NIS=0.44 vs target 3.0) due to contact-phase Q inflation.
Decision:   Phase-separated NIS tracking to isolate flight vs contact NIS.

---

## iter_050 — Phase-separated NIS tracking (flight vs contact) (10/10 new tests, 239/239 total)  (2026-04-08T17:00:00Z)
Hypothesis: The low overall NIS=0.44 (iter_049) is dominated by contact-phase Q inflation
            (q_vel_contact=50.0). Separating NIS by phase will reveal whether free-flight
            q_vel=0.40 is well-calibrated independently.
Change:     Added phase-separated NIS accumulators to BallEKF. PerceptionPipeline.diagnostics
            now includes mean_nis_flight, mean_nis_contact. nis_diagnostic.py displays columns.
Command:    `uv run --active python scripts/perception/test_nis_phase.py -v` → 10/10
            Full suite (14 test files): 239/239 pass.
Result:     **10/10 new tests pass.** CPU test confirms contact NIS << flight NIS as expected.
Decision:   GPU NIS phase-separated validation next: verify free-flight NIS closer to 3.0.

---

## iter_051 — subagent management + report-writer spawn  (2026-04-08T17:15:00Z)
Hypothesis: N/A — operational iteration (INBOX processing, subagent lifecycle).
Change:     (1) Killed lit-review and vel-cmd-survey subagents per Daniel's request (both already
            stopped; freed 2/5 agent slots). (2) Spawned report-writer subagent (sonnet, max 10
            iters) per Daniel's request for formatted project report with embedded media/graphs.
            Wrote detailed INBOX to report-writer with section outline and data sources.
            (3) Notified policy agent about report-writer and killed subagents.
            (4) Attempted GPU phase-separated NIS validation but GPU locked by policy training
            (12288-env, 1500-iter run). Killed my waiting gpu_lock processes.
Command:    `c3r kill /home/daniel-grant/Research/QuadruJuggle lit-review` → ✓
            `c3r kill /home/daniel-grant/Research/QuadruJuggle vel-cmd-survey` → ✓
            `c3r spawn ... report-writer generic "..." --model sonnet` → ✓
Result:     Agents: 3/5 (perception, policy, report-writer). GPU unavailable this iter.
Decision:   GPU NIS phase-separated validation next iter (when GPU free). Monitor report-writer
            progress. If GPU still busy, do more non-GPU improvements (e.g. documentation,
            test cleanup, or data preparation for report).

---

## iter_052 — nis_sweep phase-separated output + test verification  (2026-04-08T18:10:00Z)
Hypothesis: Updating nis_sweep.py to parse and display phase-separated NIS (flight vs contact)
            will make Q-tuning sweeps immediately actionable.
Change:     Updated nis_sweep.py: now parses Flight/Contact NIS from nis_diagnostic.py stdout
            and displays them in the summary table (new columns: flight, contact). Also verified
            full test suite: 239/239 pass (CPU). GPU NIS validation blocked by policy training
            (step 6300/7248, ~55 min remaining — exceeds iteration budget).
Command:    `uv run --active python -m pytest scripts/perception/ --ignore=... -v` → 239/239 pass
Result:     nis_sweep.py improved. GPU still locked by policy (iter_016 training, 12288 envs).
            Report-writer completed iter_001 (880-line HTML report). Policy at step 6300 with
            sustained juggling (apex ~10, curriculum advancing through stages).
Decision:   GPU NIS phase-separated validation FIRST PRIORITY next iter. Policy run should finish
            within ~55 min. If GPU still busy, run nis_sweep with reduced q_vel values once free.

---

## iter_053 — eval_perception_live.py: EKF accuracy under trained policy  (2026-04-08T19:30:00Z)
Hypothesis: EKF accuracy measured under random actions (iter_049) may differ from accuracy
            during actual juggling behavior. A trained-policy evaluation script is needed.
Change:     Wrote scripts/perception/eval_perception_live.py — loads trained pi1 checkpoint
            via rsl_rl OnPolicyRunner, runs env with EKF mode, logs phase-separated NIS,
            position RMSE, gate rejections, timeout%, and episode stats. Supports all EKF
            tuning overrides (q_vel, q_vel_contact, etc.) and target height specification.
            GPU blocked by policy training (iter_016, step ~6350/7248, ~50 min remaining).
            Report-writer at iter_003 (1014 lines), progressing well.
Command:    AST parse verification only (no GPU). Syntax clean, 276 lines.
Result:     Script ready. Cannot GPU-validate until policy training completes.
            Policy agent actively training with our d435i noise model — curriculum advancing
            through stages with noise_scale ramping 0→1.0.
Decision:   GPU phase-separated NIS validation FIRST when GPU frees up. Then run
            eval_perception_live.py with policy's best checkpoint to compare NIS under
            realistic vs random trajectories.

---

## iter_054 — eval_perception_live.py improvements: JSON output + RMSE tracking  (2026-04-08T21:20:00Z)
Hypothesis: eval_perception_live.py needs JSON output and per-interval RMSE averaging to
            enable systematic comparison across EKF parameter sweeps.
Change:     (1) Added `--output` arg for JSON summary file. (2) Added per-interval EKF/raw
            RMSE tracking with running averages in summary. (3) Results dict includes EKF
            config, episode stats, and all key metrics. (4) Verified all 239 tests still pass
            (14 test files, run standalone; test_ekf_integration.py excluded as GPU-only).
            GPU blocked by policy training (iter_016, step ~6550/7248, ~40 min remaining).
            Report-writer at iter_006 (1394-line HTML report, nearly complete — 1 task left).
            Policy agent at iter_015 with curriculum advancing (Stage B, apex=10.5, timeout=68%).
Command:    AST parse + manual test suite run (no GPU). 14/14 test files pass (239 tests).
Result:     Script improved, ready for GPU eval. Policy checkpoints available at
            QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_10-36-50/.
            Pi2 checkpoint at QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_17-16-01/.
Decision:   GPU phase-separated NIS validation + eval_perception_live.py run NEXT iter when
            GPU frees up. Policy training should complete within ~40 min. Prepare exact GPU
            commands: (1) nis_diagnostic.py 2048 envs × 500 steps, (2) eval_perception_live.py
            512 envs × 1000 steps with policy's model_best.pt.
