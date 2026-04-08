# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## Compacted summary (through iter_028)

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

**Iters 019-024 (NIS debugging — body-frame EKF is fundamentally broken):** Created
nis_diagnostic.py and nis_sweep.py for rapid EKF validation. NIS=966 with q_vel=0.30
(target 3.0) — 30x worse than raw noise. Root cause: body-frame pseudo-forces (Coriolis,
centrifugal, Euler) from robot angular motion, not modeled in EKF dynamics. Body-frame
accel compensation tried (iter_022) — NIS=1025 (worse). World-frame EKF implemented
(iter_023) — NIS still 970. Final diagnosis (iter_024): **contact normal forces** during
paddle contact are the dominant unmodeled acceleration, not frame effects. q_vel sweep
{0.30-10.0}: at q_vel>=5.0 NIS in-band but EKF = raw noise accuracy. Set q_vel=7.0 default.
EKF value is velocity estimation + dropout bridging, not position smoothing during contact.

**Iters 025-028 (real hardware pipeline + feature-complete):** Declared sim pipeline
feature-complete. Created docs/hardware_pipeline_architecture.md. Calibrated noise model
(sigma_z 2->5mm/m, dropout 2->10%). Built perception/real/ with 6 files (camera, detector,
calibration, config, pipeline stubs + from_known_mount implementation). Created MockCamera +
MockDetector for hardware-free integration testing. 17/17 real util tests, 15/15 mock
pipeline tests. All remaining Phase 3 items hardware-blocked.

**Key architectural findings through iter_028:**
- Body-frame EKF structurally wrong due to unmodeled pseudo-forces -> world-frame required
- Even world-frame EKF broken by unmodeled contact normal forces (NIS=970)
- Raw d435i noise outperforms EKF for training (noise acts as regularization)
- EKF value: velocity estimation + dropout bridging during free-flight only
- TiledCamera adds obs to pi2 scene -> must isolate to DEBUG scene subclass only
- Policy agent consumes perception via BallObsNoiseCfg(mode="d435i") — drop-in swap

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
            summary to cover NIS debugging saga (019-024: body-frame broken, world-frame still
            broken, root cause = contact forces, contact-aware EKF solution) and real hardware
            pipeline (025-028: stubs, mocks, unit tests). Kept iters 029-035 verbatim.
            Pruned fix_plan.md — removed all completed Phase 4 items.
Command:    No GPU commands.
Result:     Log shrunk from 310->~120 lines. Archive now has 28 verbatim entries (001-028).
            Fix plan consolidated to 9 forward-looking tasks (6 hardware-blocked, 3 Phase 5).
Decision:   Next iter: check policy agent status. If they've moved past reward shaping to
            noise robustness, provide perception support. If not, start IMU-aided EKF (Phase 5)
            — compensate platform angular motion during EKF predict step for improved real-world
            accuracy. This is the highest-value sim-side work remaining.

---

## iter_037 — IMU-aided EKF: Coriolis + centrifugal corrections (16/16 tests pass)  (2026-04-08T16:30:00Z)
Hypothesis: Adding Coriolis (-2ω×v) and centrifugal (-ω×(ω×r)) pseudo-force corrections
            using robot angular velocity will make body-frame EKF physically correct under
            platform rotation, improving tracking accuracy without requiring world-frame mode.
Change:     Added `robot_ang_vel_b` parameter to `BallEKF.predict()` and `.step()`.
            When provided, computes Coriolis + centrifugal accelerations and adds them to the
            prediction dynamics. Linearised Jacobian F updated: d(a_cor)/d(vel) = -2[ω]_x,
            d(a_cent)/d(pos) = -[ω]_x² for correct covariance propagation. Added `_batch_skew()`
            helper. Pipeline (`ball_obs_spec.py`) now passes `robot.data.root_ang_vel_b` through
            to EKF in body-frame mode. Euler force omitted (would need noisy gyro finite-diff;
            ~1 m/s² at typical Go1 rates — covered by process noise).
Command:    `python scripts/perception/test_imu_aided_ekf.py -v` (16 tests)
            Full suite: 90/90 tests pass (7+6+15+13+16+17+16).
Result:     **16/16 new tests pass.** Coriolis magnitude exact to 3 decimal places.
            Centrifugal magnitude exact. 3D omega analytical match. Multi-step tracking
            under 1.5 rad/s rotation: pos RMSE <10mm, vel RMSE <0.5 m/s.
            Tracking degrades without IMU (controlled comparison). Per-env omega works.
            90/90 total tests pass — no regressions.
Decision:   GPU NIS validation of body-frame+IMU vs world-frame next iteration. If body-frame
            NIS comes in-band with IMU corrections, it becomes the simpler default (no full
            robot pose needed). Also need to check if policy agent needs perception support.

---

## iter_038 — 9D spin estimation: Magnus effect EKF extension (25/25 tests, 109/109 total)  (2026-04-08T17:45:00Z)
Hypothesis: Extending the EKF state to 9D [pos, vel, spin] with Magnus force dynamics allows
            the filter to estimate ball spin from trajectory curvature, improving prediction
            accuracy for spinning balls without requiring direct spin measurement.
Change:     Added optional `enable_spin` mode to BallEKF (default False — 6D unchanged):
            - Config: `enable_spin`, `magnus_coeff=0.0149` (Kutta-Joukowski for 40mm ball),
              `spin_decay_rate=0.008` (Stokes viscous torque), `q_spin=1.0`, `q_spin_contact=100.0`,
              `p_spin_init=10.0`
            - Magnus force: a_M = Cm * (spin × vel), with correct Jacobians for spin_skew
              and vel_skew cross products in F matrix (d(a)/d(vel) = Cm*[spin]_x,
              d(a)/d(spin) = -Cm*[vel]_x)
            - Spin decay: exponential with factor exp(-decay_rate*dt), Jacobian on F diagonal
            - Contact-aware: q_spin inflated to q_spin_contact when ball Z < threshold
            - All matrices (P, F, Q, H, K) properly sized as D×D where D=6 or 9
            - `spin` property, `init_spin` param in reset(), `state_dim` property
            Created test_spin_estimation.py with 25 tests across 6 test classes.
Command:    `pytest scripts/perception/test_spin_estimation.py -v` (25 tests)
            Full suite: `pytest scripts/perception/test_*.py` (CPU-only) → 109/109 pass.
Result:     **25/25 new tests pass.** Magnus direction correct (cross product verified for
            3 axis combinations). Physical magnitude: Cm*ω*v = 0.0149*20*3 = 0.894 m/s²
            matches actual EKF output. Spin decay exponential exact. Topspin curves ball
            downward as expected. EKF successfully estimates spin=40 rad/s from curvature alone
            (converges to >5 rad/s z-component from zero initial spin). Contact noise 10000×
            larger than free-flight. All 84 existing tests pass — zero regressions.
            GPU NIS validation deferred — GPU locked by policy agent training.
Decision:   Next iteration: GPU NIS validation comparing body-frame+IMU (iter_037) against
            world-frame. Also try 9D spin mode in NIS diagnostic if GPU available. If GPU
            still locked, integrate spin mode into ball_obs_spec.py pipeline (wire enable_spin
            through BallObsNoiseCfg so sim can toggle it).

---

## iter_039 — Wire enable_spin + spawn vel-cmd-survey subagent (12/12 new tests, 127/127 total)  (2026-04-08T15:00:00Z)
Hypothesis: BallObsNoiseCfg.enable_spin should propagate to EKF via PerceptionPipeline so users
            can toggle spin estimation from the env config without touching EKF internals.
Change:     (1) Added `enable_spin: bool = False` field to BallObsNoiseCfg.
            (2) PerceptionPipeline.__init__ propagates enable_spin to BallEKFConfig (copies config
                if needed, sets enable_spin=True). Redundant propagation also in _get_or_create_pipeline.
            (3) Added `spin` property to PerceptionPipeline (returns body-frame spin or None).
            (4) Created test_pipeline_config.py: 12 tests covering spin wiring (7), world_frame (2),
                noise_scale (3). All config flags validated.
            (5) Spawned vel-cmd-survey subagent (sonnet, max 5 iters) per Daniel's request to
                research user-defined velocity input methods from 2023-2026 papers.
            (6) Forwarded velocity input request to policy agent's INBOX.
Command:    `python -m pytest scripts/perception/test_*.py -v` (excluding test_ekf_integration.py)
Result:     **127/127 tests pass** (12 new config tests + 115 existing). No regressions.
            GPU locked by policy agent — no NIS validation this iter.
Decision:   Next: monitor vel-cmd-survey subagent (kill after 5 iters or findings).
            GPU NIS validation of body-frame+IMU vs world-frame when GPU available.
            Check vel-cmd-survey progress and policy agent status.

---

## iter_040 — enable_imu flag + NIS diagnostic --no-imu/--enable-spin flags (5 new tests, 132/132 total)  (2026-04-08T08:10:00Z)
Hypothesis: Adding an `enable_imu` toggle to BallObsNoiseCfg allows ablation of IMU-aided
            Coriolis/centrifugal corrections, enabling GPU NIS comparison of IMU vs no-IMU.
Change:     (1) Added `enable_imu: bool = True` to BallObsNoiseCfg. Default True (backward-compat).
            (2) `ball_pos_perceived` and `ball_vel_perceived` conditionally pass `robot_ang_vel_b`
                as None when `enable_imu=False`, disabling Coriolis/centrifugal corrections.
            (3) Added `--no-imu` and `--enable-spin` flags to nis_diagnostic.py.
            (4) Added 5 tests to test_pipeline_config.py: default=True, can disable, independent
                of world_frame, step-without-IMU works, step-with-IMU works.
Command:    `pytest scripts/perception/test_*.py` (excluding test_ekf_integration.py)
Result:     **132/132 tests pass** (5 new + 127 existing). GPU NIS validation blocked — policy
            agent training holds GPU lock (train_juggle_hier.py with d435i noise mode).
            vel-cmd-survey subagent running (iter_0, first iteration in progress).
Decision:   Next: GPU NIS comparison (IMU ON vs OFF) when GPU available.
            Monitor vel-cmd-survey subagent progress. If GPU still blocked next iter,
            do world-frame NIS comparison preparation or other CPU-only work.

---

## iter_041 — Velocity command modules: UserVelocityInput + CommandMixer (21/21 tests, 153/153 total)  (2026-04-08T09:30:00Z)
Hypothesis: Implementing the Method 1 (Direct Override) velocity command spec from vel-cmd-survey
            subagent gives users joystick/keyboard control of robot vx/vy during play, overriding
            pi1's velocity channels while preserving pi1's height/tilt control.
Change:     Created `vel_cmd/` package with 3 files:
            - `user_velocity_input.py`: thread-safe joystick/keyboard reader (pygame, pynput, zero backends).
              Safety-clamped to ±0.30 m/s (60% of pi2 Stage G training max). Deadband, axis invert.
              `get_cmd_tensor()` broadcasts to all envs for Isaac Lab batch injection.
            - `command_mixer.py`: 3 modes (override/blend/passthrough). Override fully replaces
              pi1 cmd[6], cmd[7] with user input. Blend uses alpha. Passthrough = no-op.
            - `__init__.py`: re-exports all public classes.
            Created `test_vel_cmd.py` with 21 tests (8 classes: zero backend, config, manual state,
            passthrough, override, blend, config customisation).
Command:    `pytest scripts/perception/test_vel_cmd.py -v` (21 tests)
            Full suite (excluding GPU test_ekf_integration.py): 153/153 pass.
Result:     **21/21 new tests pass.** All blend modes verified numerically. Override preserves
            dims 0-5 and does not modify input tensor. Zero backend returns (0,0) without threads.
            Normalization constants match action_term._CMD_SCALES[6:8]. 153/153 total — no regressions.
Decision:   Next: create play_teleop.py integration script (hooks mixer into play loop).
            Monitor vel-cmd-survey subagent (iter_002 done; Methods 2/3 specs pending).
            GPU NIS validation still blocked by policy agent training.

---

## iter_042 — play_teleop.py integration script (7/7 new tests, 160/160 total)  (2026-04-08T11:00:00Z)
Hypothesis: A standalone play_teleop.py script that wires UserVelocityInput + CommandMixer into
            the Isaac Lab play loop gives users keyboard/joystick control of vx/vy during
            hierarchical ball juggling, completing the Method 1 (Direct Override) pipeline.
Change:     Created `scripts/rsl_rl/play_teleop.py`:
            - Pre-parses --backend (pygame/keyboard/zero), --blend-mode, --blend-alpha, --pi2-checkpoint
            - Initializes UserVelocityInput + CommandMixer before the play loop
            - In loop: policy(obs) → mixer.mix(actions, vel_user) → env.step(mixed_actions)
            - Telemetry: prints user vx/vy vs actual vx/vy + ball_z every 20 steps
            - Clean shutdown: KeyboardInterrupt → vel_input.stop() → env.close()
            - Video recording support via --video flag
            Added 7 integration tests to test_vel_cmd.py (TestTeleopFlow class): zero input,
            walk forward, strafe left, blend mode, passthrough, max speed clamp, multi-step consistency.
Command:    `pytest scripts/perception/test_vel_cmd.py -v` (28 tests)
            Full suite (CPU-only, 10 test files): 160/160 pass.
Result:     **7/7 new teleop flow tests pass.** 28/28 vel_cmd tests total. 160/160 full suite.
            Script ready for use with:
            `uv run --active python scripts/rsl_rl/play_teleop.py --task Isaac-BallJuggleHier-Go1-Play-v0 --pi2-checkpoint <path> --num_envs 1 --backend keyboard`
Decision:   Next: check vel-cmd-survey subagent (should be on iter_004 with final proposal).
            If subagent done, kill it and assess Method 2/3 recommendations.
            GPU NIS validation (IMU on/off comparison) when GPU available.

---

## iter_043 — ResidualMixer (Method 2) + kill vel-cmd-survey (16/16 new tests, 176/176 total)  (2026-04-08T16:30:00Z)
Hypothesis: Implementing ResidualMixer (Method 2 from vel-cmd-survey) as a separate module
            provides the production architecture for velocity commands — pi1 outputs residual
            corrections added to user base velocity, preserving pi1's ability to compensate
            for ball drift during locomotion.
Change:     (1) Killed vel-cmd-survey subagent (4 iters, final proposal delivered with
                Method 1/2/3 comparison, risk analysis, phased impl order).
            (2) Created `vel_cmd/residual_mixer.py`: ResidualMixer + ResidualMixerCfg.
                mix() computes cmd[vx] = clamp(pi1_residual + vel_user, -max, +max).
                Configurable indices, max_total_norm, optional cfg (defaults to standard).
            (3) Updated `vel_cmd/__init__.py` to export ResidualMixer + ResidualMixerCfg.
            (4) Added 16 tests in 5 classes: basic arithmetic (7), config (4), vs-override
                comparison (2), teleop flow integration (3).
Command:    `pytest scripts/perception/test_vel_cmd.py -v` (44 tests)
            Full suite (10 test files, CPU-only): 176/176 pass.
Result:     **16/16 new tests pass.** 44/44 vel_cmd tests total. 176/176 full suite.
            Key verification: residual mixer adds pi1+user (unlike override which discards pi1),
            clamps correctly at boundaries, preserves dims 0-5, doesn't mutate input.
            GPU still locked by policy agent (fail_streak=8 on policy side).
Decision:   Next: write handoff note to policy agent INBOX about Method 2 requirements
            (obs 40→42D, vel_tracking reward, hot-start). GPU NIS validation when available.
            Consider compaction next iter (298 lines → close to 300 threshold).
