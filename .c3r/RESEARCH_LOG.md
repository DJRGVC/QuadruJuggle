# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## Compacted summary (through iter_018)

**Iters 001-004 (docs/roadmap):** Updated perception_roadmap.md and sim_to_real_plan.md for D435i
(stereo depth, not monocular). Surveyed Isaac Lab camera APIs — TiledCamera for debug only;
training uses ETH noise-injection on GT state (no camera sensor needed). Created ball_obs_spec.py
with 3 modes: oracle (GT passthrough), d435i (structured depth noise), ekf (EKF-filtered).
CAMERA_CHOICE.md and REFERENCES.md document decisions.

**Iters 005-008 (camera + EKF core):** Mounted simulated D435i TiledCamera in PLAY scene
(trunk/D435i, 30Hz, 640×480, 86° HFOV, 45° pitch). Verified RGB+depth capture in headless.
Implemented ball_ekf.py: batched 6-state EKF (pos+vel), ballistic+quadratic-drag dynamics,
Joseph-form covariance update, 5 CPU tests pass.

**Iters 009-011 (full pipeline + handoff):** Created noise_model.py (depth-dependent σ,
hold-last-value dropout, latency buffer) and PerceptionPipeline class (chains GT→noise→EKF→obs).
Integration tested in Isaac Lab: 4096 envs × 50 iters, mean_ep_len 21→123. Wrote
PERCEPTION_HANDOFF.md for policy agent.

**Iters 012-015 (diagnostics + curriculum + lit-review):** Added _PerceptionDiagnostics
(RMSE, NIS). Added noise_scale curriculum support (0.0-1.0 multiplier). Added body-frame
gravity correction to EKF. Spawned lit-review subagent (sonnet). Fixed compare script
(subprocess isolation for mode switching).

**Iters 016-018 (comparison + EKF tuning):** Fixed critical EKF vel-view mutation bug →
covariance explosion → NaN. 3-mode comparison (2048 envs × 50 iters): oracle=13.7,
d435i=10.5, ekf=7.6 reward. EKF 28% below raw d435i — over-smoothing from q_vel=0.15
(CWNA prescribes ~0.40). Fixed pi2 obs dim mismatch (41→53, auto-detect from checkpoint).
CWNA-tuned Q/R: q_pos=0.003, q_vel=0.15→0.30, r_xy=0.002, time-varying r_z.

**Key architectural findings through iter_018:**
- EKF in body frame suffers from unmodeled pseudo-forces (Coriolis/centrifugal/Euler)
- Raw d435i noise actually outperforms EKF for training (noise acts as regularization)
- TiledCamera adds obs to pi2 scene → must isolate to DEBUG scene subclass only
- Policy agent consumes perception via BallObsNoiseCfg(mode="d435i") — drop-in swap

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
            iters 019-028 verbatim. Pruned fix_plan.md — removed 20+ completed items, kept
            7 hardware-blocked + 4 sim-side improvement tasks.
Command:    No GPU commands.
Result:     Log shrunk from 313→~115 lines. Archive preserved 18 verbatim entries. Fix plan
            consolidated to 11 forward-looking tasks.
Decision:   Next iter: check policy agent status (iter_013, training with ball_low=-2.0).
            If policy needs perception support, prioritise that. If not, consider contact-aware
            EKF (Phase 4) — detecting paddle contact and switching to zero-accel dynamics
            would fix the NIS=970 root cause and make EKF useful during training too.

---

## iter_030 — Contact-aware EKF: adaptive process noise during paddle contact  (2026-04-08T17:15:00Z)
Hypothesis: Inflating q_vel during contact (ball Z < 25mm) lets EKF use low q_vel=0.40 for
            free-flight smoothing while trusting measurements during contact (q_vel=50.0),
            fixing the NIS=970 root cause without permanently degrading accuracy.
Change:     BallEKFConfig: q_vel 7.0→0.40 (free-flight CWNA), added contact_aware=True,
            q_vel_contact=50.0, contact_z_threshold=0.025m. predict() builds per-env Q
            matrix based on ball Z position. Added --no-contact-aware flag to nis_diagnostic.py.
Command:    test_contact_aware_ekf.py (7 tests), test_world_frame_ekf.py (6), test_mock_pipeline.py (15).
Result:     **28/28 tests pass.** Free-flight P growth ~6.4e-5 (low noise → good smoothing).
            Contact P growth >0.5 (high noise → trust measurements). Mixed-env contact/flight
            ratio >100×. Bounce trajectory tracked without divergence. Backward compat OK.
Decision:   GPU NIS validation next: run nis_diagnostic.py with contact_aware=True. Expected:
            NIS near 3.0 during free flight (was 970 everywhere with q_vel=7.0). Contact phases
            will still have high NIS (unavoidable — contact forces truly unpredictable) but
            free-flight should now show proper EKF smoothing benefit over raw noise.

---

## iter_031 — GPU NIS validation: contact-aware EKF vs uniform q_vel  (2026-04-08T18:35:00Z)
Hypothesis: Contact-aware EKF (q_vel=0.40 free-flight, 50.0 contact) will achieve NIS≈3.0
            during free-flight phases, vs NIS≈700 with uniform q_vel=0.40.
Change:     No code changes — diagnostic validation of iter_030's contact-aware EKF.
Command:    `nis_diagnostic.py --num_envs 256 --steps 200 --headless` (with and without --no-contact-aware)
Result:     **Contact-aware ON**: NIS=0.78, 10/10 intervals in 95% band. EKF RMSE=5.4mm, raw=4.5mm.
            **Contact-aware OFF**: NIS=671, 0/10 in band. EKF RMSE=155mm (divergent).
            
            Key findings:
            - 860× NIS improvement (671→0.78). NIS now in-band for all intervals.
            - NIS=0.78 slightly over-conservative (target 3.0) → q_vel could be reduced to ~0.15
              but current 0.40 is safe and conservative.
            - EKF position RMSE (5.4mm) worse than raw (4.5mm) because random-action test
              keeps ball mostly in contact phase where EKF is intentionally loose.
            - EKF value is velocity estimation + dropout bridging, not position during contact.
            - Detection rate ~90% consistent across both modes (expected).
Decision:   GPU NIS validation complete. Contact-aware EKF confirmed working. Mark task done
            in fix_plan. Next: ballistic trajectory testing in mock pipeline, or check if
            policy agent needs perception support for noise curriculum tuning.

---

## iter_032 — Ballistic trajectory tests for mock pipeline  (2026-04-08T19:30:00Z)
Hypothesis: EKF with contact-aware mode correctly tracks parabolic arcs across all
            curriculum stages (A/D/G), achieving pos RMSE <20mm during flight.
Change:     Created test_ballistic_trajectory.py with 13 tests covering:
            - Free-flight arcs at Stage A (10cm), D (45cm), G (1.0m) apex heights
            - Apex velocity tracking (vz ≈ 0 at top)
            - Noisy measurements (5mm std) and 20% dropout
            - Contact-aware vs uniform comparison
            - Contact→flight→contact transitions (single bounce cycle)
            - Contact/flight phase Q inflation verification
            - Multi-bounce tracking (3 consecutive bounces)
            - NIS boundedness across bounces
            - Off-axis diagonal launch (vx=0.3, vy=0.1, vz=2.5)
Command:    `python scripts/perception/test_ballistic_trajectory.py`
Result:     **13/13 pass** (1.86s). Key metrics:
            - Stage A (10cm): pos RMSE 15.5mm (near contact zone → looser bound 20mm)
            - Stage D (45cm): pos RMSE <15mm, apex Z error <20mm
            - Stage G (1.0m): pos RMSE <20mm, vel RMSE <0.8 m/s
            - Noisy 5mm: pos RMSE <25mm
            - 20% dropout: pos RMSE <30mm (EKF bridges gaps)
            - Contact phase: P vel growth >0.5 (q_vel_contact=50 working)
            - Flight phase: P vel growth <0.01 (q_vel=0.40 smooth)
            - Multi-bounce RMSE <20mm, NIS bounded
            - Existing tests: 15/15 mock + 7/7 contact-aware still pass (35 total)
Decision:   Ballistic trajectory validation complete. Next: latency injection testing
            (verify policy robustness to 1-3 frame observation delays), or check if
            policy agent needs perception support.

---

## iter_033 — Latency injection tests: buffer + EKF degradation  (2026-04-08T20:30:00Z)
Hypothesis: D435i latency buffer correctly delays observations by N steps, and EKF
            tracking degrades gracefully (monotonic RMSE increase) with 1-3 frame delays.
Change:     Created test_latency_injection.py with 16 tests in 3 test classes:
            - TestLatencyBuffer (6): zero/1/2/3-step delay correctness, reset, dropout interaction
            - TestLatencyEKFDegradation (8): RMSE bounds per latency level, monotonic increase,
              velocity usable at 3 frames, combined latency+dropout, multi-env independence
            - TestLatencyPolicyImpact (2): apex detection accuracy with 3-frame delay
            Used free-flight arcs (z > contact_z_threshold) to isolate latency from contact-aware mode.
Command:    `python scripts/perception/test_latency_injection.py`
Result:     **16/16 pass** (0.072s). Key findings:
            - Latency buffer delays exactly N steps (verified with zero-noise deterministic tests)
            - Pos RMSE: 0-lat <15mm, 1-lat <40mm, 2-lat <80mm, 3-lat <120mm (all pass)
            - Vel RMSE at 3-frame delay: <2 m/s (usable for pi1 planning)
            - 2-frame + 10% dropout: pos RMSE <100mm, max error <200mm
            - 2-frame + 20% dropout: pos RMSE <150mm, max error <300mm
            - Multi-env independence: no cross-talk through shared latency buffer
            - At apex (vz≈0), 3-frame delay causes <50mm Z error (minimal)
            - All existing tests still pass: 57 total (7+6+15+13+16)
Decision:   Latency injection validated. All sim-side fix_plan items complete except
            "check if policy needs support". Policy agent at iter_014 (ball_release_velocity
            reward), not yet back to noise robustness. Next: either propose new sim-side work
            (e.g. observation noise during contact phases, or spin estimation) or wait for
            policy to reach noise curriculum stage and provide support then.
