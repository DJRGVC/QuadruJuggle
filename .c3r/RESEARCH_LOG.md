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
