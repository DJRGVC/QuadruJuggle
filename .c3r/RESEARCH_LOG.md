# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## Compacted summary (through iter_055)

**Iters 001-004 (docs/roadmap):** Updated perception_roadmap.md and sim_to_real_plan.md for D435i
(stereo depth, not monocular). Surveyed Isaac Lab camera APIs — TiledCamera for debug only;
training uses ETH noise-injection on GT state (no camera sensor needed). Created ball_obs_spec.py
with 3 modes: oracle (GT passthrough), d435i (structured depth noise), ekf (EKF-filtered).

**Iters 005-008 (camera + EKF core):** Mounted simulated D435i TiledCamera in PLAY scene.
Implemented ball_ekf.py: batched 6-state EKF (pos+vel), ballistic+quadratic-drag dynamics,
Joseph-form covariance update.

**Iters 009-011 (full pipeline + handoff):** Created noise_model.py (depth-dependent sigma,
hold-last-value dropout, latency buffer) and PerceptionPipeline class. Integration tested.
Wrote PERCEPTION_HANDOFF.md for policy agent.

**Iters 012-018 (diagnostics + EKF tuning):** Added diagnostics (RMSE, NIS), noise_scale
curriculum, body-frame gravity correction. Fixed critical vel-view mutation bug. 3-mode
comparison: oracle=13.7, d435i=10.5, ekf=7.6 reward. EKF 28% below raw d435i.

**Iters 019-024 (NIS debugging — contact forces):** NIS=966 with q_vel=0.30. Root cause:
unmodeled contact normal forces during paddle contact. q_vel sweep found q_vel>=5.0 needed.

**Iters 025-028 (real hardware pipeline):** Hardware pipeline stubs, MockCamera+MockDetector.

**Iters 029-035 (contact-aware EKF + calibrated noise):** Contact-aware EKF (q_vel=0.40
free-flight, q_vel=50.0 contact) achieved 860x NIS improvement. GPU-validated NIS=1.60
with Ahn 2019 noise model. All tests pass.

**Iters 036-043 (extensions + velocity commands):** IMU-aided EKF (Coriolis corrections), 9D spin
estimation (Magnus force), config wiring through BallObsNoiseCfg. vel_cmd/ package:
UserVelocityInput, CommandMixer, ResidualMixer, play_teleop.py. 176/176 tests.

**Iters 044-048 (real pipeline + NIS tooling):** Hough circle fallback detector (<10mm at
0.3-1.0m). Threaded RealPerceptionPipeline with MockCamera integration tests. Chi-squared
NIS gating (11.345 threshold, 50-step warmup). 239/239 tests.

**Iters 049-055 (GPU NIS validation + live-policy eval):** IMU/spin modes functionally identical
under random actions (NIS≈0.44). Added phase-separated NIS (flight vs contact). Built
eval_perception_live.py for accuracy under trained policy. **CRITICAL: EKF severely
overconfident under active juggling (flight NIS=52.9, RMSE 22mm > raw 19mm).** Root cause:
violent paddle strikes produce near-instant velocity reversals; q_vel=0.4 trusts prediction
too much. Contact q_vel_contact=50.0 close to right (NIS=5.3). Random-action flight NIS=1.45
well-calibrated — the gap is entirely from active policy dynamics.

**Key architectural findings through iter_055:**
- Contact-aware EKF: inflate Q during paddle contact → NIS from 966 to 0.78
- Ahn 2019 noise: sigma_xy=0.0025*z, sigma_z=1mm+0.005*z², dropout 20-50%
- Raw d435i noise outperforms EKF for training (noise = regularization)
- EKF value: velocity estimation + dropout bridging during free-flight
- IMU/spin: non-harmful in sim, ready for real hardware
- Velocity commands: Method 1 (override) done; Method 2 needs pi1 retraining
- **Active-policy EKF tuning is fundamentally different from random-action tuning**
- 3-level q_vel architecture: contact=50.0, post_contact=20.0 (10 steps), flight=TBD

---

## iter_056 — compaction (summarized iters 044-048)  (2026-04-08T19:10:00Z)
Hypothesis: N/A — compaction iteration (log at 308 lines > 300 threshold).
Change:     Archived iters 044-048 verbatim to RESEARCH_LOG_ARCHIVE.md. Rewrote compacted summary
            through iter_048. Kept iters 049-055 verbatim. Killed all 3 children (lit-review,
            vel-cmd-survey, report-writer) per Daniel's request. Copied project report to
            docs/project_report.html. Processed 3 INBOX messages.
Command:    No GPU commands.
Result:     Log shrunk from 308 → ~170 lines. Archive now has 48 verbatim entries (001-048).
            Agents: 2/5 (perception + policy). Report accessible via browser.
Decision:   Next iter: q_vel sweep with eval_perception_live.py [2.0, 5.0, 10.0, 20.0] to find
            optimal flight q_vel where NIS ≈ 3.0 and EKF RMSE < raw RMSE.

---

## Iteration 57 — post-contact P inflation + q_vel sweep script  (2026-04-08T19:45:00Z)
Hypothesis: Flight NIS=52.9 under active policy is caused by stale velocity estimates in the
            first ~10 steps after a paddle bounce. A post-contact q_vel inflation window
            (20.0 for 10 steps) will help the filter converge to the new velocity faster.
Change:     (1) Added 3-level q_vel to BallEKF: contact=50.0, post_contact=20.0, flight=0.4.
            New BallEKFConfig fields: post_contact_steps=10, q_vel_post_contact=20.0.
            Predict method tracks per-env _post_contact_countdown (set on contact entry,
            decremented each flight step). Reset clears countdown.
            (2) Wrote sweep_q_vel.py: single env session, iterates q_vel values, collects
            NIS/RMSE per setting. Reuses env (no restart per q_vel point).
            (3) Added 2 new tests: post_contact_inflation_window, post_contact_reset_on_new_contact.
Command:    CPU tests only (GPU locked by policy agent training 12288-env run).
            uv run --active python scripts/perception/test_contact_aware_ekf.py → 9/9 pass
            uv run --active python scripts/perception/test_world_frame_ekf.py → 8/8 pass
            uv run --active python scripts/perception/test_imu_aided_ekf.py → 16/16 pass
Result:     33/33 CPU tests pass. Post-contact inflation implemented and verified.
            GPU sweep blocked — policy agent holds gpu_lock for train_juggle_hier.
Decision:   Next iter: run sweep_q_vel.py on GPU when lock frees. Sweep q_vel=[0.4,2.0,5.0,10.0,20.0,50.0]
            with post-contact inflation enabled. If GPU still locked, look at improving the
            sweep script or adding more CPU-only analysis.

---

## Iteration 58 — sweep script bugfixes + child cleanup  (2026-04-08T20:30:00Z)
Hypothesis: sweep_q_vel.py has a critical bug (EKF reset missing init_pos arg) that would
            crash on GPU. Fix and add warmup period for clean per-q_vel statistics.
Change:     (1) Fixed EKF reset in sweep script: now passes init_pos + init_vel (was missing).
            (2) Replaced crude diagnostic dict zeroing with proper pipeline.diagnostics flush.
            (3) Added --warmup-steps (default 50) to let EKF converge before collecting stats.
            (4) Added --no-post-contact flag to compare with/without post-contact inflation.
            (5) Killed 3 stale children (lit-review, vel-cmd-survey, report-writer) that
                survived iter_056 kill attempt.
Command:    CPU tests only (GPU locked by policy 2000-iter training from model_early_stop.pt).
            224/224 CPU tests pass. Script syntax verified.
Result:     Sweep script ready for GPU. Children cleaned up (agents: 2/5).
Decision:   Next iter: GPU q_vel sweep when lock frees. If still locked, consider writing
            a test that validates the sweep logic on CPU with synthetic data.

---

## Iteration 59 — compaction (summarized iters 049-055)  (2026-04-08T20:50:00Z)
Hypothesis: N/A — compaction iteration (context at 100% per system auto-compact trigger).
Change:     Archived iters 049-055 verbatim to RESEARCH_LOG_ARCHIVE.md. Rewrote compacted summary
            through iter_055 (including GPU NIS validation findings + live-policy eval critical
            result). Kept iters 056-058 verbatim. Pruned fix_plan.md (removed completed items).
Command:    No GPU commands.
Result:     Log shrunk from ~206 → ~110 lines. Archive now has 55 verbatim entries (001-055).
            fix_plan.md pruned to forward-looking tasks only.
Decision:   Next iter: GPU q_vel sweep with sweep_q_vel.py when GPU lock frees. Primary goal:
            find flight q_vel where NIS ≈ 3.0 and EKF RMSE < raw RMSE under active policy.
