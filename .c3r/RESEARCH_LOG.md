# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## Compacted summary (through iter_048)

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

**Iters 036-040 (extensions + wiring):** IMU-aided EKF (Coriolis corrections), 9D spin
estimation (Magnus force), config wiring through BallObsNoiseCfg. Spawned vel-cmd-survey.

**Iters 041-043 (velocity commands):** vel_cmd/ package: UserVelocityInput, CommandMixer,
ResidualMixer. play_teleop.py. Killed vel-cmd-survey. 176/176 tests.

**Iters 044-048 (real pipeline + NIS tooling):** Hough circle fallback detector (<10mm at
0.3-1.0m). Threaded RealPerceptionPipeline with MockCamera integration tests. Chi-squared
NIS gating (11.345 threshold, 50-step warmup). Gate rejection stats in diagnostics.
239/239 tests. Two compaction iterations (iters 036, 044).

**Key architectural findings through iter_048:**
- Contact-aware EKF: inflate Q during paddle contact → NIS from 966 to 0.78
- Ahn 2019 noise: sigma_xy=0.0025*z, sigma_z=1mm+0.005*z², dropout 20-50%
- Raw d435i noise outperforms EKF for training (noise = regularization)
- EKF value: velocity estimation + dropout bridging during free-flight
- IMU/spin: non-harmful in sim, ready for real hardware
- Velocity commands: Method 1 (override) done; Method 2 needs pi1 retraining

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
Change:     Killed lit-review and vel-cmd-survey. Spawned report-writer (sonnet, max 10 iters).
            GPU locked by policy training (12288-env, 1500-iter run).
Result:     Agents: 3/5 (perception, policy, report-writer). GPU unavailable.
Decision:   GPU NIS phase-separated validation next iter.

---

## iter_052 — nis_sweep phase-separated output + test verification  (2026-04-08T18:10:00Z)
Hypothesis: Phase-separated columns in nis_sweep.py make Q-tuning sweeps actionable.
Change:     Updated nis_sweep.py with flight/contact NIS columns. 239/239 tests pass (CPU).
Result:     GPU still locked by policy (iter_016 training).
Decision:   GPU NIS phase-separated validation FIRST PRIORITY next iter.

---

## iter_053 — eval_perception_live.py: EKF accuracy under trained policy  (2026-04-08T19:30:00Z)
Hypothesis: EKF accuracy under random actions may differ from accuracy during juggling.
Change:     Wrote eval_perception_live.py — loads pi1 checkpoint, runs env with EKF mode,
            logs phase-separated NIS, RMSE, gate rejections, episode stats.
Result:     Script ready (276 lines). GPU blocked by policy training.
Decision:   GPU eval with policy checkpoint when GPU frees up.

---

## iter_054 — eval_perception_live.py improvements: JSON output + RMSE tracking  (2026-04-08T21:20:00Z)
Hypothesis: JSON output and per-interval RMSE tracking enable systematic parameter sweeps.
Change:     Added --output JSON, per-interval EKF/raw RMSE tracking. 239/239 tests pass.
Result:     Script improved. Policy checkpoints available at
            QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_10-36-50/.
Decision:   GPU phase-separated NIS + eval_perception_live.py run NEXT.

---

## iter_055 — GPU NIS validation: phase-separated + live-policy eval  (2026-04-08T18:57:00Z)
Hypothesis: Phase-separated NIS under random actions will confirm contact-phase Q inflation as
            the cause of low overall NIS=0.44. Live-policy eval will reveal whether q_vel=0.4
            is adequate for actual juggling dynamics.
Change:     (1) Killed report-writer (completed, 1537-line HTML report).
            (2) Ran nis_diagnostic.py (2048 envs × 500 steps, random actions).
            (3) Ran eval_perception_live.py (512 envs × 1000 steps, trained pi1, target=0.10m).
Result:     **CRITICAL FINDING: EKF severely overconfident under active policy.**

            Random-action NIS (confirms iter_050 hypothesis):
            | Phase   | NIS   | Diagnosis                              |
            |---------|-------|----------------------------------------|
            | Flight  | 1.454 | Well-tuned (target 3.0)                |
            | Contact | 0.434 | Over-conservative (q_vel_contact=50.0) |
            | Overall | 0.444 | Dominated by contact phase             |

            Live-policy eval (TARGET HEIGHT 0.10m, trained pi1):
            | Phase   | NIS    | Diagnosis                              |
            |---------|--------|----------------------------------------|
            | Flight  | 52.9   | SEVERELY overconfident (q_vel=0.4 too low) |
            | Contact | 5.3    | Slightly overconfident                 |
            | Overall | 19.9   | Way above target 3.0                   |

            EKF RMSE: 22.0mm vs Raw: 19.3mm → EKF is WORSE than raw (-14%)!
            Gate rejections: 1.26% (2281/181k). Timeout: 0% (ball off/below).

            **Root cause**: Active juggling produces violent paddle strikes with near-instantaneous
            velocity reversals. q_vel=0.4 trusts prediction too much → filter lags behind.
Decision:   q_vel needs MAJOR increase for live policy. Next: sweep q_vel=[2.0, 5.0, 10.0, 20.0]
            with eval_perception_live.py to find optimal setting. Contact q_vel_contact=50.0
            is close to right (NIS=5.3 under active policy).

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
