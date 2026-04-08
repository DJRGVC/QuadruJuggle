# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## Compacted summary (through iter_059)

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

**Iters 056-059 (post-contact q_vel + sweep prep):** Two compaction iters (056, 059). Killed all
3 children (lit-review, vel-cmd-survey, report-writer). Implemented 3-level q_vel architecture:
contact=50.0, post_contact=20.0 (10 steps), flight=0.4. Wrote sweep_q_vel.py with warmup,
post-contact toggle, and fixed EKF reset bug. 224/224 CPU tests pass. GPU sweep blocked by
policy agent training. Sweep ready to run: q_vel=[0.4,2.0,5.0,10.0,20.0,50.0].

**Key architectural findings through iter_059:**
- Contact-aware EKF: inflate Q during paddle contact → NIS from 966 to 0.78
- Ahn 2019 noise: sigma_xy=0.0025*z, sigma_z=1mm+0.005*z², dropout 20-50%
- Raw d435i noise outperforms EKF for training (noise = regularization)
- EKF value: velocity estimation + dropout bridging during free-flight
- IMU/spin: non-harmful in sim, ready for real hardware
- Velocity commands: Method 1 (override) done; Method 2 needs pi1 retraining
- **Active-policy EKF tuning is fundamentally different from random-action tuning**
- 3-level q_vel architecture: contact=50.0, post_contact=20.0 (10 steps), flight=TBD
- GPU q_vel sweep is next priority — find flight q_vel where NIS ≈ 3.0

---

## Iteration 60 — compaction (summarized iters 056-059)  (2026-04-08T21:15:00Z)
Hypothesis: N/A — compaction iteration (context at 80% per system auto-compact trigger).
Change:     Archived iters 056-059 verbatim to RESEARCH_LOG_ARCHIVE.md. Rewrote compacted summary
            through iter_059 (folded post-contact q_vel work + sweep prep). Processed 1 system
            INBOX message.
Command:    No GPU commands.
Result:     Log shrunk from ~127 → ~60 lines. Archive now has 59 verbatim entries (001-059).
Decision:   Next iter: GPU q_vel sweep with sweep_q_vel.py when GPU lock frees. Primary goal:
            find flight q_vel where NIS ≈ 3.0 and EKF RMSE < raw RMSE under active policy.
