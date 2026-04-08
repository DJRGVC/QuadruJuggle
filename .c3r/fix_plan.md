# fix_plan.md — experiment queue for perception
#
# Replace this preamble with 3-5 concrete starting tasks (one per line,
# as a markdown bullet). Lines starting with # are kept as comments.
# Agents read the TOP of this file at the start of every iteration.
#
# Example format:
#   - [ ] Task one — one full sentence, no line breaks
#   - [ ] Task two
#
# Save and exit your editor when done. Empty file = agent picks its own direction.

- [x] Audit the current docs/perception_roadmap.md against the D435i switch and commit updates noting the sensor change and its implications (depth instead of monocular, different noise model, easier 3D localization but slower frame rate); do NOT write code yet, just update the roadmap
- [x] Survey Isaac Lab's camera sensor API (RayCasterCamera, TiledCamera) and pick the one that best matches D435i semantics (RGB + depth + rate config); document the choice in perception/CAMERA_CHOICE.md with rationale
- [x] Create source/go1_ball_balance/perception/__init__.py and ball_obs_spec.py stub — the stub should return oracle ball state wrapped with an explicit zero-noise flag, committed to the c3r/quadrujuggle base branch so policy can start consuming it
- [x] Survey docs/sim_to_real_plan.md for the existing camera mount location assumptions and update for the rear-paddle-mounted 45-degree upward orientation; commit updates to docs
- [x] Mount a simulated D435i behind the paddle in source/**/*env_cfg.py scene setup (use Isaac Lab camera API), rendering at 30 Hz with RGB + depth, no noise yet; verify by taking a snapshot during a short ball-dropping episode and saving to perception/debug/frame_000.png
- [x] Run debug_d435i_capture.py smoke test when GPU is free — verified: RGB + depth frames saved to perception/debug/ (grey/black in headless, but pipeline works end-to-end)
- [x] Read Forrai et al. 2023 (event-based ANYmal badminton paper) if available in the repo, summarize their architectural choices in perception/REFERENCES.md, and flag the sensor-modality differences relative to our D435i — SKIPPED: paper not in repo; wrote REFERENCES.md with Ma et al. (primary ref), Ji et al., Portela et al., and noted Forrai modality differences
- [x] Implement ball_ekf.py — batched 6-state (pos+vel) Kalman filter in PyTorch, ballistic+drag dynamics model, runs on GPU across all envs
- [x] Implement noise_model.py — D435i-structured noise sampling (depth-dependent σ, dropout, latency) consumed by ball_obs_spec.py d435i mode
- [x] Wire EKF mode into ball_obs_spec.py — PerceptionPipeline class (noise→EKF→filtered obs), lazy init on env, reset event, idempotent step dedup
- [x] Integration test: swap ball_juggle_hier obs to mode="ekf", run short training (50 iters) — PASSED, mean_len=123 at iter 50, pipeline stable
- [x] Oracle vs d435i vs EKF comparison: 2048 envs × 50 iters — oracle 294/22.0, d435i 317/20.1, ekf 279/19.3 (ep_len/reward). EKF 12% below oracle on reward. Fixed critical vel-view covariance bug.
- [x] Body-frame gravity in EKF: pass projected_gravity_b * 9.81 into EKF.predict() to account for trunk tilt
- [x] Handoff to policy agent: document how to enable EKF mode in env_cfg (swap noise_cfg, add reset event), provide example config diff — PERCEPTION_HANDOFF.md committed
- [x] Perception diagnostics: added _PerceptionDiagnostics to PerceptionPipeline (pos/vel RMSE, detection rate, EKF improvement %)
- [x] Tune EKF parameters (process noise Q, measurement noise R) based on CWNA analysis + lit-review — q_pos 0.01→0.003, q_vel 1.0→0.15, r_xy 0.003→0.002, time-varying r_z, ANEES diagnostic added
- [x] Run tuned EKF comparison test (oracle/d435i/ekf, 2048 envs × 50 iters) — DONE: oracle 13.7, d435i 10.5, ekf 7.6. EKF trails d435i; may be over-smoothing. Fixed pi2 obs dim bug (41→53).
- [ ] Run NIS diagnostic: `gpu_lock.sh uv run --active python scripts/perception/nis_diagnostic.py --pi2-checkpoint .../2026-03-12_17-16-01/model_best.pt --num_envs 2048 --steps 500 --headless`
- [ ] If NIS too low (< 0.35): increase q_vel from 0.15 toward 0.5 and re-run
- [ ] If NIS too high (> 7.81): decrease q_vel or increase r_xy/r_z
- [x] Fix compare_perception_modes.py diagnostic capture — used base_env reference, added flush + warning
- [x] Noise curriculum support: `noise_scale` field on BallObsNoiseCfg + `update_perception_noise_scale()` for runtime curriculum updates — DONE, 6 unit tests pass
- [x] Monitor lit-review subagent — 3 iterations complete, all 3 docs committed (perception, noise_curriculum, ekf_tuning). Attempted kill (c3r binary path issue).
- [x] Review lit-review findings — EKF Q/R values applied this iter; ANEES diagnostic from Bar-Shalom Ch. 5 added; Sage-Husa deferred to real deployment
