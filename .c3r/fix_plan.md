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
- [ ] Oracle vs d435i vs EKF comparison: run compare_perception_modes.py (50 iters each, 4096 envs) — script ready, awaiting GPU
- [x] Handoff to policy agent: document how to enable EKF mode in env_cfg (swap noise_cfg, add reset event), provide example config diff — PERCEPTION_HANDOFF.md committed
- [x] Perception diagnostics: added _PerceptionDiagnostics to PerceptionPipeline (pos/vel RMSE, detection rate, EKF improvement %)
- [ ] Tune EKF parameters (process noise Q, measurement noise R) based on comparison results
- [ ] Noise curriculum support: parameterize noise model scaling so policy agent can ramp noise across stages
