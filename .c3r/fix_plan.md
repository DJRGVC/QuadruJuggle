# fix_plan.md — experiment queue for perception

# Status: ETH noise+EKF pipeline FEATURE-COMPLETE (311/311 CPU tests).
# EKF tuned: adaptive R_xy, 3-level q_vel, flight NIS≈3.3-3.8. q_vel=0.40 default.
# GPU DEMO VALIDATED (iter 87): camera sees ball, SimBallDetector works, 100% det rate.

# PRIORITY: Sim camera visualization (Daniel's direct request 2026-04-09)
- [x] Fix D435i camera mount — 45° → 75° tilt (iter 76), ros → world convention (iter 77)
- [x] Write SimBallDetector for TiledCamera float32 depth (iter 77, 8 tests)
- [x] Integrate SimBallDetector into debug_d435i_capture.py (iter 77)
- [x] Sim pipeline integration tests — 8 tests (iter 82, 294/294 pass)
- [x] Pixel projection tests — 17 tests (iter 84, 311/311 pass; projection table validated)
- [x] GPU smoke test: capture RGB+depth frames with ball visible (iter 87 — 100% det rate)
- [x] Wire camera→detect→EKF pipeline for visual demo (iter 87 — summary.png + 60 frames)
- [x] Demo script Daniel can run + capture video for Quarto (iter 87 — run_gpu_demo.sh)
- [x] Add video to Quarto page (iter 87 — gpu_demo_iter087.mp4)
- [~] Wire camera pipeline into live eval with trained policy (demo_camera_ekf.py --pi1-checkpoint)
  - [x] Add --pi1-checkpoint flag + RSL-RL policy loading to demo_camera_ekf.py
  - [x] Multi-env EKF + per-env camera detection loop
  - [x] Episode tracking (timeouts, resets) with EKF re-init on done
  - [x] GPU smoke test with trained pi1 — pipeline works, but policy incompatible (iter 89)
  - [x] UNBLOCKED: synced env config (restitution=0.99, perceived obs, ball_low/release_vel rewards)
  - [x] GPU validation: policy loads + runs (iter 92). Env sync confirmed working.
  - [x] Full eval completed (run_full_eval.sh): oracle 1.7% det rate, d435i 4.5% det rate
    EKF diverges in both — ball on paddle 68-84% of episode at target=0.42m.
    FINDING: camera works at 200-300mm (100% det rate) but ball rarely reaches that height.
  - [x] Experiment write-up: experiments/perception/2026-04-10_oracle_vs_d435i_eval.qmd
  - [x] Fixed SameFileError in analyze_eval_trajectory.py --quarto-copy

# Next: Higher target height eval — DONE (iter 102)
- [x] Re-run eval at target=0.42/0.50/0.70/1.00m — policy caps at 0.25m apex regardless
- [x] ROOT CAUSE: policy is balancing (Stage F), not juggling. Camera works fine during flight.
- [ ] BLOCKED: need policy agent to produce a sustained-bouncing policy (Stage G+)
- [x] EKF covariance clamping for sparse-measurement regime (iter 103)
  - p_max_pos=0.25m, p_max_vel=5.0m/s prevents P divergence during long predict-only
  - steps_since_measurement counter for diagnostics
  - 8 new tests (test_p_clamping.py), 362/362 total
- [x] Paddle-anchor virtual measurement for contact phase (iter 104)
  - anchor_enabled=True, anchor_r_pos=5mm, min_starve=5 steps
  - Zeros velocity for anchored envs; reduces P during contact
  - 9 new tests (test_paddle_anchor.py), 371/371 total
- [ ] Wire paddle_anchor_update() into demo_camera_ekf.py eval loop
- [ ] Consider "flight window" detection mode — only run detection during known flight arcs

# EKF tuning (lower priority, mostly done)
- [x] GPU sweep at higher target heights (0.3-0.5m) — bounce mode sweep done (iter 90)
  - EKF diverges on artificial kicks; raw det wins above 200mm (190mm constant RMSE)
  - EKF wins at 0-200mm (near paddle, ballistic prediction accurate between measurements)
  - Real comparison needs trained policy where contact-aware Q triggers on paddle bounces
- [ ] Re-run sweep with noise-trained pi1 (from policy agent) once it converges
- [ ] Add trajectory npz analysis script for Quarto experiment write-up

# Support policy agent
- [ ] Support policy agent with noise curriculum tuning (when requested)

# Real Hardware Integration (blocked on Go1 + D435i access)
- [ ] D435iCamera wrapper (pyrealsense2, depth-only 848x480 @ 90fps)
- [ ] BallDetector._detect_yolo (TRT FP16 inference)
- [ ] CameraCalibrator.from_checkerboard()
- [ ] YOLO training data collection + fine-tune
- [ ] End-to-end hardware test

# Velocity commands
- [ ] Policy agent handoff: Method 2 requires pi1 retraining (obs 40->42D, vel_tracking reward)
- [ ] Wire ResidualMixer into play_teleop.py (after policy trains M2)
