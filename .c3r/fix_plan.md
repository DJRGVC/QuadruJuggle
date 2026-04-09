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
    Policy branch checkpoints should now be compatible. GPU validation needed.

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
