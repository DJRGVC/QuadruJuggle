# fix_plan.md — experiment queue for perception

# Status: ETH noise+EKF pipeline FEATURE-COMPLETE (294/294 CPU tests).
# EKF tuned: adaptive R_xy, 3-level q_vel, flight NIS≈3.3-3.8. q_vel=0.40 default.
# GAP: Sim camera visualization NOT validated on GPU. Convention fix (world) awaiting GPU.
# testing-dashboard child KILLED (task done iter 4, was idle at iter 10).

# PRIORITY: Sim camera visualization (Daniel's direct request 2026-04-09)
- [x] Fix D435i camera mount — 45° → 75° tilt (iter 76), ros → world convention (iter 77)
- [x] Write SimBallDetector for TiledCamera float32 depth (iter 77, 8 tests)
- [x] Integrate SimBallDetector into debug_d435i_capture.py (iter 77)
- [x] Sim pipeline integration tests — 8 tests (iter 82, 294/294 pass)
- [ ] GPU smoke test: capture RGB+depth frames with ball visible (blocked by policy PID 1118275)
- [ ] If camera still wrong: try identity quaternion, systematically rotate
- [ ] Wire camera→detect→EKF pipeline for visual demo (demo_camera_ekf.py ready, needs GPU)
- [ ] Demo script Daniel can run + capture video for Quarto
- [ ] Add video to Quarto page (Daniel requested 2026-04-09)

# EKF tuning (lower priority, mostly done)
- [ ] GPU sweep at higher target heights (0.3-0.5m) to confirm EKF beats raw at distance
- [ ] Re-run sweep with noise-trained pi1 (from policy agent) once it converges

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
