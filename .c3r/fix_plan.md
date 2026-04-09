# fix_plan.md — experiment queue for perception

# Status: ETH noise+EKF pipeline FEATURE-COMPLETE (256/256 CPU tests).
# EKF tuned: adaptive R_xy, 3-level q_vel, flight NIS≈3.3-3.8. q_vel=0.40 default.
# GAP: Sim camera visualization NOT working. No camera→detect→EKF loop.

# PRIORITY: Sim camera visualization (Daniel's direct request 2026-04-09)
- [x] Fix D435i camera mount — 45° → 75° tilt, moved pos from (-0.05,0,0.08) to (-0.08,0,0.06)
- [ ] GPU smoke test: capture RGB+depth frames with ball visible (blocked by policy training)
      Debug script updated: applies 2m/s upward ball velocity before capture.
- [ ] Simple sim ball detector (from camera depth/RGB) — Hough circle or threshold
- [ ] Wire camera→detect→EKF pipeline for visual demo
- [ ] Demo script Daniel can run: camera feed + ball detection + EKF overlay

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
