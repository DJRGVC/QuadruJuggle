# fix_plan.md — experiment queue for perception

# Status: Sim pipeline FEATURE-COMPLETE (iters 001-035)
# 176/176 CPU tests pass, GPU NIS validated (1.60, in-band)
# IMU-aided EKF + 9D spin estimation implemented (iters 037-040)
# Velocity commands: Method 1 (override) + Method 2 (residual) + play_teleop.py done (iters 041-043)
# Lit-review + vel-cmd-survey subagents killed

# Phase 3: Real Hardware Integration (blocked on Go1 + D435i access)
- [x] Implement BallDetector Hough circle fallback (_detect_hough)
- [x] Implement CameraCalibrator.from_yaml()
- [x] Implement threaded RealPerceptionPipeline (camera+detector thread, EKF on main thread)
- [ ] Implement D435iCamera wrapper (pyrealsense2, depth-only 848x480 @ 90fps) — needs hardware
- [ ] Implement BallDetector._detect_yolo (TRT FP16 inference) — needs YOLO model
- [ ] Implement CameraCalibrator.from_checkerboard() — needs hardware
- [ ] YOLO training data collection + fine-tune on real Go1
- [ ] End-to-end hardware test: perception -> pi1 on real Go1

# Phase 5: Sim perception refinements
- [x] NIS chi-squared gating in BallEKF (reject outlier measurements)
- [x] Gate rejection stats in pipeline diagnostics + NIS diagnostic tool
- [ ] GPU NIS validation: IMU ON vs OFF comparison (blocked on GPU lock)
- [ ] GPU NIS validation: 9D spin mode (blocked on GPU lock)
- [ ] Support policy agent with noise curriculum tuning if/when they reach that stage

# Phase 6: User-defined velocity commands
- [ ] Policy agent handoff: Method 2 requires pi1 retraining (obs 40->42D, vel_tracking reward)
- [ ] Wire ResidualMixer into play_teleop.py as --mixer residual option (after policy trains M2)
