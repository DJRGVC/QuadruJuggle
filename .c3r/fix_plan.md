# fix_plan.md — experiment queue for perception

# Status: Sim pipeline FEATURE-COMPLETE (iters 001-035)
# 74/74 CPU tests pass, GPU NIS validated (1.60, in-band)
# Real hardware stubs + mocks + unit tests DONE
# Lit-review subagent killed after 29 iters (~20 survey docs)

# Phase 3: Real Hardware Integration (blocked on Go1 + D435i access)
- [ ] Implement D435iCamera wrapper (pyrealsense2, depth-only 848x480 @ 90fps)
- [ ] Implement BallDetector (YOLOv8n+P2 TRT FP16 + median-depth 3D localisation)
- [ ] Implement CameraCalibrator.from_yaml() and from_checkerboard()
- [ ] Implement RealPerceptionPipeline (threaded: camera+YOLO 90Hz, EKF 200Hz, obs 50Hz)
- [ ] YOLO training data collection + fine-tune on real Go1
- [ ] End-to-end hardware test: perception -> pi1 on real Go1

# Phase 5: Sim perception refinements (while waiting for policy/hardware)
- [x] IMU-aided EKF: use robot angular velocity to compensate platform motion during predict (iter_037)
- [x] Spin estimation: extend state to [x,y,z,vx,vy,vz,wx,wy,wz] for Magnus effect (iter_038)
- [x] Wire enable_spin through BallObsNoiseCfg → PerceptionPipeline → EKF (iter_039)
- [ ] GPU NIS validation: body-frame+IMU vs world-frame comparison (blocked on GPU lock)
- [ ] Support policy agent with noise curriculum tuning if/when they reach that stage

# Phase 6: User-defined velocity input (Daniel request 2026-04-08)
- [ ] vel-cmd-survey subagent: 5 iters MAX, then kill. Researching 2023-2026 papers on velocity commands for legged manipulation
- [x] Implement Method 1 (Direct Override): UserVelocityInput + CommandMixer (iter_041)
- [x] Create play_teleop.py integration script (hooks mixer into play loop) (iter_042)
- [ ] Decide on Method 2/3 based on subagent findings (for future retraining approach)
