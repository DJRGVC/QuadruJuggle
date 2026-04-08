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
- [ ] IMU-aided EKF: use robot angular velocity to compensate platform motion during predict
- [ ] Spin estimation: extend state to [x,y,z,vx,vy,vz,wx,wy,wz] for Magnus effect
- [ ] Support policy agent with noise curriculum tuning if/when they reach that stage
