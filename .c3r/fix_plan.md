# fix_plan.md — experiment queue for perception

# Status: Sim pipeline FEATURE-COMPLETE (iters 001-043)
# 239/239 tests pass (CPU). GPU NIS validated.
# IMU-aided EKF + 9D spin + contact-aware + NIS gating all implemented.
# Velocity commands: Method 1 (override) + Method 2 (residual) + play_teleop.py done.
# Subagents: lit-review killed, vel-cmd-survey killed, report-writer spawned (sonnet, max 10 iters).

# Phase 5: Sim perception refinements
- [ ] GPU phase-separated NIS validation: verify free-flight NIS closer to 3.0 (blocked: GPU held by policy training)
- [ ] Support policy agent with noise curriculum tuning if/when they reach that stage

# Phase 7: Project Report (report-writer subagent)
- [ ] Monitor report-writer progress, provide info as needed
- [ ] Kill report-writer after max 10 iters or when report is done
- [ ] Review and polish final report

# Phase 3: Real Hardware Integration (blocked on Go1 + D435i access)
- [ ] Implement D435iCamera wrapper (pyrealsense2, depth-only 848x480 @ 90fps) — needs hardware
- [ ] Implement BallDetector._detect_yolo (TRT FP16 inference) — needs YOLO model
- [ ] Implement CameraCalibrator.from_checkerboard() — needs hardware
- [ ] YOLO training data collection + fine-tune on real Go1
- [ ] End-to-end hardware test: perception -> pi1 on real Go1

# Phase 6: User-defined velocity commands
- [ ] Policy agent handoff: Method 2 requires pi1 retraining (obs 40->42D, vel_tracking reward)
- [ ] Wire ResidualMixer into play_teleop.py as --mixer residual option (after policy trains M2)
