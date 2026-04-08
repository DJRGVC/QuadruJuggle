# fix_plan.md — experiment queue for perception

# Status: Sim pipeline FEATURE-COMPLETE (iters 001-043)
# 239/239 tests pass (CPU). GPU NIS validated (random + live-policy).
# CRITICAL: EKF severely overconfident under active policy (NIS=52.9 flight, 19.9 overall).
# 3-level q_vel implemented (contact=50, post_contact=20, flight=TBD). sweep_q_vel.py ready.

# PRIORITY: EKF Q-tuning for active juggling
- [ ] GPU sweep: sweep_q_vel.py with q_vel=[0.4,2.0,5.0,10.0,20.0,50.0] @ target_height=0.10
- [ ] Find q_vel where flight NIS ≈ 3.0 and EKF RMSE < raw RMSE
- [ ] Re-validate NIS with random actions at new q_vel (ensure not over-conservative)
- [ ] Update BallEKFConfig defaults with tuned values

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
