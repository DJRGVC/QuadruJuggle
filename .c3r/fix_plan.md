# fix_plan.md — experiment queue for perception

# Status: Sim pipeline FEATURE-COMPLETE (iters 001-043)
# 239/239 tests pass (CPU). GPU NIS validated (random + live-policy).
# CRITICAL: EKF severely overconfident under active policy (NIS=52.9 flight, 19.9 overall).
# 3-level q_vel implemented (contact=50, post_contact=20, flight=TBD). sweep_q_vel.py ready.

# PRIORITY: EKF Q-tuning for active juggling
- [x] GPU sweep: coarse q_vel=[0.4-50] — DONE. All flight NIS < 3.0 (max 1.73).
      EKF over-conservative; RMSE improvement only 1-2% over raw d435i.
- [ ] GPU sweep: low-range q_vel=[0.01-0.4] to find NIS=3.0 crossing
- [ ] Read combined sweep results, find optimal q_vel (or accept NIS < 3 if divergence risk)
- [ ] Re-validate NIS with random actions at new q_vel (ensure not over-conservative)
- [ ] Update BallEKFConfig defaults with tuned values
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
