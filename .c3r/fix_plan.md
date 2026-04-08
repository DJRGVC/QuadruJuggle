# fix_plan.md — experiment queue for perception

# Status: Sim pipeline FEATURE-COMPLETE (iters 001-043)
# 239/239 tests pass (CPU). GPU NIS validated (random + live-policy).
# CRITICAL: EKF severely overconfident under active policy (NIS=52.9 flight, 19.9 overall).
# 3-level q_vel implemented (contact=50, post_contact=20, flight=TBD). sweep_q_vel.py ready.

# PRIORITY: EKF Q-tuning for active juggling
- [x] GPU sweep: sweep_q_vel.py with bisection — QUEUED behind policy GPU lock (PID 806215)
      Results auto-save to logs/perception/sweep_q_vel_*.json
- [ ] Read sweep results (apply_sweep_results.py), find q_vel where flight NIS ≈ 3.0
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
