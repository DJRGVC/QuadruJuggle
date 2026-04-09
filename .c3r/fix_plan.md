# fix_plan.md — experiment queue for perception

# Status: Sim pipeline FEATURE-COMPLETE (iters 001-043)
# 239/239 tests pass (CPU). GPU NIS validated (random + live-policy).
# CRITICAL: EKF severely overconfident under active policy (NIS=52.9 flight, 19.9 overall).
# 3-level q_vel implemented (contact=50, post_contact=20, flight=TBD). sweep_q_vel.py ready.

# PRIORITY: EKF Q-tuning for active juggling
- [x] GPU sweep: coarse q_vel=[0.4-50] — DONE (pre-adaptive R). All flight NIS < 3.0.
- [x] Root cause: R_xy non-adaptive. Fixed: σ_xy = max(0.0025·z, 0.0005m).
- [x] GPU sweep: low-range q_vel=[0.01-0.4] with adaptive R — DONE.
      All flight NIS now 3.3-3.8 (correctly calibrated). q_vel=0.40 closest to NIS=3.0.
      EKF loses to raw at Stage A (expected at z≈0.1m). Default q_vel=0.40 unchanged.
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
