# fix_plan.md — experiment queue for perception

# Status: Sim pipeline FEATURE-COMPLETE (iters 001-043)
# 239/239 tests pass (CPU). GPU NIS validated (random + live-policy).
# CRITICAL: EKF severely overconfident under active policy (NIS=52.9 flight, 19.9 overall).
# q_vel=0.4 good for random actions but WAY too low for juggling dynamics.

# PRIORITY: EKF Q-tuning for active juggling
- [ ] q_vel sweep: eval_perception_live.py with q_vel=[2.0, 5.0, 10.0, 20.0] @ target_height=0.10
- [ ] Find q_vel where flight NIS ≈ 3.0 and EKF RMSE < raw RMSE
- [ ] Consider dynamic q_vel (increase during post-bounce flight, decrease during stable arcs)
- [ ] Re-validate NIS with random actions at new q_vel (ensure not over-conservative)
- [ ] Update BallEKFConfig defaults with tuned values

# Phase 5: Sim perception refinements
- [x] GPU phase-separated NIS validation (flight=1.45, contact=0.43 with random actions)
- [x] eval_perception_live.py: script written + GPU-validated
- [x] GPU eval_perception_live.py run: EKF RMSE 22mm > raw 19mm (filter hurts, not helps!)
- [ ] Support policy agent with noise curriculum tuning

# Phase 7: Project Report
- [x] report-writer subagent: 1537-line HTML report complete (killed iter_055)
- [ ] Copy final report to main branch when ready

# Phase 3: Real Hardware Integration (blocked on Go1 + D435i access)
- [ ] Implement D435iCamera wrapper (pyrealsense2, depth-only 848x480 @ 90fps) — needs hardware
- [ ] Implement BallDetector._detect_yolo (TRT FP16 inference) — needs YOLO model
- [ ] Implement CameraCalibrator.from_checkerboard() — needs hardware
- [ ] YOLO training data collection + fine-tune on real Go1
- [ ] End-to-end hardware test: perception -> pi1 on real Go1

# Phase 6: User-defined velocity commands
- [ ] Policy agent handoff: Method 2 requires pi1 retraining (obs 40->42D, vel_tracking reward)
- [ ] Wire ResidualMixer into play_teleop.py as --mixer residual option (after policy trains M2)
