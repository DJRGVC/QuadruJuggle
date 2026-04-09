# fix_plan.md — experiment queue for perception

# Status: ETH noise+EKF pipeline FEATURE-COMPLETE (558/558 CPU tests, 132 iterations).
# EKF tuned: adaptive R_xy, 3-level q_vel, flight NIS≈3.3-3.8. q_vel=0.40 default.
# GPU DEMO VALIDATED: camera sees ball, SimBallDetector works, 100% det rate.
# Camera pipeline integrated into eval (demo_camera_ekf.py), paddle anchor, phase tracker, camera scheduling all working.
# Noise-to-gap prediction model: R²=0.865, gap is noise-expected not pipeline-broken.

## WAITING — Policy Stage G retrain with ES metric fix
- [ ] Policy agent needs to retrain Stage G with fixed ES metric (per-step reward, not total episode return)
- [ ] Once new checkpoint available: re-run perception eval at 0.10-0.50m targets
- [ ] Compare gap against iter 131 noise-to-gap predictions (expected: gap narrows if policy improves energy modulation)
- [ ] Run oracle-trained checkpoint through d435i pipeline (does it balance better?)

## EKF tuning (when new policy checkpoint available)
- [ ] Re-run q_vel/R_xy sweep with noise-trained pi1 that actually juggles
- [ ] Evaluate EKF accuracy during true juggling (ball >200mm apex)

## Support policy agent
- [ ] Support policy agent with noise curriculum tuning (when requested)

## Real Hardware Integration (blocked on Go1 + D435i access)
- [ ] D435iCamera wrapper (pyrealsense2, depth-only 848x480 @ 90fps)
- [ ] BallDetector._detect_yolo (TRT FP16 inference)
- [ ] CameraCalibrator.from_checkerboard()
- [ ] YOLO training data collection + fine-tune
- [ ] End-to-end hardware test

## Velocity commands (blocked on policy)
- [ ] Policy agent handoff: Method 2 requires pi1 retraining (obs 40->42D, vel_tracking reward)
- [ ] Wire ResidualMixer into play_teleop.py (after policy trains M2)
