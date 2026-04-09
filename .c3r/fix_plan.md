# fix_plan.md — experiment queue for perception

# Status: ETH noise+EKF pipeline FEATURE-COMPLETE (565/565 CPU tests, 139 iterations).
# EKF tuned: 5-level phase-aware Q schedule, adaptive R_xy. q_vel_asc=0.25, q_vel_desc=0.40, q_vel_pre_landing=2.0.
# DRAG FIX: EKF now supports linear drag mode (matching PhysX). Sim pipeline defaults to linear.
# GPU DEMO VALIDATED: camera sees ball, SimBallDetector works, 100% det rate.
# Camera pipeline integrated into eval (demo_camera_ekf.py), paddle anchor, phase tracker, camera scheduling all working.
# Noise-to-gap prediction model: R²=0.994, VALIDATED against real policy Stage G data (max error 0.8pp).

## RE-VALIDATE — Gap prediction with corrected drag model ✅ (iter 140)
- [x] Re-run EKF error decomposition with linear drag ground truth
- [x] Compare: noise dominates, drag model choice has <11% effect on vz RMSE
- [x] Gap prediction model is analytical (not drag-dependent) — no re-run needed

## VALIDATED — Gap prediction matches policy Stage G eval
- [x] Policy agent retrained Stage G with fixed ES metric (iter 32)
- [x] Validated gap predictions against real eval data: R²=0.994, max error 0.8pp
- [ ] Policy agent plans to retrain Stage G longer (ES was killing early) — re-validate when available
- [ ] Run oracle-trained checkpoint through d435i pipeline (does it balance better?)

## WAITING — Policy Stage G retrain with fixed ES metric (longer training)
- [ ] Policy agent plans to retrain from Stage F checkpoint with fixed ES metric
- [ ] Expect improved 0.40-0.50m targets (currently 63-48% timeout under d435i)
- [ ] Re-validate gap predictions after longer training

## EKF tuning (when new policy checkpoint available)
- [ ] Re-run q_vel/R_xy sweep with noise-trained pi1 that actually juggles
- [ ] Evaluate EKF accuracy during true juggling (ball >200mm apex)

## Support policy agent
- [ ] Support policy agent with noise curriculum tuning (when requested)

## Real Hardware Integration (blocked on Go1 + D435i access)
- [x] D435iCamera wrapper (pyrealsense2, depth-only 848x480 @ 90fps) — iter 141
- [x] CameraCalibrator.from_checkerboard() — gravity-aligned via PnP + Wahba — iter 142
- [ ] BallDetector._detect_yolo (TRT FP16 inference)
- [ ] YOLO training data collection + fine-tune
- [ ] End-to-end hardware test

## Velocity commands (blocked on policy)
- [ ] Policy agent handoff: Method 2 requires pi1 retraining (obs 40->42D, vel_tracking reward)
- [ ] Wire ResidualMixer into play_teleop.py (after policy trains M2)
