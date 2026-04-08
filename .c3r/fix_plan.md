# fix_plan.md — experiment queue for perception

# Sim perception pipeline: FEATURE-COMPLETE (iters 001-028)
# Real hardware pipeline: stubs + mocks + unit tests DONE
# Remaining work is hardware-blocked or policy-support

# Phase 3: Real Hardware Integration (blocked on Go1 + D435i access)
- [ ] Implement D435iCamera wrapper (pyrealsense2, depth-only 848×480 @ 90fps, non-blocking poll)
- [ ] Implement BallDetector (YOLOv8n+P2 TRT FP16 inference + median-depth-in-bbox 3D localisation)
- [ ] Implement CameraCalibrator.from_yaml() and from_checkerboard() (hardware-blocked)
- [ ] Implement RealPerceptionPipeline (threaded: camera+YOLO at 90Hz, EKF predict at 200Hz, get_observation at 50Hz)
- [ ] YOLO training data collection on real Go1 (300+ depth frames, auto-label + CVAT verify)
- [ ] YOLO fine-tune: YOLOv8n+P2, frozen backbone, 150 epochs, export TRT FP16 on Orin NX
- [ ] End-to-end hardware test: perception pipeline → pi1 policy on real Go1 with ball drop

# Phase 4: Sim-side improvements (if policy agent needs or hardware wait is long)
- [x] Contact-aware EKF: detect contact phase (ball on paddle) and inflate q_vel (0.40→50.0) — DONE iter_030
- [x] GPU NIS validation: contact_aware=True NIS=0.78 (in-band), OFF NIS=671 (divergent) — 860× improvement — DONE iter_031
- [x] Ballistic trajectory simulation in mock pipeline (parabolic arcs, not just stationary/linear) — DONE iter_032
- [x] Latency injection testing: 16 tests for buffer correctness, RMSE degradation, dropout+latency combo — DONE iter_033
- [x] Ahn 2019-calibrated noise model: σ_xy∝z, σ_z∝z², distance-dependent dropout 20→50% — DONE iter_034
- [ ] Check if policy agent needs perception support (noise curriculum tuning, new obs features)

# Phase 5: Sim perception refinements (new — while waiting for policy/hardware)
- [ ] IMU-aided EKF: use robot angular velocity to compensate platform motion during prediction step
- [ ] Spin estimation: extend EKF state to [x,y,z,vx,vy,vz,wx,wy,wz] for Magnus effect in flight
- [ ] GPU NIS re-validation with calibrated noise model (verify NIS still in-band with new σ values)
