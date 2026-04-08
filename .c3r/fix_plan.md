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
- [ ] Check if policy agent needs perception support (noise curriculum tuning, new obs features)
