# Hardware Perception Pipeline Architecture

**Author**: perception agent  
**Date**: 2026-04-08  
**Status**: Specification — not yet implemented  
**Synthesises**: lit-review iters 001–026, perception iters 001–024

---

## Overview

This document specifies the real-time perception pipeline for hardware deployment
of the QuadruJuggle ball-juggling system. The pipeline transforms D435i depth
frames into the same `(ball_pos_b, ball_vel_b)` body-frame observations that
pi1 consumes in simulation, enabling zero-change policy deployment.

**Architecture**: D435i depth → ball detection → camera-to-world transform →
world-frame EKF → world-to-body transform → pi1 observations.

**Key constraint**: the entire pipeline must complete within one policy step
(20ms at 50Hz). Camera runs at 90Hz, policy at 50Hz, EKF predict at 200Hz.

---

## 1. Pipeline Timing Budget (20ms per policy step)

| Stage | Budget | Notes |
|---|---|---|
| D435i depth acquisition | ~11ms | 848×480 @ 90Hz, hardware-pipelined |
| Ball detection (YOLO) | 3–4ms | YOLOv8n+P2, TRT FP16 on Orin NX |
| Depth lookup + outlier rejection | <0.5ms | Median of depth pixels in bbox |
| Camera→world transform | <0.1ms | IMU quaternion + extrinsics |
| EKF update | <0.1ms | Single-env, 6×6 matrix ops |
| **Total** | **~15ms** | **5ms margin** |

Between policy steps, the EKF runs 3–4 predict-only steps at 200Hz using
ballistic dynamics (no measurement). This provides smooth velocity estimates
even when the camera measurement rate (90Hz) doesn't align with policy rate.

---

## 2. File Layout

```
source/go1_ball_balance/go1_ball_balance/perception/
├── __init__.py              # (existing) sim pipeline exports
├── ball_ekf.py              # (existing) SHARED between sim and real
├── ball_obs_spec.py         # (existing) sim-only: oracle + noise
├── noise_model.py           # (existing) sim-only: D435i noise sampling
├── real/                    # NEW: hardware deployment code
│   ├── __init__.py
│   ├── camera.py            # D435i driver wrapper (pyrealsense2)
│   ├── detector.py          # Ball detection (YOLO + depth lookup)
│   ├── calibration.py       # Camera extrinsics + IMU alignment
│   ├── pipeline.py          # Real-time pipeline orchestrator
│   └── config.py            # Hardware-specific parameters
```

---

## 3. Component Specifications

### 3.1 Camera Driver (`real/camera.py`)

Wraps `pyrealsense2` for depth-only streaming. **No ROS2** in the hot path
(adds 2–8ms latency per frame — see lit_review_d435i_ros2_integration.md §1).

```python
class D435iCamera:
    """Non-blocking D435i depth stream at 848×480 @ 90fps."""

    def __init__(self, serial: str | None = None):
        """Init pipeline. serial=None uses first available device."""

    def start(self) -> None:
        """Enable depth stream. Blocks until first frame."""

    def get_frame(self) -> tuple[np.ndarray, float] | None:
        """Non-blocking poll. Returns (depth_u16, timestamp_s) or None."""

    def stop(self) -> None:
        """Release device."""
```

**Stream config** (from lit_review_d435i_ros2_integration.md §2):
- Resolution: **848×480** (only mode where MinZ ≤ 195mm — ball at 0.28m is valid)
- Format: `z16` (16-bit depth in mm)
- FPS: **90** (requires firmware ≥ v5.13)
- **Depth only** — do NOT enable color (adds 66ms hardware pipeline delay)
- Post-processing: decimation=1 (no downsampling), spatial filter disabled
  (YOLO bbox is already the ROI), temporal filter disabled (adds latency)

**Known Orin NX issues**:
- Use USB-C port at 5V/3A (USB-A at 4.5W causes frame drops)
- No kernel patches needed for JetPack 6.x

### 3.2 Ball Detector (`real/detector.py`)

Two-stage: YOLO bbox detection → depth-based 3D localisation.

```python
class BallDetector:
    """Detect 40mm ping-pong ball in D435i depth frame."""

    def __init__(self, model_path: str, conf_thresh: float = 0.5):
        """Load YOLOv8n+P2 TensorRT engine."""

    def detect(
        self, depth_frame: np.ndarray, intrinsics: CameraIntrinsics
    ) -> Detection | None:
        """
        Returns Detection(pos_cam=[x,y,z], confidence, bbox) or None.
        pos_cam is in camera optical frame (Z forward, X right, Y down).
        """
```

**YOLO configuration** (from lit_review_yolo_ball_detection.md):
- Model: YOLOv8n + P2 detection head (160×160 feature map for small objects)
- Training: 300+ images, frozen backbone (9-block freeze), 150 epochs,
  imgsz=640, copy_paste=0.3, hsv_s=0.9, hsv_v=0.6
- Export: TensorRT FP16 on Orin NX (`yolo export format=engine half=True`)
- Expected: mAP@0.5 ≈ 0.82–0.90 with 300–400 domain-specific images
- Inference: **2–4ms** TRT FP16 on Orin NX 16GB

**Depth lookup** (from lit_review_realsense_d435i_noise.md):
- Extract depth pixels within YOLO bbox
- **Median-over-mask**: take median of valid (>0) depth pixels within bbox
  (not mean — single outlier from background can shift mean by 50mm+)
- Convert pixel (u, v, depth_mm) → camera-frame 3D point using D435i intrinsics
- Reject if depth < MinZ (168mm) or > 2000mm (ball can't be that far)

**Fallback when YOLO confidence < 0.4**: Hough circle on depth frame ROI
(from lit_review_d435i_ros2_integration.md §3). This handles the case where
fine-tuned YOLO misses a frame but the ball is still clearly visible in depth.

### 3.3 Camera Calibration (`real/calibration.py`)

```python
@dataclass
class CameraExtrinsics:
    """Camera-to-body (IMU) rigid transform."""
    R_cam_body: np.ndarray  # (3,3) rotation: camera frame → body frame
    t_cam_body: np.ndarray  # (3,) translation: camera origin in body frame

class CameraCalibrator:
    """Compute and store camera-to-body extrinsics."""

    @staticmethod
    def from_yaml(path: str) -> CameraExtrinsics:
        """Load pre-calibrated extrinsics."""

    @staticmethod
    def calibrate_with_checkerboard(
        camera: D435iCamera, num_frames: int = 30
    ) -> CameraExtrinsics:
        """Interactive calibration using checkerboard pattern."""
```

**Nominal extrinsics** (from sim scene setup):
- Camera mounted on Go1 back, upward-facing, behind paddle
- Optical axis ≈ vertical (within ±5°) at nominal standing height
- Body-frame offset: approximately (0.0, 0.0, 0.05) relative to trunk root
- Rotation: camera Z-forward maps to body Z-up (180° rotation about X-axis)

**IMU alignment**: D435i has a built-in BMI055 IMU. For world-frame EKF, we
need `R_body_world` at each timestep. Options:
1. Use Go1's onboard IMU (preferred — same frame as joint state)
2. Use D435i IMU (requires additional camera-to-body calibration)
3. Fuse both (overkill for this application)

### 3.4 Real-Time Pipeline (`real/pipeline.py`)

Orchestrates camera → detection → EKF → policy observations.

```python
class RealPerceptionPipeline:
    """Real-time perception pipeline for hardware deployment.

    Runs on a separate thread. Policy queries latest estimate synchronously.
    """

    def __init__(self, config: HardwarePipelineConfig):
        """Init camera, detector, EKF, calibration."""

    def start(self) -> None:
        """Start camera stream and processing thread."""

    def get_observation(
        self, robot_quat_w: np.ndarray, robot_pos_w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Called by policy at 50Hz.

        Args:
            robot_quat_w: (4,) robot orientation in world frame (from IMU)
            robot_pos_w: (3,) robot position in world frame

        Returns:
            ball_pos_b: (3,) ball position in body frame
            ball_vel_b: (3,) ball velocity in body frame
            detected: bool — True if ball detected this frame
        """

    def stop(self) -> None:
        """Stop processing thread and release camera."""
```

**Threading model**:
- Camera acquisition + YOLO detection run on a dedicated thread at 90Hz
- EKF predict runs at 200Hz on the main (policy) thread between measurements
- Policy calls `get_observation()` at 50Hz — returns latest EKF estimate
  transformed to body frame using current robot orientation

**EKF configuration for deployment** (from iter_024 findings):
- `world_frame=True` (mandatory — body-frame EKF is structurally broken)
- `q_vel=7.0` (covers contact-force uncertainty during paddle contact)
- `q_pos=0.003`, `r_xy=0.002`, `r_z=0.004+0.002/m`
- NIS gate: reject measurement if NIS > 11.35 (χ²(3, p=0.99))
- Covariance trace flag: `ball_lost=True` when `trace(P[:3,:3]) > 0.001 m²`
- On lost ball (>5 consecutive dropouts): recenter EKF to paddle centre with
  high uncertainty (from lit_review_ekf_dropout_fallback.md)

### 3.5 Hardware Config (`real/config.py`)

```python
@dataclass
class HardwarePipelineConfig:
    """All hardware-specific parameters in one place."""

    # Camera
    camera_serial: str | None = None
    depth_width: int = 848
    depth_height: int = 480
    depth_fps: int = 90

    # Detector
    yolo_engine_path: str = "models/ball_detector.engine"
    yolo_conf_thresh: float = 0.5
    hough_fallback: bool = True

    # Calibration
    extrinsics_path: str = "config/camera_extrinsics.yaml"

    # EKF (reuses BallEKFConfig from ball_ekf.py)
    ekf_predict_hz: int = 200
    nis_gate_thresh: float = 11.35  # chi2(3, 0.99)
    lost_ball_frames: int = 5       # consecutive dropouts before recenter
    covariance_trace_thresh: float = 0.001  # m² — ball_lost_flag trigger

    # Policy interface
    policy_hz: int = 50
```

---

## 4. Shared Code: ball_ekf.py

The EKF (`ball_ekf.py`) is designed to run identically in sim and on real
hardware. In sim, it's batched (N envs on GPU). On real hardware, it's
single-env (N=1, CPU or GPU).

**No changes needed** to `ball_ekf.py` for hardware deployment. The world-frame
mode (`world_frame=True`) was validated in sim (iter_023–024) and uses the same
`_body_to_world_pos/vel` and `_world_to_body_pos/vel` transforms that the real
pipeline will provide via IMU data.

---

## 5. Noise Model Validation (sim ↔ real)

The sim noise model (`noise_model.py`) should match real D435i characteristics.
Current audit findings (from lit_review_realsense_d435i_noise.md):

| Parameter | Sim (noise_model.py) | Real D435i (848×480) | Match? |
|---|---|---|---|
| σ_xy base | 2mm | 1.5–3mm | ✓ |
| σ_z base | 3mm | 2–4mm @ 0.5m | ✓ |
| σ_z per metre | 2mm/m | **5–8mm/m** (quadratic) | **2.5× too low** |
| Dropout prob | 2% | **8–40%** (white ball, IR) | **4–20× too low** |
| Latency | 1 step (20ms) | ~11ms (depth pipeline) | ✓ (conservative) |

**Action items** (for next phase):
1. Increase `sigma_z_per_metre` to 5mm/m (match real D435i quadratic curve)
2. Increase `dropout_prob` to 0.10 for training (white ball has high IR
   reflectance → frequent dropouts on D435i depth stream)
3. Validate with real D435i data capture before production training

---

## 6. Data Collection Plan (Pre-Training)

Before fine-tuning YOLO, collect 300+ images:

1. **Mount D435i on Go1** in final hardware position
2. **Run Go1 standing controller** (pi2 only, no ball)
3. **Drop ball from various heights** (10cm, 30cm, 50cm, 80cm, 1m)
4. **Record depth frames** at 90fps for 30s per height → ~2700 frames per height
5. **Auto-label**: use depth circularity filter (ball is the only round object
   in upward-facing view) to generate candidate bboxes
6. **Manual verify** in CVAT: expect ~70% of auto-labels are correct
7. **Split**: 80/10/10 train/val/test

Expected dataset: ~300 verified images, covering ball at 10–50px apparent size
across the 0.1–1.5m depth range.

---

## 7. Deployment Sequence

1. **Pi2 standalone on Go1** — verify standing + torso tracking with oracle obs
2. **D435i capture** — collect dataset (Section 6)
3. **YOLO training** — fine-tune YOLOv8n+P2, export TRT FP16
4. **Camera calibration** — checkerboard routine (Section 3.3)
5. **Perception pipeline integration** — `RealPerceptionPipeline` on Orin NX
6. **EKF tuning on real data** — run pipeline, log NIS, adjust Q/R if needed
7. **Pi1 deployment** — connect real perception to pi1 policy (same obs interface)
8. **Ball dropping test** — no juggling, just perception tracking accuracy
9. **Full juggling test** — pi1 + pi2 + real perception

---

## 8. Interface Compatibility

The real pipeline outputs the same `(ball_pos_b, ball_vel_b)` as the sim
pipeline. Pi1 doesn't need to know whether it's running in sim or on hardware.

| Observation | Sim (ball_obs_spec.py) | Real (real/pipeline.py) | Same? |
|---|---|---|---|
| ball_pos_b (3D) | GT + D435i noise | D435i depth + YOLO | ✓ (same frame) |
| ball_vel_b (3D) | Finite-diff of noisy pos | EKF velocity estimate | ✓ (body frame) |
| ball_lost_flag | Dropout mask | NIS gate + cov trace | ✓ (binary) |

---

## References

- `docs/lit_review_d435i_ros2_integration.md` — pyrealsense2 setup, stream config
- `docs/lit_review_yolo_ball_detection.md` — YOLOv8n+P2 fine-tuning recipe
- `docs/lit_review_realsense_d435i_noise.md` — depth noise characterisation
- `docs/lit_review_ekf_dropout_fallback.md` — NIS gate + recenter spec
- `docs/lit_review_ekf_lag_vs_raw_noise.md` — CWNA tuning, train-without-EKF
- `docs/hardware_deployment_checklist.md` — full pre-deployment checklist
- `docs/lit_review_curriculum_advancement_criteria.md` — timeout paradox fix
