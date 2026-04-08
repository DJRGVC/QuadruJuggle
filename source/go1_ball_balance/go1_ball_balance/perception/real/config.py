"""Hardware-specific configuration for the real perception pipeline.

All tuneable parameters in one place. Reuses BallEKFConfig from
ball_ekf.py for EKF-specific settings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwarePipelineConfig:
    """All hardware-specific parameters in one place."""

    # Camera ---------------------------------------------------------------
    camera_serial: str | None = None  # None = first available D435i
    depth_width: int = 848
    depth_height: int = 480
    depth_fps: int = 90  # requires firmware >= v5.13

    # Detector -------------------------------------------------------------
    yolo_engine_path: str = "models/ball_detector.engine"
    yolo_conf_thresh: float = 0.5
    hough_fallback: bool = True  # Hough circle fallback when YOLO conf < 0.4

    # Calibration ----------------------------------------------------------
    extrinsics_path: str = "config/camera_extrinsics.yaml"

    # EKF (timing + gating; Q/R params live in BallEKFConfig) --------------
    ekf_predict_hz: int = 200  # predict-only steps between measurements
    nis_gate_thresh: float = 11.35  # chi2(3, p=0.99) — reject wild measurements
    lost_ball_frames: int = 5  # consecutive dropouts before recenter
    covariance_trace_thresh: float = 0.001  # m^2 — ball_lost_flag trigger

    # Policy interface -----------------------------------------------------
    policy_hz: int = 50

    # Depth rejection bounds (metres) --------------------------------------
    min_depth: float = 0.168  # D435i minimum Z at 848x480
    max_depth: float = 2.0  # ball can't be farther than this
