"""Real-time perception pipeline orchestrator.

Connects D435i camera -> ball detector -> world-frame EKF -> body-frame
observations for pi1. Runs camera+YOLO on a dedicated thread at 90Hz;
EKF predict runs at 200Hz on the policy thread between measurements.

The output interface is identical to the sim PerceptionPipeline:
    ball_pos_b (3,), ball_vel_b (3,), ball_lost (bool)

See docs/hardware_pipeline_architecture.md §3.4 for full specification.

Threading model:
    [Acq Thread]  camera.get_frame() -> detector.detect() -> _measurement_queue
    [Main Thread]  get_observation() pops queue, runs EKF predict+update, returns obs

The pipeline accepts either real D435i or MockCamera/MockDetector via
dependency injection for testing.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from ..ball_ekf import BallEKF, BallEKFConfig
from .calibration import CameraExtrinsics
from .camera import CameraIntrinsics
from .config import HardwarePipelineConfig
from .detector import Detection


@dataclass
class _Measurement:
    """Internal: a single camera measurement timestamped for EKF update."""
    pos_cam: np.ndarray   # (3,) ball position in camera frame
    timestamp: float       # monotonic seconds
    confidence: float
    method: str


class PipelineObservation:
    """Output of the real perception pipeline, matching sim interface."""

    __slots__ = ("ball_pos_b", "ball_vel_b", "ball_lost", "detected",
                 "timestamp", "ekf_pos_w", "ekf_vel_w")

    def __init__(
        self,
        ball_pos_b: np.ndarray,
        ball_vel_b: np.ndarray,
        ball_lost: bool,
        detected: bool,
        timestamp: float = 0.0,
        ekf_pos_w: np.ndarray | None = None,
        ekf_vel_w: np.ndarray | None = None,
    ) -> None:
        self.ball_pos_b = ball_pos_b  # (3,) body frame, metres
        self.ball_vel_b = ball_vel_b  # (3,) body frame, m/s
        self.ball_lost = ball_lost  # True when cov trace > threshold
        self.detected = detected  # True if detector found ball this cycle
        self.timestamp = timestamp
        self.ekf_pos_w = ekf_pos_w  # (3,) world frame (for debug)
        self.ekf_vel_w = ekf_vel_w  # (3,) world frame (for debug)


class RealPerceptionPipeline:
    """Real-time D435i -> detector -> EKF -> pi1 pipeline.

    Threading model:
    - Camera acquisition + detection run on a dedicated thread at camera fps.
    - EKF predict runs at 200Hz on the main (policy) thread.
    - Policy calls get_observation() at 50Hz -- returns latest EKF estimate
      transformed to body frame using current robot orientation.

    Usage::

        config = HardwarePipelineConfig()
        pipeline = RealPerceptionPipeline(config)
        pipeline.start()

        while running:
            obs = pipeline.get_observation(
                robot_quat_w=imu_quaternion,
                robot_pos_w=robot_position,
            )

        pipeline.stop()

    For testing, inject MockCamera and MockDetector::

        from perception.real.mock import MockCamera, MockDetector
        pipeline = RealPerceptionPipeline(config, camera=MockCamera(), detector=MockDetector())
    """

    def __init__(
        self,
        config: HardwarePipelineConfig,
        camera=None,
        detector=None,
        extrinsics: CameraExtrinsics | None = None,
        ekf_config: BallEKFConfig | None = None,
    ) -> None:
        self._config = config
        self._camera = camera
        self._detector = detector
        self._extrinsics = extrinsics
        self._intrinsics: CameraIntrinsics | None = None
        self._ekf_config = ekf_config or BallEKFConfig()

        # EKF: single-env on CPU for real-time use
        self._ekf: BallEKF | None = None

        # Threading
        self._running = False
        self._acq_thread: threading.Thread | None = None
        self._meas_lock = threading.Lock()
        self._meas_queue: deque[_Measurement] = deque(maxlen=10)

        # Timing
        self._last_predict_time: float | None = None
        self._last_update_time: float | None = None

        # Dropout tracking
        self._consecutive_dropouts = 0

        # Stats (for diagnostics)
        self._total_frames = 0
        self._total_detections = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        """Pipeline statistics for diagnostics."""
        return {
            "total_frames": self._total_frames,
            "total_detections": self._total_detections,
            "consecutive_dropouts": self._consecutive_dropouts,
            "detection_rate": (
                self._total_detections / max(1, self._total_frames)
            ),
        }

    def start(self) -> None:
        """Initialise camera, detector, EKF, and start acquisition thread.

        Blocks until the first depth frame is received (or camera is mock).
        """
        if self._running:
            raise RuntimeError("Pipeline already running.")

        # Start camera
        self._camera.start()
        self._intrinsics = self._camera.get_intrinsics()

        # Initialise EKF (single-env, CPU)
        self._ekf = BallEKF(num_envs=1, device="cpu", cfg=self._ekf_config)

        # Reset state
        self._meas_queue.clear()
        self._consecutive_dropouts = 0
        self._total_frames = 0
        self._total_detections = 0
        self._last_predict_time = time.monotonic()
        self._last_update_time = None

        # Start acquisition thread
        self._running = True
        self._acq_thread = threading.Thread(
            target=self._acquisition_loop,
            name="perception-acq",
            daemon=True,
        )
        self._acq_thread.start()

    def _acquisition_loop(self) -> None:
        """Camera + detector loop running on dedicated thread."""
        while self._running:
            result = self._camera.get_frame()
            if result is None:
                time.sleep(0.001)  # 1ms backoff if no frame ready
                continue

            depth_frame, timestamp = result
            self._total_frames += 1

            # Run detection
            detection = self._detector.detect(depth_frame, self._intrinsics)

            if detection is not None:
                self._total_detections += 1
                meas = _Measurement(
                    pos_cam=detection.pos_cam,
                    timestamp=timestamp,
                    confidence=detection.confidence,
                    method=detection.method,
                )
                with self._meas_lock:
                    self._meas_queue.append(meas)

    def get_observation(
        self,
        robot_quat_w: np.ndarray,
        robot_pos_w: np.ndarray,
    ) -> PipelineObservation:
        """Get latest ball observation in body frame.

        Called by the policy loop at 50Hz. Drains measurement queue,
        runs EKF predict+update steps, and transforms the estimate
        to body frame.

        Args:
            robot_quat_w: (4,) robot orientation in world frame [w, x, y, z].
            robot_pos_w: (3,) robot position in world frame.

        Returns:
            PipelineObservation with ball_pos_b, ball_vel_b, ball_lost.
        """
        if self._ekf is None:
            raise RuntimeError("Pipeline not started — call start() first.")

        now = time.monotonic()

        # Build rotation matrices
        R_body_world = _quat_to_rotmat(robot_quat_w)  # body -> world
        R_world_body = R_body_world.T  # world -> body

        # Camera-to-world: cam -> body -> world
        R_cam_world = R_body_world @ self._extrinsics.R_cam_body if self._extrinsics else R_body_world
        t_cam_world = (
            robot_pos_w + R_body_world @ self._extrinsics.t_cam_body
            if self._extrinsics else robot_pos_w
        )

        # Drain measurement queue
        measurements: list[_Measurement] = []
        with self._meas_lock:
            while self._meas_queue:
                measurements.append(self._meas_queue.popleft())

        detected = len(measurements) > 0

        if detected:
            self._consecutive_dropouts = 0
            # Use latest measurement (most recent timestamp)
            meas = measurements[-1]

            # Transform measurement from camera frame to world frame
            pos_world = R_cam_world @ meas.pos_cam + t_cam_world

            # EKF predict + update
            dt = now - self._last_predict_time if self._last_predict_time else 0.02
            dt = max(0.001, min(dt, 0.1))  # clamp to [1ms, 100ms]

            z = torch.tensor(pos_world, dtype=torch.float32).unsqueeze(0)  # (1,3)
            det_mask = torch.tensor([True])

            self._ekf.step(z, det_mask, dt=dt)
            self._last_update_time = now
        else:
            self._consecutive_dropouts += 1

            # Predict-only step (no measurement)
            dt = now - self._last_predict_time if self._last_predict_time else 0.02
            dt = max(0.001, min(dt, 0.1))

            z_dummy = torch.zeros(1, 3)
            det_mask = torch.tensor([False])
            self._ekf.step(z_dummy, det_mask, dt=dt)

        self._last_predict_time = now

        # Extract EKF state (world frame)
        ekf_pos_w = self._ekf.pos[0].numpy().copy()  # (3,)
        ekf_vel_w = self._ekf.vel[0].numpy().copy()  # (3,)

        # Transform to body frame
        ball_pos_b = R_world_body @ (ekf_pos_w - robot_pos_w)
        ball_vel_b = R_world_body @ ekf_vel_w

        # Determine ball_lost from covariance trace
        cov_trace = float(torch.diagonal(
            self._ekf._P[0, :3, :3]
        ).sum())
        ball_lost = (
            cov_trace > self._config.covariance_trace_thresh
            or self._consecutive_dropouts >= self._config.lost_ball_frames
        )

        return PipelineObservation(
            ball_pos_b=ball_pos_b.astype(np.float32),
            ball_vel_b=ball_vel_b.astype(np.float32),
            ball_lost=ball_lost,
            detected=detected,
            timestamp=now,
            ekf_pos_w=ekf_pos_w,
            ekf_vel_w=ekf_vel_w,
        )

    def stop(self) -> None:
        """Stop acquisition thread and release camera."""
        self._running = False
        if self._acq_thread is not None:
            self._acq_thread.join(timeout=2.0)
            self._acq_thread = None
        if self._camera is not None:
            self._camera.stop()

    def reset_ekf(
        self,
        init_pos: np.ndarray | None = None,
        init_vel: np.ndarray | None = None,
    ) -> None:
        """Reset the EKF state (e.g. after ball replacement).

        Args:
            init_pos: (3,) initial position in world frame, or zeros.
            init_vel: (3,) initial velocity in world frame, or zeros.
        """
        if self._ekf is None:
            return
        pos = torch.tensor(
            init_pos if init_pos is not None else np.zeros(3),
            dtype=torch.float32,
        ).unsqueeze(0)
        vel = torch.tensor(
            init_vel if init_vel is not None else np.zeros(3),
            dtype=torch.float32,
        ).unsqueeze(0)
        self._ekf.reset(
            env_ids=torch.tensor([0]),
            init_pos=pos,
            init_vel=vel,
        )
        self._consecutive_dropouts = 0
        self._last_predict_time = time.monotonic()
        self._last_update_time = None


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
