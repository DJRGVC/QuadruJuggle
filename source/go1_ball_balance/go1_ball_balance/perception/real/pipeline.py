"""Real-time perception pipeline orchestrator.

Connects D435i camera -> ball detector -> world-frame EKF -> body-frame
observations for pi1. Runs camera+YOLO on a dedicated thread at 90Hz;
EKF predict runs at 200Hz on the policy thread between measurements.

The output interface is identical to the sim PerceptionPipeline:
    ball_pos_b (3,), ball_vel_b (3,), ball_lost (bool)

See docs/hardware_pipeline_architecture.md §3.4 for full specification.
"""

from __future__ import annotations

import numpy as np

from ..ball_ekf import BallEKF, BallEKFConfig
from .calibration import CameraExtrinsics
from .camera import D435iCamera, CameraIntrinsics
from .config import HardwarePipelineConfig
from .detector import BallDetector


class RealPerceptionPipeline:
    """Real-time D435i -> YOLO -> EKF -> pi1 pipeline.

    Threading model:
    - Camera acquisition + YOLO detection run on a dedicated thread at 90Hz.
    - EKF predict runs at 200Hz on the main (policy) thread.
    - Policy calls get_observation() at 50Hz -- returns latest EKF estimate
      transformed to body frame using current robot orientation.

    Usage::

        config = HardwarePipelineConfig()
        pipeline = RealPerceptionPipeline(config)
        pipeline.start()

        while running:
            # Called at 50Hz by the policy loop
            obs = pipeline.get_observation(
                robot_quat_w=imu_quaternion,
                robot_pos_w=robot_position,
            )
            ball_pos_b = obs.ball_pos_b
            ball_vel_b = obs.ball_vel_b
            if obs.ball_lost:
                # handle dropout

        pipeline.stop()
    """

    def __init__(self, config: HardwarePipelineConfig) -> None:
        self._config = config
        self._camera: D435iCamera | None = None
        self._detector: BallDetector | None = None
        self._extrinsics: CameraExtrinsics | None = None
        self._intrinsics: CameraIntrinsics | None = None
        self._ekf: BallEKF | None = None
        self._running = False

        # Dropout tracking
        self._consecutive_dropouts = 0

    def start(self) -> None:
        """Initialise camera, detector, EKF, and start acquisition thread.

        Blocks until the first depth frame is received.
        """
        raise NotImplementedError(
            "RealPerceptionPipeline.start() is a stub — implement when "
            "hardware is available. See docs/hardware_pipeline_architecture.md §3.4."
        )

    def get_observation(
        self,
        robot_quat_w: np.ndarray,
        robot_pos_w: np.ndarray,
    ) -> PipelineObservation:
        """Get latest ball observation in body frame.

        Called by the policy loop at 50Hz. Runs EKF predict steps to
        interpolate between camera measurements (90Hz -> 200Hz predict).

        Args:
            robot_quat_w: (4,) robot orientation in world frame [w, x, y, z].
            robot_pos_w: (3,) robot position in world frame.

        Returns:
            PipelineObservation with ball_pos_b, ball_vel_b, ball_lost.
        """
        raise NotImplementedError(
            "RealPerceptionPipeline.get_observation() is a stub."
        )

    def stop(self) -> None:
        """Stop acquisition thread and release camera."""
        raise NotImplementedError(
            "RealPerceptionPipeline.stop() is a stub."
        )


class PipelineObservation:
    """Output of the real perception pipeline, matching sim interface."""

    __slots__ = ("ball_pos_b", "ball_vel_b", "ball_lost", "detected")

    def __init__(
        self,
        ball_pos_b: np.ndarray,
        ball_vel_b: np.ndarray,
        ball_lost: bool,
        detected: bool,
    ) -> None:
        self.ball_pos_b = ball_pos_b  # (3,) body frame, metres
        self.ball_vel_b = ball_vel_b  # (3,) body frame, m/s
        self.ball_lost = ball_lost  # True when cov trace > threshold
        self.detected = detected  # True if YOLO detected ball this frame
