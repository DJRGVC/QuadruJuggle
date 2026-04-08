# Real hardware perception pipeline for QuadruJuggle.
#
# This package implements the D435i -> YOLO -> EKF -> pi1 pipeline
# for deployment on Jetson Orin NX. All components produce the same
# (ball_pos_b, ball_vel_b, ball_lost) interface as the sim pipeline.
#
# See docs/hardware_pipeline_architecture.md for full specification.

from .config import HardwarePipelineConfig  # noqa: F401
from .camera import D435iCamera  # noqa: F401
from .detector import BallDetector, Detection  # noqa: F401
from .calibration import CameraExtrinsics, CameraCalibrator  # noqa: F401
from .pipeline import RealPerceptionPipeline  # noqa: F401
