# Perception pipeline for QuadruJuggle.
#
# ETH-style: GT position + noise model → EKF → pi1.
# No camera sensor is rendered during training; this directory
# contains the EKF, noise model, and sim integration wrappers.

from .ball_obs_spec import (  # noqa: F401
    BallObsNoiseCfg,
    D435iNoiseParams,
    PerceptionPipeline,
    ball_pos_perceived,
    ball_vel_perceived,
    inject_ekf_reset_event,
    reset_perception_pipeline,
    update_perception_noise_scale,
)

from .ball_ekf import BallEKF, BallEKFConfig  # noqa: F401
from .noise_model import D435iNoiseModel, D435iNoiseModelCfg  # noqa: F401
from .phase_tracker import BallPhaseTracker, BallPhaseTrackerConfig, BallPhase  # noqa: F401
