# Perception pipeline for QuadruJuggle.
#
# ETH-style: GT position + noise model → EKF → pi1.
# No camera sensor is rendered during training; this directory
# contains the EKF, noise model, and sim integration wrappers.

from .ball_obs_spec import (  # noqa: F401
    BallObsNoiseCfg,
    D435iNoiseParams,
    ball_pos_perceived,
    ball_vel_perceived,
)
