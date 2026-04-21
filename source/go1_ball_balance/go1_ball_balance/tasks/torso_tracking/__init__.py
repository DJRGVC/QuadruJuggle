"""Torso-tracking task — Go1 tracks 6D torso pose/velocity commands (pi2)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-TorsoTracking-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.torso_tracking_env_cfg:TorsoTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TorsoTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-TorsoTracking-Go1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.torso_tracking_env_cfg:TorsoTrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TorsoTrackingPPORunnerCfg",
    },
)
