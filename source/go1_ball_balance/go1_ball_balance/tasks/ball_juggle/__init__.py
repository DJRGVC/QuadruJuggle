"""Ball-juggle task — Go1 bounces a ping-pong ball to a consistent apex height."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-BallJuggle-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_juggle_env_cfg:BallJuggleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BallJugglePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BallJuggle-Go1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_juggle_env_cfg:BallJuggleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BallJugglePPORunnerCfg",
    },
)
