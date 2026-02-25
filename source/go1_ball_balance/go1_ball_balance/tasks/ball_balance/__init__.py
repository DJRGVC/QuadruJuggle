"""Ball-balance task — Go1 keeps a ping-pong ball centred on a back-mounted paddle."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-BallBalance-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_balance_env_cfg:BallBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BallBalancePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BallBalance-Go1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_balance_env_cfg:BallBalanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BallBalancePPORunnerCfg",
    },
)
