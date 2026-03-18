"""Hierarchical ball-juggle task — pi1 outputs 6D torso commands, frozen pi2 converts to joints."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-BallJuggleMirror-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_juggle_mirror_env_cfg:BallJuggleMirrorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BallJuggleHierPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BallJuggleHier-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_juggle_hier_env_cfg:BallJuggleHierEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BallJuggleHierPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BallJuggleHier-Go1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_juggle_hier_env_cfg:BallJuggleHierEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BallJuggleHierPPORunnerCfg",
    },
)
