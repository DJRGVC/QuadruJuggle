"""RSL-RL PPO runner config for the hierarchical ball-juggle task (pi1).

rsl_rl 5.0.1 API: actor/critic use RslRlMLPModelCfg with distribution_cfg
for stochastic actor and distribution_cfg=None for deterministic critic.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class BallJuggleLauncherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for V4 launcher pi1."""
    num_steps_per_env = 48
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go1_ball_launcher"
    empirical_normalization = False

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.3,
    )


@configclass
class BallJuggleHierPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go1_ball_juggle_hier"
    empirical_normalization = False

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )
