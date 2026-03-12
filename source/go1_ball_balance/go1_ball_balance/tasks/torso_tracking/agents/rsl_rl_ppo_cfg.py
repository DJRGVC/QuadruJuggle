"""RSL-RL PPO runner config for the torso-tracking task (pi2).

Matched to Isaac Lab Go1 locomotion defaults to ensure walking emerges:
  - init_noise_std=1.0 (wide exploration — must stumble into gait)
  - entropy_coef=0.01 (prevents early convergence to standing still)
  - 3-layer network [512, 256, 128] (capacity for gait coordination)
  - num_steps_per_env=48 (Isaac Lab Go1 flat; longer rollouts capture full strides)
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TorsoTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48         # Isaac Lab Go1 flat (was 24; longer rollouts = full strides)
    max_iterations = 5000
    save_interval = 50
    experiment_name = "go1_torso_tracking"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                    # Isaac Lab default (was 0.5 — not enough exploration)
        actor_hidden_dims=[512, 256, 128],     # Isaac Lab Go1 rough (was [256, 128] — too small)
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,         # Isaac Lab default (was 0.001 — 10x too low, killed exploration)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
