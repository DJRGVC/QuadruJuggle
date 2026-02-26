"""RSL-RL PPO runner config for the ball-balance task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BallBalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64        # longer rollouts improve advantage estimates for ~500-step episodes
    max_iterations = 3000
    save_interval = 50            # checkpoint more often (useful for short reward-tuning runs)
    experiment_name = "go1_ball_balance"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,               # tighter clip — prevents large single-step policy jumps
        entropy_coef=0.01,
        num_learning_epochs=3,        # fewer passes per rollout — less policy change per iteration
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,             # tighter KL — caps adaptive LR growth near a good solution
        max_grad_norm=0.5,            # tighter grad clip — prevents single bad gradient from destabilising
    )
