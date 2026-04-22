"""RSL-RL PPO runner config for the hierarchical ball-juggle task (pi1).

pi1 is the high-level ball planner: it observes ball state + proprioception
and outputs 6D torso commands.  The action space is simpler (6D vs 12D) but
the planning task is more complex, so we use 3 hidden layers.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BallJuggleLauncherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for V4 launcher pi1.

    Stability changes vs BallJuggleHierPPORunnerCfg:
      - entropy_coef 0.01 → 0.05: strong entropy regularisation prevents std collapse
        (collapse was the recurring crash: 'normal expects all elements of std >= 0.0')
      - num_steps_per_env 24 → 48: launcher episodes are short (~20-100 steps after
        success termination); 24 steps gives degenerate rollout statistics.  48 steps
        collects enough diverse data per update to keep KL estimates stable.
      - schedule="fixed": adaptive LR with short/variable episodes swings LR wildly,
        amplifying gradient spikes. Fixed LR at 1e-4 is simpler and more stable.
      - max_grad_norm 0.5 → 0.3: extra clipping safety for the launcher task
      - learning_rate 3e-4 → 1e-4: conservative updates (launcher reward scale differs
        from pi1 — critic needs time to re-calibrate value estimates)
    """
    num_steps_per_env = 48       # was 24 — short episodes need more steps/update
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go1_ball_launcher"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,       # low — std clamp in train_launcher.py handles crash prevention
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",        # was "adaptive" — fixed avoids LR spikes on short eps
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

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],   # 3 layers for complex planning task
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
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
