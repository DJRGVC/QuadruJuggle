"""RSL-RL PPO runner config for the hierarchical ball-juggle task (pi1).

pi1 is the high-level ball planner: it observes ball state + proprioception
and outputs 8D torso commands.  The action space is simpler (8D vs 12D) but
the planning task is more complex, so we use 3 hidden layers.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BallJuggleHierPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go1_ball_juggle_hier"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        # 1.0→0.5: pi2 amplifies noise via _CMD_SCALES (up to 4.0×); 1.0 sends OOD
        # commands (±8 rad/s omega) that crash pi2 immediately.  0.5 keeps 95% of
        # initial commands within pi2's training range.
        actor_hidden_dims=[256, 128],   # 2 layers — match pi2 architecture
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        # 0.01→0.005→0.001: Stage G entropy explosion (noise_std 0.34→0.98 over 1500
        # iters at 0.005). Mixed-target variance amplifies entropy term. 0.001 proven
        # stable in balance task.
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
