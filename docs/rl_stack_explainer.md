# QuadruJuggle RL Stack — Technical Reference

> A complete walkthrough of the algorithm, hyperparameters, simulator, and training loop
> powering the Go1 locomotion baseline and the QuadruJuggle project.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [The Algorithm: PPO](#2-the-algorithm-ppo)
3. [Advantage Estimation: GAE](#3-advantage-estimation-gae)
4. [The Neural Network Architecture](#4-the-neural-network-architecture)
5. [Hyperparameters — What They Actually Do](#5-hyperparameters--what-they-actually-do)
6. [The Simulator: Isaac Sim](#6-the-simulator-isaac-sim)
7. [The RL Framework: Isaac Lab](#7-the-rl-framework-isaac-lab)
8. [The Full Training Loop, End to End](#8-the-full-training-loop-end-to-end)
9. [Reading TensorBoard](#9-reading-tensorboard)
10. [Relevant Papers](#10-relevant-papers)

---

## 1. The Big Picture

We are training a **Unitree Go1 quadruped** to walk by optimizing a neural network policy
using **deep reinforcement learning**. The setup has three layers:

```
┌─────────────────────────────────────────────┐
│  RSL-RL (PPO algorithm)                     │  ← decides how to improve the policy
├─────────────────────────────────────────────┤
│  Isaac Lab (RL environment framework)       │  ← defines obs, actions, rewards, resets
├─────────────────────────────────────────────┤
│  Isaac Sim + PhysX 5 (GPU physics)          │  ← simulates 20,480 robots in parallel
└─────────────────────────────────────────────┘
```

The policy is a small MLP: it takes 48 numbers describing the robot's state and outputs
12 joint position targets (one per motor). Training runs ~20,000 robots simultaneously on
a single GPU, collecting nearly **1 million transitions per iteration**.

---

## 2. The Algorithm: PPO

> **Paper:** [Proximal Policy Optimization Algorithms — Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
> **Implementation:** [RSL-RL — ETH Zurich Robotic Systems Lab](https://github.com/leggedrobotics/rsl_rl)

### The problem PPO solves

In RL, we improve a policy by collecting experience and then updating the network weights
to make good actions more likely. The fundamental danger: if you update too aggressively,
you destroy the policy — one bad gradient step can send parameters into a region of weight
space the policy never recovers from.

Earlier algorithms like TRPO ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477))
solved this with a hard KL-divergence constraint, but it was expensive to compute.
PPO approximates the same constraint cheaply using a **clipped objective**.

### The clipped surrogate objective

At the heart of PPO is this loss function (directly from the RSL-RL source):

```python
ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
surrogate          = -advantages_batch * ratio
surrogate_clipped  = -advantages_batch * torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
surrogate_loss     = torch.max(surrogate, surrogate_clipped).mean()
```

`ratio` is the **probability ratio** π_new(a|s) / π_old(a|s) — how much more (or less)
likely the *current* policy is to take the same action that was taken during rollout,
compared to the policy that collected the data.

`clip_param = 0.2` clamps this ratio to `[0.8, 1.2]`. The `torch.max` then picks
whichever of the clipped or unclipped objective is more **pessimistic** — it never lets
a gradient step benefit from moving the ratio outside the safe zone. This is the
"proximal" part: stay close to the policy that collected the data.

### The value function loss

Alongside the actor, a **critic** network estimates V(s) — "how good is this state overall".
It is trained with a clipped MSE loss:

```python
value_clipped = target_values + (value - target_values).clamp(-clip_param, clip_param)
value_loss = torch.max(
    (value - returns).pow(2),
    (value_clipped - returns).pow(2)
).mean()
```

Again, clipping prevents destructively large value function updates.

### The entropy bonus

```python
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()
```

The `- entropy_coef * entropy` term **rewards the policy for staying uncertain**.
Without it, PPO often converges prematurely to a deterministic-but-suboptimal policy.
`entropy_coef = 0.01` keeps a small pressure toward exploration throughout training.

### Adaptive learning rate

Rather than a fixed LR, RSL-RL tracks KL divergence after each mini-batch and adjusts:

```python
if kl_mean > desired_kl * 2.0:
    learning_rate = max(1e-5, learning_rate / 1.5)   # policy changed too much → slow down
elif kl_mean < desired_kl / 2.0:
    learning_rate = min(1e-2, learning_rate * 1.5)   # barely changed → speed up
```

`desired_kl = 0.01`. This is more robust than fixed LR schedules and largely removes the
need to tune the learning rate manually.

---

## 3. Advantage Estimation: GAE

> **Paper:** [High-Dimensional Continuous Control Using GAE — Schulman et al., 2015](https://arxiv.org/abs/1506.02438)

The **advantage** A(s, a) answers: *was this specific action better or worse than average
for this state?* It is the signal that drives the actor update — positive advantage →
make this action more likely, negative → less likely.

We can't compute it exactly, so we use **Generalized Advantage Estimation**:

```
δ_t   = r_t + γ · V(s_{t+1}) - V(s_t)       ← TD residual at time t
A_t   = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

This is a geometric sum of TD residuals, controlled by two parameters:

- **γ (gamma) = 0.99** — the discount factor. A reward 100 steps in the future is worth
  γ^100 ≈ 0.37 of an immediate reward. High gamma = agent thinks long-term. For locomotion
  at 50Hz, 100 steps = 2 seconds, so γ=0.99 means the robot genuinely cares about
  what happens 2-3 seconds ahead.

- **λ (lambda) = 0.95** — the GAE trade-off parameter:
  - λ=0: pure TD(0) — low variance, high bias (critic errors dominate)
  - λ=1: pure Monte Carlo — unbiased, but very high variance (noisy returns)
  - λ=0.95: slightly biased but dramatically lower variance than MC

In practice, GAE with γ=0.99, λ=0.95 is the near-universal default for legged locomotion
tasks and works well across a wide range of reward structures.

---

## 4. The Neural Network Architecture

Both actor and critic are small MLPs with ELU activations:

```
Input: obs (48,)
  └─ Linear(48 → 128) → ELU
       └─ Linear(128 → 128) → ELU
            └─ Linear(128 → 128) → ELU
                 └─ Actor:  Linear(128 → 12)  → action means μ
                    Critic: Linear(128 → 1)   → state value V(s)
```

The actor outputs **Gaussian distribution parameters**: means μ (one per joint) and a
learned (but state-independent) standard deviation σ. During training, actions are
*sampled* from N(μ, σ) for exploration. At deployment, only μ is used.

`init_noise_std = 1.0` — the standard deviation starts at 1.0 (wide distribution, lots
of exploration) and is learned down during training as the policy becomes confident.

**Why ELU over ReLU?** ELU has negative saturation instead of hard zero, which avoids
the "dying ReLU" problem and provides smoother gradients — important when the network
must produce smooth, continuous joint targets.

**Why small?** The network needs to run at 50Hz on an embedded computer on the physical
Go1. A 3×128 MLP is tiny and runs in microseconds. Larger networks train better in sim
but are impractical on-device.

---

## 5. Hyperparameters — What They Actually Do

### Rollout parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `num_envs` | 20,480 | Parallel simulation instances. More envs = more diverse transitions per iteration = stabler gradient estimates. The key lever for GPU utilization. |
| `num_steps_per_env` | 48 | Steps collected from each env before a learning update. Total buffer size = 20480 × 48 ≈ 983k transitions. Larger = better GAE estimates but longer between updates. |

### PPO parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `clip_param` | 0.2 | Proximal constraint. Policy can change at most ±20% per update. Too high → unstable. Too low → slow convergence. 0.2 is the standard default from the original paper. |
| `num_learning_epochs` | 5 | Full passes over the rollout buffer per iteration. More epochs = squeeze more signal from the same data, but risk overfitting to stale transitions. |
| `num_mini_batches` | 4 | Splits the buffer into 4 chunks. Total gradient updates per iteration = 5 × 4 = 20. Smaller batches add gradient noise, which can help generalization. |
| `entropy_coef` | 0.01 | Weight on the entropy bonus. Prevents premature convergence. Small but important, especially early in training. |
| `value_loss_coef` | 1.0 | Relative weight of critic vs. actor loss. Equal weighting is standard. |
| `max_grad_norm` | 1.0 | Gradient clipping. Any gradient larger than 1.0 is rescaled. Prevents rare but catastrophic gradient explosions. |

### Optimization parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `learning_rate` | 1e-3 (adaptive) | Starting LR, adjusted by KL tracker. Adaptive schedule largely removes the need to tune this. |
| `desired_kl` | 0.01 | Target KL divergence between old and updated policy per mini-batch. Controls effective step size in parameter space. |
| `gamma` | 0.99 | Discount factor. See GAE section. |
| `lam` | 0.95 | GAE lambda. See GAE section. |
| `init_noise_std` | 1.0 | Initial action exploration width. Learned down during training. |

---

## 6. The Simulator: Isaac Sim

Isaac Sim is built on **NVIDIA Omniverse** and uses **PhysX 5** for physics simulation.
The key differentiator for RL: physics runs as **CUDA kernels directly on the GPU**.

> **Paper:** [Isaac Gym: High Performance GPU-Based Physics Simulation — Makoviychuk et al., 2021](https://arxiv.org/abs/2108.10470)
> (Isaac Gym is the predecessor; Isaac Sim inherits the same GPU-parallel architecture.)

### Why GPU-parallel physics matters

On a CPU simulator, you can run ~4-16 environments in parallel (one per core). Every
additional environment adds CPU load linearly. GPU simulation breaks this: all 20,480
environments are stepped simultaneously by thousands of CUDA cores. Doubling `num_envs`
from 10k to 20k adds almost zero extra wall-clock time per step.

This is why the training speedup from 64 → 20,480 environments was so dramatic on your
RTX 5070 Ti.

### Scene representation

Everything in Isaac Sim is a **USD (Universal Scene Description)** prim — a NVIDIA/Pixar
standard for describing 3D scenes. The Go1 robot is loaded from a URDF that gets converted
to USD. Isaac Lab's `Cloner` then replicates this USD prim 20,480 times across a grid,
with each copy getting its own PhysX rigid body state.

### PhysX configuration relevant to us

- **Physics step = 0.005s (200Hz)** — the internal simulation runs at 200Hz
- **Control step = 0.02s (50Hz)** — policy actions are applied every 4 physics steps
- **Rendering step = 0.02s** — irrelevant in headless training

The 4:1 ratio between physics and control steps means PhysX sub-steps each action,
which improves contact stability without requiring the policy to run at 200Hz.

---

## 7. The RL Framework: Isaac Lab

Isaac Lab wraps Isaac Sim and provides a gym-style RL interface via a manager-based
architecture. Each component is a `@configclass` Python dataclass — no YAML, pure Python
inheritance, easily overridable.

### The five managers

**ObservationManager** — assembles the policy input each step.
For Go1 flat, the 48-dim observation is:

```
base_lin_vel        (3)  — IMU linear velocity estimate
base_ang_vel        (3)  — IMU angular velocity
projected_gravity   (3)  — gravity vector in robot frame (tells the robot it's tilting)
velocity_commands   (3)  — target vx, vy, ωz from the command generator
joint_pos          (12)  — all 12 joint angles, offset from default pose
joint_vel          (12)  — all 12 joint velocities
actions            (12)  — previous action (gives the policy memory of one step)
```

**ActionManager** — translates the 12-dim policy output into joint position targets.
`scale = 0.25` means the output is multiplied by 0.25 before being added to the default
joint pose. This constrains the robot to small deviations from its nominal stance.

**RewardManager** — computes the scalar reward as a weighted sum:

```
track_lin_vel_xy_exp  × +1.50   exp(-|cmd_vel_xy - actual_vel_xy|²/0.25)
track_ang_vel_z_exp   × +0.75   exp(-|cmd_ωz - actual_ωz|²/0.25)
lin_vel_z_l2          × -2.00   penalize vertical bouncing
ang_vel_xy_l2         × -0.05   penalize rolling/pitching
dof_torques_l2        × -2e-4   penalize motor effort (energy efficiency)
dof_acc_l2            × -2.5e-7 penalize joint jerk
action_rate_l2        × -0.01   penalize rapid action changes (smoothness)
feet_air_time         × +0.25   reward feet leaving the ground (encourages walking gait)
flat_orientation_l2   × -2.50   heavily penalize body tilt (stay upright)
```

The exponential kernels for velocity tracking (`exp(-error²/σ²)`) are important: they
give a smooth, dense signal near zero error rather than a sparse hit-or-miss reward.
This is critical for locomotion learning — the robot needs gradient information even when
it's moving in roughly the right direction.

**TerminationManager** — resets an environment when:
1. `time_out` — episode length exceeded (~20s at 50Hz)
2. `base_contact` — robot trunk contacts the ground (fell over)

Environments **reset independently**. A fallen robot doesn't stall the other 20,479.
On reset, the robot is placed at a random XY position and yaw, and a new random
velocity command is issued.

**EventManager** — applies domain randomization:
- `physics_material` (startup): randomizes ground friction and restitution
- `add_base_mass` (startup): adds ±1-3kg to the trunk (simulates payload variation)
- `base_external_force_torque` (reset): applies random forces/torques to the trunk
- `reset_base` / `reset_robot_joints` (reset): randomizes initial pose

Domain randomization is the primary mechanism for **sim-to-real transfer** — by training
across a distribution of physics parameters, the policy becomes robust to the real world's
unpredictable dynamics.

> **Paper:** [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization — Peng et al., 2018](https://arxiv.org/abs/1710.06537)

---

## 8. The Full Training Loop, End to End

```
for iteration in range(max_iterations):           # 500 iterations total

    # ── ROLLOUT PHASE ──────────────────────────────────────────
    for step in range(num_steps_per_env):          # 48 steps

        obs = env.get_observations()               # tensor [20480, 48] on GPU
        actions = ppo.act(obs)                     # actor forward pass → [20480, 12]
                                                   # also stores V(s) and log π(a|s)

        obs, rewards, dones, info = env.step(actions)
        # PhysX steps all 20,480 environments simultaneously (4 sub-steps at 200Hz)
        # RewardManager computes per-env scalar rewards
        # TerminationManager checks falls and timeouts, resets those envs immediately

        ppo.process_env_step(obs, rewards, dones)
        # Stores (obs, action, reward, done, value, log_prob) in rollout buffer
        # Applies timeout bootstrapping: if an env timed out (not fell), its final
        # value V(s_T) is added to the reward so the critic isn't penalized for cutoffs

    # Total buffer: 20,480 × 48 = 983,040 transitions
    # ── END ROLLOUT ─────────────────────────────────────────────

    # ── LEARNING PHASE ──────────────────────────────────────────
    ppo.compute_returns(obs)
    # Runs GAE backwards through the buffer to compute advantages A_t and returns R_t
    # Uses γ=0.99, λ=0.95

    ppo.update()
    # for epoch in range(5):                       # num_learning_epochs
    #     for mini_batch in shuffle_and_split(buffer, 4):   # num_mini_batches
    #         recompute log π(a|s) and V(s) with current weights
    #         compute KL divergence → adapt learning rate
    #         compute clipped surrogate loss (actor)
    #         compute clipped value loss (critic)
    #         add entropy bonus
    #         total_loss.backward()
    #         clip_grad_norm_(params, 1.0)
    #         optimizer.step()
    # 20 gradient updates total per iteration

    log_to_tensorboard()                           # rewards, losses, FPS, LR, noise std
    if iteration % 50 == 0:
        save_checkpoint("model_{iteration}.pt")
```

At our settings (20,480 envs, `num_steps_per_env=48`), each iteration processes
**983,040 environment transitions** and performs **20 Adam gradient updates**.
Total across 500 iterations: ~491 million transitions seen.

---

## 9. Reading TensorBoard

Run in a separate terminal (venv active):
```bash
tensorboard --logdir ~/IsaacLab/logs/rsl_rl/unitree_go1_flat
```
Then open `http://localhost:6006`.

### The signals that matter most

**`Train/mean_reward`** — the primary learning signal. Should rise and plateau.
A well-converged flat locomotion policy typically reaches 40-60 depending on reward
scaling. Our best run hit ~33 at iter 467 and was still climbing.

**`Train/mean_episode_length`** — how long before the robot falls over, in steps.
Max episode length is ~1000 steps (20s). A robot that never falls will plateau here.
Rising episode length = robot is staying upright longer.

**`Loss/surrogate`** — actor loss. Should decrease and stabilize near zero.
Large positive values early in training are normal (policy hasn't learned yet).

**`Loss/value_function`** — critic loss (MSE). Should decrease over time as the
critic learns to predict returns accurately. A high value late in training suggests
the reward landscape is too complex for the critic to model.

**`Loss/entropy`** — action distribution entropy. Should decrease over time as the
policy becomes more deterministic. If it drops to near zero early, the policy
converged prematurely — consider increasing `entropy_coef`.

**`Loss/learning_rate`** — the adaptive LR. Watch this oscillate around ~1e-3.
If it collapses to 1e-5 and stays there, training has stalled.

**`Policy/mean_noise_std`** — the learned action standard deviation. Starts at 1.0,
should decrease to ~0.1-0.3 for a converged locomotion policy.

**`Perf/total_fps`** — environment steps per second across all envs. Higher = faster
training. At 20,480 envs on RTX 5070 Ti expect 300k-600k fps.

### The per-reward signals (`Episode_Reward/*`)

Each reward term is logged separately. What to expect from a converging policy:
- `track_lin_vel_xy_exp` and `track_ang_vel_z_exp` → rise toward their max values (1.5, 0.75)
- Penalty terms (`dof_acc_l2`, `action_rate_l2`, etc.) → start very negative (robot flailing), then recover toward 0 as the policy learns smooth motion
- `flat_orientation_l2` → starts very negative, quickly recovers (robot learns to not fall)

The initial dip in penalty terms followed by recovery is **normal and expected** — it
reflects the explore-then-exploit trajectory of PPO.

---

## 10. Relevant Papers

| Paper | Why it matters |
|-------|----------------|
| [PPO — Schulman et al., 2017](https://arxiv.org/abs/1707.06347) | The algorithm we use |
| [GAE — Schulman et al., 2015](https://arxiv.org/abs/1506.02438) | Advantage estimation (γ, λ) |
| [TRPO — Schulman et al., 2015](https://arxiv.org/abs/1502.05477) | PPO's predecessor; explains why the proximal constraint is necessary |
| [Isaac Gym — Makoviychuk et al., 2021](https://arxiv.org/abs/2108.10470) | GPU-parallel physics for RL; architectural basis of Isaac Sim |
| [Learning to Walk in Minutes — Kumar et al., 2021](https://arxiv.org/abs/2109.11978) | ETH Zurich using exactly this RSL-RL stack for legged locomotion |
| [Legged Locomotion in Challenging Terrains — Miki et al., 2022](https://arxiv.org/abs/2209.05433) | ANYmal on rough terrain; shows terrain curriculum + domain randomization at scale |
| [Dynamics Randomization — Peng et al., 2018](https://arxiv.org/abs/1710.06537) | Why domain randomization enables sim-to-real transfer |
| [Teacher-Student for Locomotion — Lee et al., 2020](https://arxiv.org/abs/2012.01052) | The privileged-state → camera distillation approach we will use in Phase 2 |
| [DAgger — Ross et al., 2011](https://arxiv.org/abs/1011.0686) | The imitation learning algorithm underlying Phase 2 teacher-student training |
