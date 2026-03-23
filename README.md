# QuadruJuggle

Training a [Unitree Go1](https://www.unitree.com/go1/) quadruped to juggle a ping-pong ball on a disc paddle mounted to its back, using [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) (Isaac Sim 5.1.0).

The robot uses its **body as the actuator** — tilting and bouncing its torso to keep the ball airborne on the paddle. No arm, no gripper.

---

## Architecture Overview

```
pi1  (high-level: decides HOW to move the torso)
  │  6-D normalized torso command [h, ḣ, roll, pitch, ω_roll, ω_pitch]
  ▼
pi2  (low-level: frozen RL policy — executes the torso command)
  │  12-D joint position targets
  ▼
Go1 robot  +  kinematic disc paddle  +  ping-pong ball
```

### Pi2 — Torso Tracking (trained once, then frozen)

| | |
|---|---|
| **Task** | Track a 6-D torso command: height, height velocity, roll, pitch, roll rate, pitch rate |
| **Obs** | Proprioception (joint pos/vel, IMU, foot contacts) + 6-D command |
| **Act** | 12-D joint position targets |
| **Network** | MLP `[256, 128, 64]`, ELU |
| **Algorithm** | PPO, ~20 000 iterations |

### Pi1 — Ball Juggling Controller (two variants)

| Variant | Description | File |
|---------|-------------|------|
| **Mirror law** (analytic) | Closed-form reflection geometry. No training needed. | `mirror_law_action.py` |
| **Learned RL** | PPO policy that directly outputs 6-D torso command. Trained on top of frozen pi2. | `pi1_learned_action.py` |

---

## Setup

### Requirements

| | |
|---|---|
| **OS** | Ubuntu 24.04 LTS (x86_64) |
| **GPU** | NVIDIA RTX with ≥ 8 GB VRAM (RTX 3060 Ti tested) |
| **Driver** | NVIDIA ≥ 550.x (`nvidia-smi` to check) |
| **Isaac Lab** | 2.3.2 |
| **Isaac Sim** | 5.1.0 (bundled with Isaac Lab) |
| **Python** | 3.11 (bundled in conda env) |

### System packages

```bash
sudo apt-get update && sudo apt-get install -y \
    git git-lfs build-essential cmake \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    vulkan-tools libvulkan1
```

### Install Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
git checkout v2.3.2
conda create -n isaaclab python=3.11 -y
conda activate isaaclab
./isaaclab.sh --install
```

### Install this repo

```bash
git clone <this-repo> ~/QuadruJuggle
cd ~/QuadruJuggle
export REPO=$(pwd)
export ISAACLAB_PATH=~/IsaacLab   # adjust if cloned elsewhere

conda activate isaaclab
pip install -e source/go1_ball_balance
pip install -r requirements.txt
```

### Set PYTHONPATH (required for all scripts)

```bash
export PYTHONPATH=$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH
```

Add this to your `~/.bashrc` or `~/.zshrc` to avoid repeating it.

---

## Quick Start

### Play — Mirror-Law (no training needed, just pi2 checkpoint)

```bash
cd $REPO
python scripts/play_mirror_law.py \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \
    --apex_height 0.20 \
    --num_envs 4
```

### Train Pi2 (torso tracking — do this first on a fresh machine)

```bash
python scripts/rsl_rl/train_torso_tracking.py \
    --task Isaac-TorsoTracking-Go1-v0 \
    --num_envs 4096 \
    --headless
```

Converges at ~5 000–8 000 iterations. Checkpoint saved to `logs/rsl_rl/go1_torso_tracking/<TIMESTAMP>/model_best.pt`.

### Train Pi1 (learned RL juggling)

```bash
python scripts/rsl_rl/train_pi1.py \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/<TIMESTAMP>/model_best.pt \
    --num_envs 1024 \
    --headless
```

### Play — Learned Pi1

```bash
python scripts/play_pi1.py \
    --pi1_checkpoint logs/rsl_rl/go1_ball_juggle_hier/<TIMESTAMP>/model_best.pt \
    --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/<TIMESTAMP>/model_best.pt \
    --apex_height 0.20 \
    --num_envs 4
```

### Monitor Training

```bash
tensorboard --logdir logs/rsl_rl/go1_ball_juggle_hier
```

---

## Repository Layout

```
scripts/
  play_mirror_law.py          Play mirror-law pi1 + frozen pi2
  play_pi1.py                 Play learned pi1 + frozen pi2
  rsl_rl/
    train_torso_tracking.py   Train pi2 (torso tracking)
    train_pi1.py              Train learned pi1 (ball juggling)

source/go1_ball_balance/go1_ball_balance/tasks/
  torso_tracking/
    action_term.py            TorsoCommandAction — loads + runs frozen pi2
    mirror_law_action.py      Pi1: analytic mirror-law controller
    ball_kalman_filter.py     Batched 6-state Kalman filter for noisy ball state
    agents/rsl_rl_ppo_cfg.py  Pi2 PPO hyperparameters
    mdp/
      commands.py             Torso command definitions
      rewards.py              Pi2 tracking rewards
  ball_juggle_hier/
    ball_juggle_mirror_env_cfg.py   Mirror-law play env config
    ball_juggle_pi1_env_cfg.py      Learned pi1 training env config
    pi1_learned_action.py           Learned pi1 action term
    mdp/
      events.py               Per-env apex height randomisation
  ball_juggle/mdp/
    rewards.py                Ball juggling rewards (apex height, bouncing, xy dist)
    observations.py           Ball-state observations, target apex height obs
  ball_balance/mdp/
    events.py                 Ball + robot reset functions
    rewards.py                Stability rewards (trunk tilt, body contact, foot contact)
    terminations.py           Termination conditions (tilt, height, ball off)

assets/paddle/
  disc.usda                   Paddle mesh (scaled 1.8× in XY → radius ≈ 0.153 m)

logs/rsl_rl/
  go1_torso_tracking/         Pi2 checkpoints
  go1_ball_juggle_hier/       Pi1 checkpoints

docs/
  OPERATION_MANUAL.md         Detailed tuning and troubleshooting guide
```

---

## Results

| Controller | Apex height | Episode length | Notes |
|-----------|-------------|----------------|-------|
| Mirror law (GT state) | 0.20 m | 1500/1500 steps | All 4 envs survive full episode |
| Mirror law + noise + KF + EMA | 0.20 m | 400–1500 steps | `--cmd_smooth_alpha 0.25 --h_nominal 0.41` |
| Learned pi1 v1 (1741 iters) | 0.20 m fixed | 1500/1500 (3/4 envs) | Active bouncing (ball_vz ±2 m/s) |
| Learned pi1 v3 (in training) | 0.10–0.60 m variable | — | Noise training, tilt termination, paddle XY gate |

---

## Sim-to-Real Roadmap

| Phase | Status | Notes |
|-------|--------|-------|
| Pi2 torso tracking (sim) | Done | Frozen checkpoint |
| Mirror-law juggling (sim) | Done | No training needed |
| Learned pi1 (sim) | Done | Variable apex height 0.10–0.60 m |
| Noisy ball state + KF | Done | Simulates stereo camera noise |
| Pi1 robustness (tilt termination + body contact penalty) | Training | Current |
| Egocentric camera perception | Not started | Stereo / ping-pong detection lib |
| Domain randomization (mass, friction) | Not started | Required for sim-to-real |
| Hardware deployment | Not started | unitree_rl_gym / ROS2 |

---

## References

- [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) — PPO implementation
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) — Go1 sim-to-real deployment
