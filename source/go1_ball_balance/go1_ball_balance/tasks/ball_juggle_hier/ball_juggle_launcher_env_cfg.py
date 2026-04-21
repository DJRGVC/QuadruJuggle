"""V4 Launcher Pi1 — environment config for precision-launch training.

Design philosophy (vs V3 ball_juggle_pi1_env_cfg.py):
------------------------------------------------------
V3 pi1 was trained to juggle *indefinitely* at randomized heights.
It overshoots targets and has no incentive to hand off cleanly to mirror law.

V4 launcher pi1 has a single job:
  Get the ball to exactly [target - 0.05, target + 0.05] m above the paddle
  in as few bounces as possible, then stop commanding — mirror law takes over.

Key reward changes:
  - ball_apex_height: std tightened 0.10 → 0.03 m (precise targeting)
  - apex_overshoot:   NEW penalty if apex > target + 0.05 m
  - launcher_success: NEW terminal reward for landing in window
  - ball_bouncing:    weight reduced (speed less important than accuracy)
  - alive:            reduced (don't reward just surviving)

Key termination changes:
  - launcher_success: episode SUCCEEDS (high reward) when apex in window
  - time_out:         shortened to 10 s — fast precision, not endurance

This env is identical to BallJugglePi1EnvCfg in scene/observations/actions/events.
Only rewards and terminations differ.

Usage:
    Register "Isaac-BallJuggleLauncher-Go1-v0" in __init__.py, then:
    python scripts/rsl_rl/train_launcher.py --num_envs 4096
    python scripts/play_launcher_hybrid.py \\
        --launcher_checkpoint logs/rsl_rl/go1_ball_launcher/TIMESTAMP/model_best.pt \\
        --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \\
        --apex_height 0.30
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from . import mdp
from .pi1_learned_action import LearnedPi1TorsoActionCfg
from .ball_juggle_pi1_env_cfg import (
    BallJugglePi1SceneCfg,
    ObservationsCfg,
    ActionsCfg,
    EventCfg,
    TerminationsCfg as Pi1TerminationsCfg,
)

from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip

_ASSETS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..",
    "assets", "paddle",
))

_PADDLE_OFFSET_B    = (0.0, 0.0, 0.070)
_PADDLE_HALF_EXTENT = 0.153
_BALL_RADIUS        = 0.020

# How close to target counts as a successful launch [m]
_LAUNCH_WINDOW = 0.05


# ---------------------------------------------------------------------------
# Rewards — precision launcher variant
# ---------------------------------------------------------------------------

@configclass
class LauncherRewardsCfg:
    """Rewards tuned for precision launching, not sustained juggling."""

    # Survival: much lower weight — don't reward just staying alive
    alive = RewTerm(func=mdp.is_alive, weight=0.1)

    # Height safety — same as pi1
    base_height = RewTerm(
        func=mdp.base_height_penalty,
        weight=-2.0,
        params={"min_height": 0.34, "robot_cfg": SceneEntityCfg("robot")},
    )
    base_height_max = RewTerm(
        func=mdp.base_height_max_penalty, weight=-3.0,
        params={"max_height": 0.43, "robot_cfg": SceneEntityCfg("robot")},
    )
    early_termination = RewTerm(func=mdp.early_termination_penalty, weight=-5.0)

    # ── Success bonus (NEW) ───────────────────────────────────────────────────
    # early_termination_penalty fires on ALL non-timeout dones, including
    # launcher_success — this explicitly offsets that penalty and adds a strong
    # positive terminal reward so the robot is incentivised to hit the window.
    launcher_success_bonus = RewTerm(
        func=mdp.ball_apex_in_window,
        weight=25.0,
        params={
            "window": _LAUNCH_WINDOW,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.34,
            "nominal_height": 0.40,
        },
    )

    # ── Precision apex reward (TIGHT std vs V3's 0.10) ────────────────────────
    # Peak reward only within ±0.03 m of target — forces accurate launching
    ball_apex_height = RewTerm(
        func=mdp.ball_apex_height_reward,
        weight=5.0,                     # reduced 10→5 to prevent value fn explosion
        params={
            "target_height": 0.30,     # overridden by randomize_apex_height event
            "std": 0.05,               # loosened 0.03→0.05: less spiky, more stable
            "ball_radius": _BALL_RADIUS,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.34,
            "nominal_height": 0.40,
            "paddle_radius": _PADDLE_HALF_EXTENT,
        },
    )

    # ── Overshoot penalty (NEW — not in V3) ───────────────────────────────────
    # Penalize any bounce that exceeds target + window. Discourages the
    # aggressive energy building that made V3 pi1 useless for mirror law handoff.
    apex_overshoot = RewTerm(
        func=mdp.ball_apex_overshoot_penalty,
        weight=-10.0,
        params={
            "window": _LAUNCH_WINDOW,       # penalty starts at target + 0.05 m
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.34,
            "nominal_height": 0.40,
        },
    )

    # ── Bouncing: lower weight than V3 (speed less critical than precision) ───
    ball_bouncing = RewTerm(
        func=mdp.ball_bouncing_reward,
        weight=0.1,                    # V3=2.0 → V4=0.5 → V4.1=0.1: prevent safe-bounce local optimum
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.34,
            "nominal_height": 0.40,
            "target_vz": 1.5,
            "std": 0.5,
        },
    )

    # Centering and smoothness penalties — same as V3
    ball_xy_dist = RewTerm(
        func=mdp.ball_xy_dist_penalty,
        weight=-2.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.34,
            "nominal_height": 0.40,
        },
    )
    trunk_tilt    = RewTerm(func=mdp.trunk_tilt_penalty, weight=-3.0,
                            params={"robot_cfg": SceneEntityCfg("robot")})
    trunk_contact = RewTerm(
        func=mdp.trunk_contact_penalty,
        weight=-2.0,
        params={"contact_cfg": SceneEntityCfg("contact_forces")},
    )
    body_lin_vel  = RewTerm(func=mdp.body_lin_vel_penalty, weight=-0.10,
                            params={"robot_cfg": SceneEntityCfg("robot")})
    body_ang_vel  = RewTerm(func=mdp.body_ang_vel_penalty, weight=-0.05,
                            params={"robot_cfg": SceneEntityCfg("robot")})
    action_rate   = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2, weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    foot_contact  = RewTerm(
        func=mdp.feet_off_ground_penalty, weight=-1.0,
        params={"foot_contact_cfg": SceneEntityCfg("foot_contact_forces")},
    )
    foot_spread = RewTerm(
        func=mdp.foot_pair_spread_penalty, weight=-5.0,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "min_spread": 0.12,
        },
    )


# ---------------------------------------------------------------------------
# Terminations — add success condition
# ---------------------------------------------------------------------------

@configclass
class LauncherTerminationsCfg(Pi1TerminationsCfg):
    """Adds launcher_success on top of V3 terminations.

    Episode ends immediately (with high terminal reward from early_termination
    penalty being flipped positive via launcher_success reward) when the ball
    reaches the target window. This trains pi1 to be fast and accurate.
    """

    # Shortened timeout: 10 s = 500 steps. Launcher should hit target quickly.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Success: ball apex enters [target - window, target + window]
    launcher_success = DoneTerm(
        func=mdp.ball_apex_in_window,
        params={
            "window": _LAUNCH_WINDOW,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.34,
            "nominal_height": 0.40,
        },
    )


# ---------------------------------------------------------------------------
# Top-level env configs
# ---------------------------------------------------------------------------

@configclass
class BallJuggleLauncherEnvCfg(ManagerBasedRLEnvCfg):
    """V4 launcher pi1 training env."""

    # Reuse scene/obs/actions/events from V3 pi1 unchanged
    scene: BallJugglePi1SceneCfg = BallJugglePi1SceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # Launcher-specific rewards and terminations
    rewards: LauncherRewardsCfg = LauncherRewardsCfg()
    terminations: LauncherTerminationsCfg = LauncherTerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0    # 500 steps — fast precision, not endurance
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.sim.physx.gpu_max_rigid_patch_count = 100000


@configclass
class BallJuggleLauncherEnvCfg_PLAY(BallJuggleLauncherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.5
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 30.0   # longer for demo
        # Disable training-only termination — don't end episode when apex hits window
        self.terminations.launcher_success = None
