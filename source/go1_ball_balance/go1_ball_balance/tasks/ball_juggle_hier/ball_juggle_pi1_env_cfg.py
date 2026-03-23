"""Environment configuration for learned-pi1 hierarchical ball juggling.

The RL policy (pi1) outputs a 6D normalized torso command directly.
The frozen pi2 converts it to 12D joint targets.

This is the TRAINING environment for pi1. Play is done via scripts/play_pi1.py.

Differences from ball_juggle_mirror_env_cfg.py:
  - Action: LearnedPi1TorsoActionCfg (6D RL output) instead of mirror law (1D)
  - Rewards: adds ball_bouncing_reward to penalize cradling
  - Observations: adds last_action (6D pi1 output) for temporal awareness
  - num_envs: 4096 for training (mirror env uses 16 for play)
  - episode_length_s: 30s (1500 steps)
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
from dataclasses import MISSING

from . import mdp
from .pi1_learned_action import LearnedPi1TorsoActionCfg

from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip

_ASSETS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..",
    "assets", "paddle",
))

_PADDLE_OFFSET_B   = (0.0, 0.0, 0.070)
_PADDLE_HALF_EXTENT = 0.153   # 0.085 * 1.8 scale
_BALL_RADIUS       = 0.020


# ---------------------------------------------------------------------------
# Scene — identical to ball_juggle_mirror
# ---------------------------------------------------------------------------

@configclass
class BallJugglePi1SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=_BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.01,
                angular_damping=0.01,
                max_linear_velocity=10.0,
                max_angular_velocity=50.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.99,
                restitution_combine_mode="max",
                static_friction=0.3,
                dynamic_friction=0.3,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.6, 1.0),  # blue — distinct from mirror-law (orange)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.54)),
    )

    paddle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Paddle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_ASSETS_DIR}/disc.usda",
            scale=(1.8, 1.8, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.53)),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        history_length=3,
        track_air_time=False,
    )
    foot_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        history_length=3,
        track_air_time=False,
    )


# ---------------------------------------------------------------------------
# Observations — 46D (same as mirror env 40D + last_action 6D)
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Ball state — highest noise: stereo camera / optical flow
        ball_pos = ObsTerm(
            func=mdp.ball_pos_in_paddle_frame,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "paddle_offset_b": _PADDLE_OFFSET_B,
            },
            noise=GaussianNoiseCfg(std=0.010),   # ~1 cm stereo uncertainty
        )
        ball_vel = ObsTerm(
            func=mdp.ball_vel_in_paddle_frame,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
            noise=GaussianNoiseCfg(std=0.100),   # ~0.1 m/s differentiated camera
        )
        # Proprioception — moderate noise: state estimator + IMU
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,
                                    noise=GaussianNoiseCfg(std=0.050))  # state estimator
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,
                                    noise=GaussianNoiseCfg(std=0.020))  # IMU gyro
        projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                    noise=GaussianNoiseCfg(std=0.010))  # IMU accel
        joint_pos         = ObsTerm(func=mdp.joint_pos_rel,
                                    noise=GaussianNoiseCfg(std=0.010))  # encoder quantisation
        joint_vel         = ObsTerm(func=mdp.joint_vel_rel,
                                    noise=GaussianNoiseCfg(std=0.050))  # differentiated encoder
        # Command and action — no noise (internal signals)
        target_apex_height = ObsTerm(
            func=mdp.target_apex_height_obs,
            params={
                "reward_term_name": "ball_apex_height",
                "max_target_height": 1.00,
            },
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True   # applies noise above during training
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Actions — learned pi1: 6D normalized torso command → frozen pi2
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    torso_cmd = LearnedPi1TorsoActionCfg(
        asset_name="robot",
        pi2_checkpoint=MISSING,   # must be set before env creation
    )


# ---------------------------------------------------------------------------
# Events — identical to ball_juggle_mirror
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.3, 0.3)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_paddle = EventTerm(
        func=mdp.reset_paddle_pose,
        mode="reset",
        params={"paddle_cfg": SceneEntityCfg("paddle"), "offset_b": _PADDLE_OFFSET_B},
    )
    update_paddle = EventTerm(
        func=mdp.update_paddle_pose,
        mode="interval",
        interval_range_s=(0.005, 0.005),
        params={
            "paddle_cfg": SceneEntityCfg("paddle"),
            "robot_cfg": SceneEntityCfg("robot"),
            "offset_b": _PADDLE_OFFSET_B,
        },
    )
    reset_ball = EventTerm(
        func=mdp.reset_ball_on_paddle,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "ball_radius": _BALL_RADIUS,
            "xy_std": 0.01,
            "drop_height_mean": 0.05,
            "drop_height_std": 0.01,
            "vel_xy_std": 0.0,
            "vel_z_mean": 1.2,
        },
    )
    randomize_apex_height = EventTerm(
        func=mdp.randomize_apex_height,
        mode="reset",
        params={"height_range": (0.10, 0.60), "std_range": (0.08, 0.12)},
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.5)

    base_height = RewTerm(
        func=mdp.base_height_penalty,
        weight=-2.0,
        params={"min_height": 0.34, "robot_cfg": SceneEntityCfg("robot")},
    )
    early_termination = RewTerm(func=mdp.early_termination_penalty, weight=-10.0)

    ball_apex_height = RewTerm(
        func=mdp.ball_apex_height_reward,
        weight=5.0,
        params={
            "target_height": 0.30,   # fallback if randomize_apex_height not used
            "std": 0.10,
            "ball_radius": _BALL_RADIUS,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.34,
            "nominal_height": 0.40,
            "paddle_radius": _PADDLE_HALF_EXTENT,  # no reward outside paddle disc
        },
    )

    # Penalize cradling: reward high |ball_vz| to force active bouncing
    ball_bouncing = RewTerm(
        func=mdp.ball_bouncing_reward,
        weight=2.0,
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
    trunk_tilt   = RewTerm(func=mdp.trunk_tilt_penalty, weight=-3.0,
                           params={"robot_cfg": SceneEntityCfg("robot")})
    trunk_contact = RewTerm(
        func=mdp.trunk_contact_penalty,
        weight=-2.0,
        params={"contact_cfg": SceneEntityCfg("contact_forces")},
    )
    body_lin_vel = RewTerm(func=mdp.body_lin_vel_penalty, weight=-0.10,
                           params={"robot_cfg": SceneEntityCfg("robot")})
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_penalty, weight=-0.05,
                           params={"robot_cfg": SceneEntityCfg("robot")})
    # Slightly higher penalty on pi1 action rate — smooth command changes matter more
    action_rate  = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2, weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    base_height_max = RewTerm(
        func=mdp.base_height_max_penalty, weight=-3.0,
        params={"max_height": 0.43, "robot_cfg": SceneEntityCfg("robot")},
    )
    foot_contact = RewTerm(
        func=mdp.feet_off_ground_penalty, weight=-1.0,
        params={"foot_contact_cfg": SceneEntityCfg("foot_contact_forces")},
    )


# ---------------------------------------------------------------------------
# Terminations — identical to ball_juggle_mirror
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    robot_tilt = DoneTerm(
        func=mdp.robot_tilt,
        params={"max_tilt": 0.5, "grace_steps": 50, "asset_cfg": SceneEntityCfg("robot")},
    )

    ball_off = DoneTerm(
        func=mdp.ball_off_paddle,
        params={
            "radius": 0.25,   # tightened from 0.50 — ball must stay near paddle
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
        },
    )
    ball_below = DoneTerm(
        func=mdp.ball_below_paddle,
        params={
            "min_height_offset": -0.05,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
        },
    )


# ---------------------------------------------------------------------------
# Top-level env configs
# ---------------------------------------------------------------------------

@configclass
class BallJugglePi1EnvCfg(ManagerBasedRLEnvCfg):
    scene: BallJugglePi1SceneCfg = BallJugglePi1SceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0    # 1500 steps — same as mirror play
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.sim.physx.gpu_max_rigid_patch_count = 100000


@configclass
class BallJugglePi1EnvCfg_PLAY(BallJugglePi1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.5
        self.observations.policy.enable_corruption = False
