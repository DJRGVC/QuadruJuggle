"""Environment configuration for the hierarchical ball-juggle task (pi1 + frozen pi2).

Same scene, observations, rewards, and terminations as the flat ball_juggle task.
The key difference is the action space: 8D torso commands (via TorsoCommandAction)
instead of 12D joint position targets.

pi1 (this task's policy) outputs 8D torso commands.
TorsoCommandAction runs the frozen pi2 actor to produce 12D joint targets.

The pi2_checkpoint path must be set before env creation — see train_juggle_hier.py.
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

from . import mdp
from go1_ball_balance.tasks.torso_tracking.action_term import TorsoCommandActionCfg

from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip

# Path to local assets directory (shared with ball_balance)
_ASSETS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),          # .../tasks/ball_juggle_hier/
    "..", "..", "..", "..", "..",        # up to QuadruJuggle/
    "assets", "paddle",
))

# Geometry constants (identical to ball_balance/ball_juggle)
_PADDLE_OFFSET_B  = (0.0, 0.0, 0.070)
_PADDLE_HALF_EXTENT = 0.085
_BALL_RADIUS      = 0.020


# ---------------------------------------------------------------------------
# Scene — same as ball_juggle
# ---------------------------------------------------------------------------

@configclass
class BallJuggleHierSceneCfg(InteractiveSceneCfg):
    """Scene: Go1 + kinematic paddle + ping-pong ball on flat ground."""

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
                linear_damping=0.1,
                angular_damping=0.1,
                max_linear_velocity=10.0,
                max_angular_velocity=50.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.85,
                restitution_combine_mode="max",
                static_friction=0.3,
                dynamic_friction=0.3,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.8),   # purple — visually distinct from flat juggle (blue)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.54)),
    )

    paddle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Paddle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_ASSETS_DIR}/disc.usda",
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
# MDP: Observations (40D — same as flat juggle)
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        ball_pos = ObsTerm(
            func=mdp.ball_pos_in_paddle_frame,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "paddle_offset_b": _PADDLE_OFFSET_B,
            },
        )
        ball_vel = ObsTerm(
            func=mdp.ball_vel_in_paddle_frame,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos         = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel         = ObsTerm(func=mdp.joint_vel_rel)
        target_apex_height = ObsTerm(
            func=mdp.target_apex_height_obs,
            params={
                "reward_term_name": "ball_apex_height",
                "max_target_height": 1.00,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# MDP: Actions (8D — via TorsoCommandAction wrapping frozen pi2)
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    torso_cmd = TorsoCommandActionCfg(
        asset_name="robot",
        pi2_checkpoint="MUST_BE_SET_BEFORE_ENV_CREATION",
    )


# ---------------------------------------------------------------------------
# MDP: Events (same as flat juggle)
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (-0.3, 0.3),
            },
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
        params={
            "paddle_cfg": SceneEntityCfg("paddle"),
            "offset_b": _PADDLE_OFFSET_B,
        },
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

    randomize_target = EventTerm(
        func=mdp.randomize_target_apex,
        mode="reset",
        params={
            "target_min": 0.05,
            "target_max": 0.05,
            "sigma_ratio": 2.0,
        },
    )

    reset_ball = EventTerm(
        func=mdp.reset_ball_on_paddle,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "ball_radius": _BALL_RADIUS,
            "xy_std": 0.02,
            "drop_height_mean": 0.05,
            "drop_height_std": 0.005,
            "vel_xy_std": 0.0,
        },
    )

    # -- Domain Randomization (pi1-specific: ball & paddle properties) --

    # Ball mass ±20% (competition vs practice balls, 2.7g nominal)
    # Ref: Huang et al. "Goalkeeper" IROS 2023
    randomize_ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "mass_distribution_params": (0.80, 1.20),
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("ball"),
        },
    )

    # Ball restitution [0.75, 0.95] (ball wear, temperature, brand variation)
    # Ref: Ji et al. "DribbleBot" ICRA 2023 — ball property DR
    randomize_ball_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "static_friction_range": (0.2, 0.5),
            "dynamic_friction_range": (0.2, 0.5),
            "restitution_range": (0.75, 0.95),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("ball"),
        },
    )

    # Paddle surface friction [0.2, 0.8] (surface wear, moisture, material)
    randomize_paddle_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "static_friction_range": (0.2, 0.8),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("paddle"),
        },
    )


# ---------------------------------------------------------------------------
# MDP: Rewards (same as flat juggle)
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    base_height = RewTerm(
        func=mdp.base_height_penalty,
        weight=-5.0,
        params={"min_height": 0.17, "robot_cfg": SceneEntityCfg("robot")},
    )

    early_termination = RewTerm(func=mdp.early_termination_penalty, weight=-200.0)

    ball_apex_height = RewTerm(
        func=mdp.ball_apex_height_reward,
        weight=25.0,
        params={
            "target_height": 0.05,
            "std": 0.025,
            "ball_radius": _BALL_RADIUS,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.20,
            "nominal_height": 0.35,
        },
    )

    ball_xy_dist = RewTerm(
        func=mdp.ball_xy_dist_penalty,
        weight=-5.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.20,
            "nominal_height": 0.35,
        },
    )

    trunk_tilt = RewTerm(
        func=mdp.trunk_tilt_penalty,
        weight=-0.5,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    body_lin_vel = RewTerm(
        func=mdp.body_lin_vel_penalty,
        weight=-0.10,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    body_ang_vel = RewTerm(
        func=mdp.body_ang_vel_penalty,
        weight=-0.05,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    base_height_max = RewTerm(
        func=mdp.base_height_max_penalty,
        weight=-5.0,
        params={"max_height": 0.53, "robot_cfg": SceneEntityCfg("robot")},
    )

    foot_contact = RewTerm(
        func=mdp.feet_off_ground_penalty,
        weight=-0.5,
        params={"foot_contact_cfg": SceneEntityCfg("foot_contact_forces")},
    )


# ---------------------------------------------------------------------------
# MDP: Terminations (same as flat juggle)
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    ball_off = DoneTerm(
        func=mdp.ball_off_paddle,
        params={
            "radius": 0.50,
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
# Top-level environment configs
# ---------------------------------------------------------------------------

@configclass
class BallJuggleHierEnvCfg(ManagerBasedRLEnvCfg):
    scene: BallJuggleHierSceneCfg = BallJuggleHierSceneCfg(num_envs=12288, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.sim.physx.gpu_max_rigid_patch_count = 400000
        self.sim.physx.enable_ccd = True


@configclass
class BallJuggleHierEnvCfg_PLAY(BallJuggleHierEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.5
        self.observations.policy.enable_corruption = False
        self.events.reset_ball.params["xy_std"] = 0.0
        self.events.reset_ball.params["drop_height_std"] = 0.0
        # Randomize target heights across envs for visual variety
        self.events.randomize_target.params["target_min"] = 0.30
        self.events.randomize_target.params["target_max"] = 0.92
        self.events.randomize_target.params["sigma_ratio"] = 2.5
