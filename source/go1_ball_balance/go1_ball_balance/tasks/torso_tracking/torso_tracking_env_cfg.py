"""Environment configuration for Go1 torso-tracking task (pi2).

Goal: The Go1 must track randomised 6D torso pose/velocity commands
(height, height velocity, roll, pitch, roll rate, pitch rate) using
its 12 joint actuators.  This is the low-level policy in the
hierarchical architecture: pi1 (ball planner) outputs 6D commands,
pi2 (this task) converts them to joint targets.

Scene:
  - Unitree Go1 on flat ground
  - Kinematic paddle tracked to trunk at 200 Hz (same as ball tasks —
    keeps the visual consistent and ensures pi2 trains with the same
    inertia distribution it will see at deployment under pi1)
  - Foot contact sensors for anti-exploit penalty
  - Target marker (translucent disc) showing commanded height/tilt

Observations (39D):
  - torso_command (normalized)  (6)  — target pose/velocity
  - base_lin_vel               (3)
  - base_ang_vel               (3)
  - projected_gravity          (3)
  - joint_pos (relative)      (12)
  - joint_vel                 (12)

Actions (12D):
  - Joint position targets (scale=0.25, default offset)

Rewards:
  - alive              +1.0   per-step survival
  - height_tracking    +5.0   Gaussian on trunk z vs h_target
  - height_vel_tracking +2.0  Gaussian on z_dot vs h_dot_target
  - roll_tracking      +3.0   Gaussian on roll vs roll_target
  - pitch_tracking     +3.0   Gaussian on pitch vs pitch_target
  - roll_rate_tracking +1.5   Gaussian on omega_roll vs target
  - pitch_rate_tracking +1.5  Gaussian on omega_pitch vs target
  - base_height        -5.0   penalise trunk below 0.22m
  - base_height_max    -5.0   penalise trunk above 0.53m
  - foot_contact       -0.5   per foot off ground
  - action_rate        -0.01  smooth joint commands
  - joint_torques      -2e-4  joint effort

Terminations:
  - time_out: 20s (1000 steps)
  - trunk_collapsed: trunk z < 0.15m

Curriculum (managed by train_torso_tracking.py):
  3 stages A→C: narrow → full command ranges.
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

from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip

# Path to local assets directory (shared with ball_balance)
_ASSETS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),          # .../tasks/torso_tracking/
    "..", "..", "..", "..", "..",        # up to QuadruJuggle/
    "assets", "paddle",
))

# Geometry constants (identical to ball_balance)
_PADDLE_OFFSET_B = (0.0, 0.0, 0.070)   # body frame, metres


# ---------------------------------------------------------------------------
# Scene — Go1 on flat ground + paddle + target marker
# ---------------------------------------------------------------------------

@configclass
class TorsoTrackingSceneCfg(InteractiveSceneCfg):
    """Scene: Go1 + kinematic paddle + target marker on flat ground."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Kinematic paddle — same USD asset as ball_balance.
    # Even though no ball is present, having the paddle:
    # 1. Keeps the visual consistent with deployment under pi1
    # 2. Adds the same inertia distribution pi2 will see later
    paddle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Paddle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_ASSETS_DIR}/dumbbell.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.53)),
    )

    # Target marker — translucent green disc (larger than real paddle) showing
    # the commanded height/tilt.  A thin cylinder "normal arrow" sticks up from
    # the centre so you can see the tilt direction at a glance.
    # Kinematic, no collision — purely visual.  Updated by update_target_marker event.
    target_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetMarker",
        spawn=sim_utils.CylinderCfg(
            radius=0.13,               # ~1.5× the real paddle (0.085)
            height=0.005,              # very thin disc
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.9, 0.2),
                opacity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.77)),
    )

    # Normal arrow — thin cylinder pointing "up" from the target disc centre.
    # Attached to the same commanded pose so you can see tilt direction clearly.
    target_normal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetNormal",
        spawn=sim_utils.CylinderCfg(
            radius=0.004,              # thin stick
            height=0.15,               # 15 cm arrow
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 0.2),
                opacity=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.85)),
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
# MDP: Observations (39D)
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 6D torso command (normalized to ~[-1,1])
        torso_command = ObsTerm(func=mdp.torso_command_obs)
        # Robot proprioception (33D)
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos         = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel         = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# MDP: Actions (12D)
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# MDP: Events
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

    # Paddle tracking — same as ball_balance/ball_juggle
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
        interval_range_s=(0.005, 0.005),   # 200 Hz — physics rate
        params={
            "paddle_cfg": SceneEntityCfg("paddle"),
            "robot_cfg": SceneEntityCfg("robot"),
            "offset_b": _PADDLE_OFFSET_B,
        },
    )

    # Resample torso commands on episode reset (always snaps, even in smooth mode)
    resample_commands_reset = EventTerm(
        func=mdp.resample_torso_commands_reset,
        mode="reset",
    )

    # Resample torso commands periodically (1–4s interval)
    resample_commands_interval = EventTerm(
        func=mdp.resample_torso_commands,
        mode="interval",
        interval_range_s=(0.15, 0.5),
    )

    # Smooth command blending (runs every physics step, only active when
    # env._torso_smooth_enabled is True — set by play config)
    smooth_commands = EventTerm(
        func=mdp.update_torso_commands_smooth,
        mode="interval",
        interval_range_s=(0.005, 0.005),   # 200 Hz — every physics step
    )

    # Update target disc + normal arrow to show commanded pose
    update_target_marker = EventTerm(
        func=mdp.update_target_marker_pose,
        mode="interval",
        interval_range_s=(0.005, 0.005),   # 200 Hz
        params={
            "marker_cfg": SceneEntityCfg("target_marker"),
            "normal_cfg": SceneEntityCfg("target_normal"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
        },
    )


# ---------------------------------------------------------------------------
# MDP: Rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    # -- Survival --
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # -- Tracking rewards --
    height_tracking = RewTerm(
        func=mdp.height_tracking_reward,
        weight=5.0,
        params={"std": 0.10, "robot_cfg": SceneEntityCfg("robot")},
    )
    height_vel_tracking = RewTerm(
        func=mdp.height_vel_tracking_reward,
        weight=3.0,
        params={"std": 0.3, "robot_cfg": SceneEntityCfg("robot")},
    )
    roll_tracking = RewTerm(
        func=mdp.roll_tracking_reward,
        weight=3.0,
        params={"std": 0.05, "robot_cfg": SceneEntityCfg("robot")},
    )
    pitch_tracking = RewTerm(
        func=mdp.pitch_tracking_reward,
        weight=4.0,
        params={"std": 0.08, "robot_cfg": SceneEntityCfg("robot")},
    )
    roll_rate_tracking = RewTerm(
        func=mdp.roll_rate_tracking_reward,
        weight=2.5,
        params={"std": 0.8, "robot_cfg": SceneEntityCfg("robot")},
    )
    pitch_rate_tracking = RewTerm(
        func=mdp.pitch_rate_tracking_reward,
        weight=2.5,
        params={"std": 0.8, "robot_cfg": SceneEntityCfg("robot")},
    )

    # -- Penalties --
    base_height = RewTerm(
        func=mdp.base_height_penalty,
        weight=-5.0,
        params={"min_height": 0.22, "robot_cfg": SceneEntityCfg("robot")},
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
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


# ---------------------------------------------------------------------------
# MDP: Terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    trunk_collapsed = DoneTerm(
        func=mdp.trunk_height_collapsed,
        params={"minimum_height": 0.15, "grace_steps": 20, "asset_cfg": SceneEntityCfg("robot")},
    )


# ---------------------------------------------------------------------------
# Top-level environment configs
# ---------------------------------------------------------------------------

@configclass
class TorsoTrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene: TorsoTrackingSceneCfg = TorsoTrackingSceneCfg(num_envs=12288, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4              # policy at 50 Hz (sim at 200 Hz)
        self.episode_length_s = 20.0    # 1000 steps
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.4)
        self.sim.physx.gpu_max_rigid_patch_count = 200000


@configclass
class TorsoTrackingEnvCfg_PLAY(TorsoTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.2
        self.observations.policy.enable_corruption = False
        # Use full command ranges (Stage C) for play evaluation
        # Enable smooth command interpolation for visual clarity
        self._torso_smooth_enabled = True
