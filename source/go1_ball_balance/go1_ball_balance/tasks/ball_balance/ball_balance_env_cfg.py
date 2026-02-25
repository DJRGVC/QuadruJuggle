"""Environment configuration for Go1 ball-balance task (Phase 1 — privileged state).

Scene:
  - Unitree Go1 on flat ground
  - A paddle: flat rectangular plate (0.26 m × 0.20 m × 0.008 m) spawned as a
    kinematic rigid body.  A custom interval event tracks the robot trunk each
    policy step and moves the paddle to trunk_pos + _PADDLE_OFFSET_B.
  - A ping-pong ball: 40 mm sphere, mass 2.7 g, low restitution so it doesn't
    bounce off immediately.

Observations (policy):
  - ball_pos_in_paddle_frame  (3)  — XYZ of ball relative to paddle centre
  - ball_vel_in_paddle_frame  (3)  — linear velocity of ball in trunk frame
  - base_lin_vel              (3)  — robot base linear velocity (body frame)
  - base_ang_vel              (3)  — robot base angular velocity (body frame)
  - projected_gravity         (3)  — gravity vector in body frame (tilt indicator)
  - joint_pos (relative)     (12)  — joint positions minus default
  - joint_vel                (12)  — joint velocities
  Total: 39 dimensions

Actions:
  - 12-DOF joint position targets (same as flat-terrain locomotion)

Rewards:
  - ball_on_paddle_exp        +1.0   Gaussian kernel on XY distance, std=0.1 m
  - body_lin_vel_penalty      -0.1   penalise trunk lateral / forward motion
  - body_ang_vel_penalty      -0.05  penalise trunk rotation
  - action_rate_l2            -0.01  smooth joint commands
  - joint_torques_l2          -2e-4  joint effort penalty

Terminations:
  - ball_off_paddle  (radius 0.15 m)
  - ball_below_paddle
  - base_contact     (trunk hits ground)
  - time_out         (episode_length_s)
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

# Path to local assets directory (QuadruJuggle/assets/paddle/)
_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__),          # .../tasks/ball_balance/
    "..", "..", "..", "..", "..",        # up to QuadruJuggle/
    "assets", "paddle",
)
_ASSETS_DIR = os.path.normpath(_ASSETS_DIR)

# ---------------------------------------------------------------------------
# Paddle geometry constants (must match wherever the ball is reset)
# ---------------------------------------------------------------------------
# Offset of the paddle centre from the robot trunk origin, in the trunk body
# frame (x=forward, y=left, z=up).  Tune this single number if the paddle sits
# too high or too low on the robot's back.
_PADDLE_OFFSET_B = (0.0, 0.0, 0.045)   # metres, body frame
# Paddle radius (0.125 m → 0.25 m diameter / 2 = 0.125 m)
_PADDLE_HALF_EXTENT = 0.085           # metres  (170 mm diameter)
# Ball radius
_BALL_RADIUS = 0.020                  # metres (40 mm ping-pong ball)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class BallBalanceSceneCfg(InteractiveSceneCfg):
    """Scene: Go1 + thin paddle box + ping-pong ball on flat ground."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Go1 robot
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Ping-pong ball — free rigid body
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),   # 2.7 g
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.85,   # ping-pong ball bounces
                static_friction=0.3,
                dynamic_friction=0.3,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.9, 0.0),   # yellow
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Drop from 1 m above the robot: trunk z≈0.40m + paddle offset 0.07m
            # + half-thickness 0.025m + ball radius 0.020m + 1.0m drop = 1.515m
            pos=(0.0, 0.0, 0.58),
        ),
    )

    # Paddle — kinematic disc (64-sided polygon mesh) tracked to trunk.
    # Mesh generated by scripts/create_disc_usd.py:
    #   radius=0.0625 m (0.125 m diameter), thickness=10 mm, 256 triangles.
    # Physics APIs (RigidBodyAPI, CollisionAPI) are baked into disc.usda so
    # Isaac Lab can find and manage the rigid body correctly.
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
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.53),
        ),
    )

    # Contact sensor on trunk for base-contact termination
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        history_length=3,
        track_air_time=False,
    )


# ---------------------------------------------------------------------------
# MDP: Observations
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Ball state relative to paddle
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
        # Robot proprioception
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# MDP: Actions
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
# MDP: Events (reset / randomisation)
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    # Reset robot joints to default
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Reset robot base pose — pinned to env origin for Phase 1 static balance.
    # XY fixed at zero so ball always spawns above paddle centre.
    # Yaw randomised slightly so policy generalises across headings.
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
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Snap paddle to trunk position at episode reset
    reset_paddle = EventTerm(
        func=mdp.reset_paddle_pose,
        mode="reset",
        params={
            "paddle_cfg": SceneEntityCfg("paddle"),
            "offset_b": _PADDLE_OFFSET_B,
        },
    )

    # Track trunk with paddle every policy step (kinematic body)
    update_paddle = EventTerm(
        func=mdp.update_paddle_pose,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "paddle_cfg": SceneEntityCfg("paddle"),
            "robot_cfg": SceneEntityCfg("robot"),
            "offset_b": _PADDLE_OFFSET_B,
        },
    )

    # Reset ball directly onto the paddle surface at zero velocity.
    # No drop → no bounce → immediate ball_on_paddle signal from step 1.
    reset_ball = EventTerm(
        func=mdp.reset_ball_on_paddle,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "ball_radius": _BALL_RADIUS,
            "xy_offset_range": 0.02,
        },
    )


# ---------------------------------------------------------------------------
# MDP: Rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    # Primary: keep ball centred on paddle
    ball_on_paddle = RewTerm(
        func=mdp.ball_on_paddle_exp,
        weight=2.0,
        params={
            "std": 0.25,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.28,       # gate: 0 reward when trunk below this
            "nominal_height": 0.40,   # gate: full reward at standing height
        },
    )

    # Shaping: penalise lateral ball velocity (gradient when ball starts sliding)
    ball_lateral_vel = RewTerm(
        func=mdp.ball_lateral_vel_penalty,
        weight=-1.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "min_height": 0.28,
            "nominal_height": 0.40,
        },
    )

    # Shaping: penalise ball being airborne above paddle (encourages damping bounces)
    ball_height = RewTerm(
        func=mdp.ball_height_penalty,
        weight=-2.0,
        params={
            "ball_radius": _BALL_RADIUS,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.28,
            "nominal_height": 0.40,
        },
    )

    # Shaping: penalise trunk dropping below standing height (prevents collapsing)
    base_height = RewTerm(
        func=mdp.base_height_penalty,
        weight=-20.0,
        params={
            "min_height": 0.28,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    # Shaping: penalise trunk rolling/pitching (prevents the lean-to-side exploit)
    trunk_tilt = RewTerm(
        func=mdp.trunk_tilt_penalty,
        weight=-2.0,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # Shaping: discourage body motion (stay still to balance ball)
    body_lin_vel = RewTerm(
        func=mdp.body_lin_vel_penalty,
        weight=-0.1,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    body_ang_vel = RewTerm(
        func=mdp.body_ang_vel_penalty,
        weight=-0.05,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # Regularisation
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


# ---------------------------------------------------------------------------
# MDP: Terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    ball_off_paddle = DoneTerm(
        func=mdp.ball_off_paddle,
        params={
            "radius": 0.30,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
        },
    )

    ball_below_paddle = DoneTerm(
        func=mdp.ball_below_paddle,
        params={
            "min_height_offset": -0.05,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
        },
    )

    # base_contact intentionally removed from training — early in training the
    # robot always falls, which causes instant termination before the ball ever
    # reaches the paddle, giving zero learning signal.  The height penalty
    # (base_height in RewardsCfg) discourages collapsing instead.
    # base_contact is re-enabled in BallBalanceEnvCfg_PLAY for evaluation.


# ---------------------------------------------------------------------------
# Top-level environment config
# ---------------------------------------------------------------------------

@configclass
class BallBalanceEnvCfg(ManagerBasedRLEnvCfg):
    scene: BallBalanceSceneCfg = BallBalanceSceneCfg(num_envs=12288, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4              # policy runs at 50 Hz (sim at 200 Hz)
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # PhysX GPU contact patch buffer — default is too small for 12k+ envs
        # with ball+paddle+robot contacts.  Set well above the observed peak (~255k).
        self.sim.physx.gpu_max_rigid_patch_count = 400000


@configclass
class BallBalanceEnvCfg_PLAY(BallBalanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.5
        self.observations.policy.enable_corruption = False
        # Spawn ball perfectly centred for play evaluation
        self.events.reset_ball.params["xy_offset_range"] = 0.0
        # Don't end the episode just because an early policy squats — lets us
        # observe ball behaviour even with an undertrained policy
        self.terminations.base_contact = None