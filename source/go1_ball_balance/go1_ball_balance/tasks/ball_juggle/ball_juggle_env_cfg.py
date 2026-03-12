"""Environment configuration for Go1 ball-juggle task.

Goal: The Go1 must bounce a ping-pong ball to a consistent target apex height
above its back paddle.  The robot starts with a 5 cm drop (ball arrives at
~1 m/s) and must learn to deliver controlled upward energy on each bounce so
the ball reaches the target.

Scene:
  - Unitree Go1 on flat ground
  - Kinematic paddle tracked to trunk at 200 Hz (shared with ball_balance)
  - Ping-pong ball: 40 mm sphere, spawned 5 cm above paddle centre

Observations (policy):
  - ball_pos_in_paddle_frame  (3)  — relative XYZ including height above paddle
  - ball_vel_in_paddle_frame  (3)  — relative velocity; z-component signals apex
  - base_lin_vel              (3)
  - base_ang_vel              (3)
  - projected_gravity         (3)
  - joint_pos (relative)     (12)
  - joint_vel                (12)
  - target_apex_height        (1)  — normalised curriculum stage target
  Total: 40 dimensions

Actions:
  - 12-DOF joint position targets

Rewards:
  - alive              +1.0   per-step survival
  - base_height        -5.0   penalise trunk below 0.34 m
  - early_termination -200.0  one-shot on ball_off / ball_below
  - ball_apex_height  +10.0   Gaussian at target height above paddle surface
  - ball_xy_dist       -1.0   linear XY distance penalty (lighter than balance)
  - trunk_tilt         -2.0   penalise roll/pitch
  - body_lin_vel       -0.10  light trunk motion penalty
  - body_ang_vel       -0.05  light trunk rotation penalty
  - action_rate        -0.01  smooth joint commands
  - joint_torques      -2e-4  joint effort
  - base_height_max    -5.0   penalise trunk above 0.45 m (prevents leg-extension exploit)
  - foot_contact       -0.5   per foot off ground (prevents kickstand splaying)

Terminations:
  - time_out             (episode_length_s)
  - ball_off_paddle      (XY dist > 0.50 m from paddle centre)
  - ball_below_paddle    (ball drops 50 mm below paddle surface)

Curriculum (managed by scripts/rsl_rl/train_juggle.py):
  14 stages A→N with finer height steps (~0.07 m each) and σ starting wide (0.20 m)
  so the reward fires from the first bounce.  Lateral ball velocity (vel_xy_std) is
  introduced at Stage G (0.56 m) and grows to 0.18 m/s at Stage N (1.00 m).
  See train_juggle.py _BJ_STAGES for full table.
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
    os.path.dirname(__file__),          # .../tasks/ball_juggle/
    "..", "..", "..", "..", "..",        # up to QuadruJuggle/
    "assets", "paddle",
))

# ---------------------------------------------------------------------------
# Geometry constants (identical to ball_balance)
# ---------------------------------------------------------------------------
_PADDLE_OFFSET_B  = (0.0, 0.0, 0.070)   # body frame, metres
_PADDLE_HALF_EXTENT = 0.085              # 170 mm diameter disc
_BALL_RADIUS      = 0.020                # 40 mm ping-pong ball


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class BallJuggleSceneCfg(InteractiveSceneCfg):
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),   # 2.7 g
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.85,
                # "max" combine mode: effective_r = max(ball_r, paddle_r) = max(0.85, 0.0)
                # = 0.85 regardless of paddle USD material.  See ball_balance notes.
                restitution_combine_mode="max",
                static_friction=0.3,
                dynamic_friction=0.3,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.6, 1.0),   # blue — visually distinct from balance (yellow)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # trunk z≈0.40 + paddle_offset 0.07 + ball_radius 0.02 + drop 0.05 = 0.54
            pos=(0.0, 0.0, 0.54),
        ),
    )

    # Kinematic disc paddle — same USD asset as ball_balance
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

    # Foot contact sensors — all four feet via regex.
    # Used by feet_off_ground_penalty to detect the "kickstand" exploit:
    # the robot splaying one leg sideways for lateral stability without
    # maintaining proper ground contact.  Normal juggling thrust keeps all
    # feet planted (ground reaction force is the energy source for the upward
    # impulse); a kickstand foot indicates gaming rather than learned control.
    foot_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
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
        # Ball state relative to paddle (includes height — z component is key for juggling)
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
        base_lin_vel    = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel    = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos       = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel       = ObsTerm(func=mdp.joint_vel_rel)
        # Curriculum stage signal: lets policy generalise across target heights
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

    # Ball spawned 5 cm above paddle with very small XY offset.
    # With e=0.85, this bounce naturally reaches ~3.6 cm (< Stage A target of 10 cm),
    # so the robot must actively add energy from the first step.
    # Curriculum (train_juggle.py) advances xy_std in lock-step with target height.
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
            "vel_xy_std": 0.0,       # Stage A default; curriculum increases this from Stage G
        },
    )


# ---------------------------------------------------------------------------
# MDP: Rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    # ── Survival ──────────────────────────────────────────────────────────────

    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    base_height = RewTerm(
        func=mdp.base_height_penalty,
        weight=-5.0,
        params={"min_height": 0.34, "robot_cfg": SceneEntityCfg("robot")},
    )

    early_termination = RewTerm(func=mdp.early_termination_penalty, weight=-200.0)

    # ── Juggling ──────────────────────────────────────────────────────────────

    # Gaussian reward centred at target_height above the paddle surface.
    # Fires on every step while the ball is near the target height, whether
    # ascending or descending — rewarding controlled, repeated bouncing.
    # Curriculum advances target_height: 0.10 → 0.20 → 0.30 → 0.45 → 0.60 m.
    # Gaussian std tightens from 0.10 → 0.05 m to require precision at stage E.
    ball_apex_height = RewTerm(
        func=mdp.ball_apex_height_reward,
        weight=25.0,
        params={
            "target_height": 0.10,    # Stage A default; updated by train_juggle.py
            "std": 0.10,              # Stage A default; updated by train_juggle.py
            "ball_radius": _BALL_RADIUS,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.34,
            "nominal_height": 0.40,
        },
    )

    # Light XY centering gradient — constant pull toward paddle centre at all offsets.
    # Weight -1.0 (half of ball_balance -2.0): lateral drift is expected during high
    # bounces so the penalty should not compete with the juggling reward.
    ball_xy_dist = RewTerm(
        func=mdp.ball_xy_dist_penalty,
        weight=-1.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.34,
            "nominal_height": 0.40,
        },
    )

    # No ball_lateral_vel penalty: lateral velocity is expected and intentional
    # during the bounce arc and should not be suppressed.

    # ── Posture ───────────────────────────────────────────────────────────────

    trunk_tilt = RewTerm(
        func=mdp.trunk_tilt_penalty,
        weight=-2.0,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # ── Regularisation ────────────────────────────────────────────────────────

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

    # ── Anti-exploit penalties ─────────────────────────────────────────────────

    # Prevent leg-extension exploit: robot fully extends all 4 legs to raise
    # trunk to ~0.47-0.50 m (vs natural 0.40 m) for a more stable platform.
    # Ceiling set at 0.45 m (5 cm above nominal) so brief upward thrust during
    # juggling (~0.42-0.43 m peak) is not penalised.  Linear penalty above ceiling.
    # Weight -5.0: at 5 cm over (0.50 m trunk) = -0.25/step = -375/episode.
    base_height_max = RewTerm(
        func=mdp.base_height_max_penalty,
        weight=-8.0,
        params={"max_height": 0.43, "robot_cfg": SceneEntityCfg("robot")},
    )

    # Prevent kickstand exploit: robot splays one leg sideways for lateral
    # stability, losing ground contact.  Penalises count of airborne feet (0-4).
    # Weight -0.5/foot: one kickstand foot = -0.5/step = -750/episode (≈50% of
    # alive reward).  Strong deterrent without dominating the juggling signal.
    # Academic basis: DribbleBot (Ji et al. ICRA 2023) — stable foot-ground
    # contact required throughout manipulation to generate controlled impulses.
    foot_contact = RewTerm(
        func=mdp.feet_off_ground_penalty,
        weight=-3.0,
        params={"foot_contact_cfg": SceneEntityCfg("foot_contact_forces")},
    )


# ---------------------------------------------------------------------------
# MDP: Terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Same 0.50 m radius as ball_balance — high bounces (Stage E, 60 cm) drift
    # laterally before descending; terminating at 85 mm paddle edge would be too tight.
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
class BallJuggleEnvCfg(ManagerBasedRLEnvCfg):
    scene: BallJuggleSceneCfg = BallJuggleSceneCfg(num_envs=12288, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4              # policy at 50 Hz (sim at 200 Hz)
        self.episode_length_s = 30.0    # 1 500 steps
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.sim.physx.gpu_max_rigid_patch_count = 400000
        self.sim.physx.enable_ccd = True


@configclass
class BallJuggleEnvCfg_PLAY(BallJuggleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.5
        self.observations.policy.enable_corruption = False
        # Fixed-centre spawn for evaluation
        self.events.reset_ball.params["xy_std"] = 0.0
        self.events.reset_ball.params["drop_height_std"] = 0.0
        # Use final-stage (Stage G) reward parameters so the policy is evaluated
        # at the hardest curriculum stage and play videos show the 1 m apex target.
        # Without this the play script uses Stage A defaults (target=0.10 m, std=0.10)
        # which makes the robot look like it's barely bouncing the ball.
        self.rewards.ball_apex_height.params["target_height"] = 1.00
        self.rewards.ball_apex_height.params["std"] = 0.05
