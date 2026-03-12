"""Environment configuration for Go1 torso-tracking task (pi2).

Goal: The Go1 must track randomised 8D torso pose/velocity commands
(height, height velocity, roll, pitch, roll rate, pitch rate,
body-frame vx, body-frame vy) using its 12 joint actuators.  This is
the low-level policy in the hierarchical architecture: pi1 (ball
planner) outputs 8D commands, pi2 (this task) converts them to joint
targets.

Scene:
  - Unitree Go1 on flat ground
  - Kinematic paddle tracked to trunk at 200 Hz (same as ball tasks —
    keeps the visual consistent and ensures pi2 trains with the same
    inertia distribution it will see at deployment under pi1)
  - Foot contact sensors for anti-exploit penalty
  - Target marker (translucent disc) showing commanded height/tilt

Observations (53D):
  - torso_command (normalized)  (8)  — target pose/velocity
  - base_lin_vel               (3)  + noise ±0.1
  - base_ang_vel               (3)  + noise ±0.2
  - projected_gravity          (3)  + noise ±0.05
  - joint_pos (relative)      (12)  + noise ±0.01
  - joint_vel                 (12)  + noise ±1.5
  - last_action               (12)  — critical for gait coordination

Actions (12D):
  - Joint position targets (scale=0.25, default offset)

Rewards (aligned with Isaac Lab Go1 flat terrain velocity tracking config):
  POSITIVE (velocity tracking — only way to earn reward):
  - vx_tracking        +1.5   Gaussian on body vx (std=0.50, Isaac Lab standard)
  - vy_tracking        +1.5   Gaussian on body vy (std=0.50)
  - feet_air_time      +0.25  gait encouragement (Isaac Lab Go1 flat: 0.25)
  NEGATIVE (velocity error — constant gradient to walk):
  - vxy_error          -4.0   linear L2 on (v_actual - v_cmd); fills Gaussian flat tail
  NEGATIVE (pose tracking as squared-error penalties):
  - height_error       -2.0   (z - h_target)^2
  - height_vel_error   -0.2   (z_dot - h_dot_target)^2
  - roll_error         -1.0   (roll - roll_target)^2
  - pitch_error        -1.0   (pitch - pitch_target)^2
  - roll_rate_error    -0.05  (omega_roll - target)^2
  - pitch_rate_error   -0.05  (omega_pitch - target)^2
  NEGATIVE (regularization — Isaac Lab standard set):
  - base_height        -1.0   safety net below 0.17m
  - base_height_max    -1.0   safety net above 0.53m
  - lin_vel_z          -2.0   vertical bounce damping
  - ang_vel_xy         -0.05  roll/pitch rate penalty
  - action_rate        -0.01  smooth joint commands
  - joint_torques      -2e-4  joint effort
  - dof_acc            -2.5e-7 joint acceleration smoothing
  NOTE: No foot_contact penalty (Isaac Lab doesn't use one for locomotion;
  it penalizes foot lifting which kills walking)

Terminations:
  - time_out: 20s (1000 steps)
  - trunk_collapsed: trunk z < 0.15m

Curriculum (managed by train_torso_tracking.py):
  3 stages: A (walk) → B (mild pose) → C (full 8D).  Full vxy from the start.
  A→B: timeout ≥ 85% AND vxy_error > -0.80 (proves actual walking).
  B→C: timeout ≥ 85% for 100 iters.  Resample interval 5-10s (Isaac Lab: 10s).
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

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
            usd_path=f"{_ASSETS_DIR}/disc.usda",
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

    # Commanded velocity arrow — cyan, placed adjacent to paddle.
    # Points in the direction of the commanded body-frame vx/vy.
    cmd_velocity_arrow = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CmdVelocityArrow",
        spawn=sim_utils.CylinderCfg(
            radius=0.006,
            height=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.8, 1.0),   # cyan = commanded
                opacity=0.9,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )

    # Actual velocity arrow — orange, placed adjacent to paddle.
    # Points in the direction of the actual body-frame vx/vy.
    actual_velocity_arrow = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ActualVelocityArrow",
        spawn=sim_utils.CylinderCfg(
            radius=0.006,
            height=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),   # orange = actual
                opacity=0.9,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        history_length=3,
        track_air_time=False,
    )

    foot_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        history_length=3,
        track_air_time=True,
    )


# ---------------------------------------------------------------------------
# MDP: Observations (41D)
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 8D torso command (normalized to ~[-1,1])
        torso_command = ObsTerm(func=mdp.torso_command_obs)
        # Robot proprioception (33D — matches Isaac Lab velocity tracking)
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos         = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel         = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # Last action (12D — Isaac Lab includes this; critical for gait coordination)
        actions           = ObsTerm(func=mdp.last_action)

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

    # Resample torso commands periodically.
    # Isaac Lab resamples velocity every 10s.  Slower = more time to respond = easier.
    resample_commands_interval = EventTerm(
        func=mdp.resample_torso_commands,
        mode="interval",
        interval_range_s=(5.0, 10.0),
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

    # Update velocity arrows (commanded + actual) adjacent to paddle
    update_velocity_arrows = EventTerm(
        func=mdp.update_velocity_arrows_pose,
        mode="interval",
        interval_range_s=(0.005, 0.005),   # 200 Hz
        params={
            "cmd_arrow_cfg": SceneEntityCfg("cmd_velocity_arrow"),
            "actual_arrow_cfg": SceneEntityCfg("actual_velocity_arrow"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "paddle_radius": 0.085,
            "margin": 0.20,
        },
    )

    # Circle command pattern (only active when env._torso_circle_enabled=True)
    circle_commands = EventTerm(
        func=mdp.update_circle_commands,
        mode="interval",
        interval_range_s=(0.005, 0.005),   # 200 Hz — checks internally for policy step
    )

    # -- Domain Randomization --
    # MINIMAL DR during walking acquisition (matched to Isaac Lab Go1 rough).
    # Isaac Lab Go1 only randomizes trunk mass (+3/-1 kg, additive) — nothing else.
    # Aggressive DR (motor gains, joint friction, external forces) PREVENTED
    # walking from emerging.  Re-enable in Stage C after gait is stable.

    # Trunk mass only (Isaac Lab Go1: (-1.0, 3.0) additive on trunk)
    randomize_robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
        },
    )

    # Foot-ground friction — keep but tighten to Isaac Lab range
    randomize_foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    # DISABLED until walking is stable:
    # randomize_robot_com — CoM shifts
    # randomize_actuator_gains — motor PD gain variation
    # randomize_joint_friction — joint friction variation
    # randomize_paddle_material — paddle surface
    # push_robot — velocity perturbation at reset
    # external_wrench — persistent external forces

    # External force/torque — DISABLED (zeroed out)
    external_wrench = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
        },
    )


# ---------------------------------------------------------------------------
# MDP: Rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    # ══════════════════════════════════════════════════════════════════════════
    # POSITIVE REWARDS — velocity tracking (Isaac Lab Go1 flat terrain config)
    # Standing still earns 0 positive reward.  Moving is the only way to earn.
    # ══════════════════════════════════════════════════════════════════════════
    vx_tracking = RewTerm(
        func=mdp.vx_tracking_reward,
        weight=1.5,
        # Per-axis Gaussian.  std=0.50 = Isaac Lab's sqrt(0.25).
        # gate_std=10.0 = effectively no gate (velocity is ungated).
        params={"std": 0.50, "min_cmd": 0.10, "gate_std": 10.0, "robot_cfg": SceneEntityCfg("robot")},
    )
    vy_tracking = RewTerm(
        func=mdp.vy_tracking_reward,
        weight=1.5,
        params={"std": 0.50, "min_cmd": 0.10, "gate_std": 10.0, "robot_cfg": SceneEntityCfg("robot")},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_reward,
        weight=0.25,
        # Isaac Lab Go1 flat: 0.25.  Critical for gait emergence — encourages
        # lifting feet instead of shuffling.  Was 0.01 (way too low).
        params={
            "threshold": 0.5,
            "min_cmd": 0.10,
            "foot_contact_cfg": SceneEntityCfg("foot_contact_forces"),
        },
    )

    # ══════════════════════════════════════════════════════════════════════════
    # NEGATIVE PENALTIES — velocity error (constant gradient to walk)
    # The Gaussian tracking rewards have flat tails — standing still barely
    # hurts.  This linear penalty provides constant gradient at ALL errors.
    # ══════════════════════════════════════════════════════════════════════════
    vxy_error = RewTerm(
        func=mdp.vxy_error_penalty,
        weight=-4.0,
        # L2 norm of (v_actual - v_cmd) in XY.  Masked when cmd near zero.
        # Provides the "stick" to complement the Gaussian "carrot".
        params={"min_cmd": 0.10, "robot_cfg": SceneEntityCfg("robot")},
    )

    # ══════════════════════════════════════════════════════════════════════════
    # NEGATIVE PENALTIES — pose tracking (squared error, 0 at perfect)
    # In Phase 1 (walking), pose commands are fixed (h=0.38, roll/pitch=0,
    # rates=0) so these act like Isaac Lab's flat_orientation_l2: keep upright.
    # In Phase 2 they track varying 8D commands.
    # ══════════════════════════════════════════════════════════════════════════
    height_error = RewTerm(
        func=mdp.height_error_penalty,
        weight=-2.0,
        # Reduced from -5.0.  Acts like flat_orientation during walking.
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    height_vel_error = RewTerm(
        func=mdp.height_vel_error_penalty,
        weight=-0.2,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    roll_error = RewTerm(
        func=mdp.roll_error_penalty,
        weight=-1.0,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    pitch_error = RewTerm(
        func=mdp.pitch_error_penalty,
        weight=-1.0,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    roll_rate_error = RewTerm(
        func=mdp.roll_rate_error_penalty,
        weight=-0.05,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    pitch_rate_error = RewTerm(
        func=mdp.pitch_rate_error_penalty,
        weight=-0.05,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # ══════════════════════════════════════════════════════════════════════════
    # NEGATIVE PENALTIES — regularization (Isaac Lab standard set)
    # Matched to Isaac Lab Go1 rough/flat weights.  No foot_contact penalty
    # (Isaac Lab doesn't use one for locomotion — it kills walking).
    # ══════════════════════════════════════════════════════════════════════════
    base_height = RewTerm(
        func=mdp.base_height_penalty,
        weight=-1.0,
        # Safety net only.  Reduced from -5.0.
        params={"min_height": 0.17, "robot_cfg": SceneEntityCfg("robot")},
    )
    base_height_max = RewTerm(
        func=mdp.base_height_max_penalty,
        weight=-1.0,
        # Safety net only.  Reduced from -5.0.
        params={"max_height": 0.53, "robot_cfg": SceneEntityCfg("robot")},
    )
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
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
        params={"minimum_height": 0.08, "grace_steps": 20, "asset_cfg": SceneEntityCfg("robot")},
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
        self.scene.num_envs = 5
        self.scene.env_spacing = 5.0   # wide spacing — robots walk around
        self.observations.policy.enable_corruption = False
        # Resample from full Stage C ranges every 5-10s (same as training)
        self.events.resample_commands_interval.interval_range_s = (2.0, 5.0)
        self._torso_smooth_enabled = False   # no smoothing — snap like training
        self._torso_circle_enabled = False   # use random commands, not circle pattern
