"""Environment configuration for Go1 ball-balance task.

Phase 2: Teach the robot to keep a ping-pong ball centred on its back paddle.
The robot must stand upright AND actively balance the ball simultaneously.
Episodes terminate early if the ball falls off; a -200 one-shot penalty makes
this strictly worse than any standing configuration.

Scene:
  - Unitree Go1 on flat ground
  - A paddle: kinematic rigid body tracked to trunk at 200 Hz.
  - A ping-pong ball: 40 mm sphere, spawned just above the paddle centre.

Observations (policy):
  - ball_pos_in_paddle_frame  (3)
  - ball_vel_in_paddle_frame  (3)
  - base_lin_vel              (3)
  - base_ang_vel              (3)
  - projected_gravity         (3)
  - joint_pos (relative)     (12)
  - joint_vel                (12)
  Total: 39 dimensions

Actions:
  - 12-DOF joint position targets

Rewards (Phase 2):
  - alive              +1.0   per-step survival (primary axis)
  - base_height        -5.0   penalise trunk below 0.34 m
  - early_termination -200.0  one-shot on ball_off / ball_below (> max episode alive)
  - ball_on_paddle     +8.0   Gaussian centering, std=0.25m, time_scale=4, height-gated
  - ball_lateral_vel   -1.0   penalise ball sliding, height-gated
  - ball_xy_dist       -2.0   linear XY distance from paddle centre — constant centering gradient
  - trunk_tilt         -2.0   penalise roll/pitch
  - body_lin_vel       -0.10  light trunk motion penalty
  - body_ang_vel       -0.05  light trunk rotation penalty
  - action_rate        -0.01  smooth joint commands
  - joint_torques      -2e-4  joint effort

Terminations (Phase 2):
  - time_out         (episode_length_s)
  - ball_off_paddle  (XY dist > 0.50 m from paddle centre)
  - ball_below_paddle (ball drops 50 mm below paddle surface)
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
# frame (x=forward, y=left, z=up).
#
# Go1 trunk collision box: 0.114 m full height, root at centre → top at +0.057 m.
# Paddle disc thickness: 10 mm → half-thickness 5 mm.
# offset = trunk_half_height(0.057) + paddle_half_thickness(0.005) + clearance(0.008)
#        = 0.070 m  → paddle bottom sits 8 mm above the trunk top surface.
_PADDLE_OFFSET_B = (0.0, 0.0, 0.070)   # metres, body frame
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
                restitution=0.85,
                # "max" combine mode: PhysX effective_r = max(ball_r, paddle_r).
                # The paddle is loaded from USD so its material cannot be set via
                # UsdFileCfg.  With combine_mode="max", effective_r = max(0.85, 0.0)
                # = 0.85 regardless of the paddle's default material — giving the
                # full 4–5× bounce increase vs the default "average" (0.43).
                # Real-life equivalent: carbon-fibre plate, acrylic sheet, or
                # melamine-coated ping-pong table surface on the robot back.
                restitution_combine_mode="max",
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
    #
    # UsdFileCfg does not accept physics_material — bounce is controlled via the
    # ball's restitution_combine_mode instead (see ball config below).
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

    # Foot contact sensors for anti-exploit penalties (see RewardsCfg).
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
            "position_range": (1.0, 1.0),   # no joint randomisation — robot always spawns stably upright
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

    # Track trunk with paddle every physics step (kinematic body).
    # interval = sim.dt = 0.005 s → fires at 200 Hz regardless of decimation.
    # Using the policy rate (0.02 s) was fine for standing but causes up to 15 ms
    # of paddle lag during rapid motion, which corrupts ball contact physics in
    # Phase 2.  Physics-rate tracking costs negligible extra compute.
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

    # Reset ball above the paddle with Gaussian-randomised height and XY position.
    #
    # Spawn curriculum is handled automatically by scripts/rsl_rl/train.py
    # (_bb_install_curriculum / _bb_apply_stage).  Stages A→D are defined there.
    # These params reflect Stage A (easy); train.py overwrites them at runtime.
    reset_ball = EventTerm(
        func=mdp.reset_ball_on_paddle,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "ball_radius": _BALL_RADIUS,
            "xy_std": 0.03,
            "drop_height_mean": 0.08,
            "drop_height_std": 0.02,
        },
    )


# ---------------------------------------------------------------------------
# MDP: Rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    # ── Standing / collapse prevention ────────────────────────────────────────

    # Survival: +1 every step. Makes longer episodes always better than dying.
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # Height: continuous penalty for trunk below standing height.
    base_height = RewTerm(
        func=mdp.base_height_penalty,
        weight=-5.0,
        params={
            "min_height": 0.34,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    # One-shot terminal penalty on ball_off / ball_below termination.
    # Weight -200 >> max episode alive reward (1500 steps × +1 = 1500), so
    # losing the ball is always strictly worse than any standing configuration.
    # Calibrated per Zhuang 2023 (Robot Parkour) and Ji 2023 (DribbleBot).
    early_termination = RewTerm(
        func=mdp.early_termination_penalty,
        weight=-200.0,
    )

    # ── Ball balancing ────────────────────────────────────────────────────────

    # Gaussian centering reward: exp(-d^2 / 2σ^2) measured in 3D from the
    # ideal resting position (paddle centre + ball_radius × paddle_normal).
    # Height-gated so a collapsed robot earns no ball reward.
    #
    # Sigma curriculum (ROGER RSS 2025 pattern) — tightened automatically by
    # scripts/rsl_rl/train.py in lock-step with spawn difficulty:
    #   Stage A: std=0.25m  (easy, wide kernel)
    #   Stage B: std=0.15m
    #   Stage C: std=0.10m
    #   Stage D: std=0.08m  (tight centering, full juggling)
    #
    # time_scale=4: ball reward grows 1× → 5× linearly over the episode,
    # strongly rewarding sustained balance over short lucky stretches.
    ball_on_paddle = RewTerm(
        func=mdp.ball_on_paddle_exp,
        weight=8.0,
        params={
            "std": 0.25,
            "ball_radius": _BALL_RADIUS,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.34,
            "nominal_height": 0.40,
            "time_scale": 4.0,
        },
    )

    # Penalise lateral (XY) ball velocity — gradient to damp sliding before
    # the ball reaches the paddle edge.  Height-gated.  See DribbleBot (Ji 2023).
    ball_lateral_vel = RewTerm(
        func=mdp.ball_lateral_vel_penalty,
        weight=-1.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "min_height": 0.34,
            "nominal_height": 0.40,
        },
    )

    # Linear XY-distance penalty: constant gradient at ALL offsets from paddle centre.
    # The Gaussian ball_on_paddle_exp kernel (std=0.25 m) is nearly flat at Stage A:
    # at 68 mm offset (80% of paddle radius) the reward is 96.4% of maximum — essentially
    # no centering gradient.  The Go1 also stands ~1.42° nose-up (front thigh=0.8 rad,
    # rear=1.0 rad → front hips sit 9 mm higher), creating a rearward gravity component
    # of 0.24 m/s² along the paddle.  During ball bounces this accumulates into a
    # stable off-centre equilibrium at the rear.
    # Weight raised from -2.0 → -5.0: at 68 mm offset this gives -0.34/step penalty
    # (-510 over 1500 steps), strong enough to overcome the rearward physical bias at all
    # curriculum stages.  (DribbleBot Ji 2023; ROGER Portela 2025 use similar linear term.)
    ball_xy_dist = RewTerm(
        func=mdp.ball_xy_dist_penalty,
        weight=-5.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "min_height": 0.34,
            "nominal_height": 0.40,
        },
    )

    # ball_height penalty removed: with restitution=0.85 the ball naturally
    # bounces 3-4 cm on every contact.  Penalising all airborne height would
    # fire constantly and fight the physics.  Re-introduce in Stage 2 once
    # bouncing is rewarded (target: consistent apex height rather than damp).

    # ── Posture ───────────────────────────────────────────────────────────────

    # Penalise roll/pitch via projected gravity XY magnitude.  More numerically
    # stable than quaternion deviation (Walk These Ways, Margolis 2022).
    # Weight raised from -2.0 → -4.0: Go1's default stance is 1.42° nose-up (front
    # thighs 0.8 rad vs rear 1.0 rad), creating a rearward gravity component on the
    # paddle.  Stronger tilt penalty encourages the policy to actively level the trunk.
    trunk_tilt = RewTerm(
        func=mdp.trunk_tilt_penalty,
        weight=-4.0,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # ── Regularisation ────────────────────────────────────────────────────────

    # Light trunk motion penalties: prevent body-swing exploitation (DribbleBot).
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

    # Smooth joint commands without freezing the policy.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Joint effort regularisation for hardware efficiency.
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # ── Anti-exploit penalties ─────────────────────────────────────────────────

    # Prevent leg-extension exploit: robot fully extends all 4 legs to raise
    # trunk to ~0.47-0.50 m (vs natural 0.40 m).  Balance task uses tighter
    # ceiling than juggling (0.43 m vs 0.45 m) because no active trunk thrust
    # is needed — the robot should hold a natural still stance.
    # At 4 cm over (0.47 m trunk): -0.20/step = -300/episode.
    base_height_max = RewTerm(
        func=mdp.base_height_max_penalty,
        weight=-5.0,
        params={"max_height": 0.43, "robot_cfg": SceneEntityCfg("robot")},
    )

    # Prevent kickstand exploit: one leg splayed sideways for lateral stability.
    # Penalises count of airborne feet (0-4) at weight -0.5/foot.
    # One kickstand foot = -750/episode ≈ 50% of alive reward.
    foot_contact = RewTerm(
        func=mdp.feet_off_ground_penalty,
        weight=-0.5,
        params={"foot_contact_cfg": SceneEntityCfg("foot_contact_forces")},
    )


# ---------------------------------------------------------------------------
# MDP: Terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    # Natural episode end.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Ball drifts too far from paddle centre (XY distance > 500 mm).
    # 500 mm gives the robot room to catch the ball during high-energy bounces
    # (Stage F/G: 80-100 cm drops, bounce arc can carry ball 300-400 mm from
    # centre before it descends back).  Paddle radius is 85 mm; termination
    # fires well outside the paddle edge in all cases.
    ball_off = DoneTerm(
        func=mdp.ball_off_paddle,
        params={
            "radius": 0.50,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
        },
    )

    # Ball drops below the paddle surface (ball has fallen off the edge).
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
        self.episode_length_s = 30.0    # 1 500 steps — at the top of current performance plateau
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
        # Use final-stage (Stage G) spawn parameters so play videos show the
        # hardest curriculum conditions the policy was trained on.
        # Without this the play script uses Stage A defaults (drop=0.08 m, σ=0.25)
        # making the ball look like it barely moves.
        self.events.reset_ball.params["drop_height_mean"] = 1.00   # Stage G
        self.events.reset_ball.params["drop_height_std"] = 0.0     # no spread for clean demo
        self.events.reset_ball.params["xy_std"] = 0.0              # centred for demo
        self.rewards.ball_on_paddle.params["std"] = 0.08           # Stage G sigma