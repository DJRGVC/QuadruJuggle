"""Custom reward terms for the ball-balance task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_on_paddle_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    ball_radius: float = 0.020,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.18),
    min_height: float = 0.28,
    nominal_height: float = 0.40,
    time_scale: float = 0.0,
) -> torch.Tensor:
    """Exponential reward for keeping the ball centred on the paddle,
    gated by trunk height so a collapsed robot earns no ball reward.

    Measures 3D distance from the ideal resting position
    (paddle_centre + ball_radius * paddle_normal_world) rather than world XY.
    This prevents the robot from exploiting the reward by tilting its trunk
    to align the paddle normal with the ball — a tilted robot places the ball
    far from the paddle surface, giving high 3D distance and low reward.

    Args:
        env: The RL environment.
        std: Gaussian half-width in metres.
        ball_radius: Physical radius of the ball (metres).
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.
        paddle_offset_b: Paddle-centre offset from trunk origin (body frame).
        min_height: Trunk Z below which the gate is 0 (metres).
        nominal_height: Trunk Z at which the gate reaches 1 (metres).
        time_scale: If > 0, multiply the reward by (1 + time_scale * progress)
            where progress = current_step / max_episode_steps.  This makes the
            ball reward grow linearly over the episode, e.g. time_scale=4 gives
            1x at step 0 and 5x at the final step, strongly rewarding sustained
            balance over short bursts.

    Returns:
        Tensor of shape (num_envs,).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

    # Paddle normal in world frame (trunk "up" direction)
    up_b = torch.zeros(env.num_envs, 3, device=env.device)
    up_b[:, 2] = 1.0
    paddle_normal_w = math_utils.quat_apply(trunk_quat_w, up_b)  # (N, 3)

    # Ideal ball position: resting on paddle surface at centre
    target_pos_w = paddle_pos_w + ball_radius * paddle_normal_w   # (N, 3)

    # 3D distance from ideal resting position
    dist = torch.norm(ball.data.root_pos_w - target_pos_w, dim=-1)

    # Height gate: 0 when trunk is at/below min_height, 1 at nominal standing height
    trunk_z = trunk_pos_w[:, 2]
    height_gate = torch.clamp(
        (trunk_z - min_height) / (nominal_height - min_height), 0.0, 1.0
    )

    base_reward = height_gate * torch.exp(-dist.pow(2) / (2.0 * std**2))

    if time_scale > 0.0:
        progress = env.episode_length_buf / env.max_episode_length  # (N,) in [0, 1]
        base_reward = base_reward * (1.0 + time_scale * progress)

    return base_reward


def body_lin_vel_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise large linear velocity of the robot base (encourages standing still).

    Returns:
        Tensor of shape (num_envs,) — negative values suitable for a negative weight.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    lin_vel = robot.data.root_lin_vel_b          # body-frame linear velocity (N, 3)
    return torch.norm(lin_vel, dim=-1)


def body_ang_vel_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise large angular velocity of the robot base.

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ang_vel = robot.data.root_ang_vel_b          # body-frame angular velocity (N, 3)
    return torch.norm(ang_vel, dim=-1)


def trunk_tilt_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise trunk roll and pitch away from level.

    Uses the projected gravity vector: when the robot is perfectly upright,
    the gravity vector in body frame is (0, 0, -1), so the XY components are
    both 0.  Any tilt increases the XY magnitude, which we penalise.

    Returns:
        Tensor of shape (num_envs,) — positive values, apply with negative weight.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # gravity projected into body frame — built-in data field
    gravity_b = robot.data.projected_gravity_b          # (N, 3)
    # XY magnitude = sin(tilt_angle) — 0 when level, 1 when fully sideways
    return torch.norm(gravity_b[:, :2], dim=-1)


def ball_lateral_vel_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.28,
    nominal_height: float = 0.40,
) -> torch.Tensor:
    """Penalise lateral (XY) speed of the ball, gated by trunk height.

    Fires as soon as the ball begins sliding, even while it is still within
    the centering reward radius.  This gives the policy a gradient to actively
    damp ball motion rather than passively watching it roll off.

    Returns:
        Tensor of shape (num_envs,) — positive values, apply with negative weight.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ball_speed_xy = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)

    trunk_z = robot.data.root_pos_w[:, 2]
    height_gate = torch.clamp(
        (trunk_z - min_height) / (nominal_height - min_height), 0.0, 1.0
    )

    return height_gate * ball_speed_xy


def ball_height_penalty(
    env: ManagerBasedRLEnv,
    ball_radius: float = 0.020,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.18),
    min_height: float = 0.28,
    nominal_height: float = 0.40,
) -> torch.Tensor:
    """Penalise the ball being above the paddle surface, gated by trunk height.

    Returns 0 when the ball rests on the paddle surface.  Once airborne the
    penalty grows linearly with bounce height.  The gate suppresses this term
    when the robot is collapsed, so it does not compete with base_height_penalty.

    Args:
        env: The RL environment.
        ball_radius: Physical radius of the ball (metres).
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.
        paddle_offset_b: Paddle-centre offset from trunk origin (body frame).
        min_height: Trunk Z below which the gate is 0 (metres).
        nominal_height: Trunk Z at which the gate reaches 1 (metres).

    Returns:
        Tensor of shape (num_envs,) — positive values, apply with negative weight.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

    ball_z = ball.data.root_pos_w[:, 2]
    paddle_z = paddle_pos_w[:, 2]

    airborne_height = torch.clamp(ball_z - paddle_z - ball_radius, min=0.0)

    trunk_z = trunk_pos_w[:, 2]
    height_gate = torch.clamp(
        (trunk_z - min_height) / (nominal_height - min_height), 0.0, 1.0
    )

    return height_gate * airborne_height


def ball_xy_dist_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.18),
    min_height: float = 0.33,
    nominal_height: float = 0.40,
) -> torch.Tensor:
    """Linear penalty on XY distance of ball from paddle centre, height-gated.

    Complements ball_on_paddle_exp: the Gaussian has near-zero gradient when the
    ball is far from centre (flat tails), so the robot gets no pull signal to chase
    a ball that has drifted to the paddle edge.  This linear term provides a constant
    gradient at ALL distances, ensuring the robot always wants to bring the ball in.

    Returns:
        Tensor of shape (num_envs,) — positive values, apply with negative weight.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

    dist_xy = torch.norm(ball.data.root_pos_w[:, :2] - paddle_pos_w[:, :2], dim=-1)

    trunk_z = trunk_pos_w[:, 2]
    height_gate = torch.clamp(
        (trunk_z - min_height) / (nominal_height - min_height), 0.0, 1.0
    )

    return height_gate * dist_xy


def joint_pos_default_tracking(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for joint positions being close to the default standing config.

    Returns 1.0 when all joints are at their default angles, decaying smoothly as
    joints deviate.  Provides a gradient toward the standing pose at ALL trunk heights,
    including when the robot is collapsed and the height-gated alive/base_height terms
    give no directional signal.

    Args:
        std: RMS joint deviation (radians) at which reward drops to exp(-0.5) ≈ 0.61.
             With 12 joints a per-joint std of std/sqrt(12) ≈ 0.14 rad is typical.

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    deviation = robot.data.joint_pos - robot.data.default_joint_pos  # (N, 12)
    sq_sum = torch.sum(deviation.pow(2), dim=-1)                      # (N,)
    return torch.exp(-sq_sum / (2.0 * std ** 2))


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant +1 survival bonus every step — makes longer episodes strictly better
    than dying early, countering the 'die-fast' local minimum.

    Returns:
        Tensor of shape (num_envs,) — all ones.
    """
    return torch.ones(env.num_envs, device=env.device)


def alive_upright(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.28,
    nominal_height: float = 0.40,
    ball_cfg: SceneEntityCfg | None = None,
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.045),
    ball_inner_radius: float = 0.15,
    ball_max_radius: float = 0.40,
) -> torch.Tensor:
    """Survival bonus gated by trunk height AND (optionally) ball proximity.

    Height gate: 0 when trunk is at/below min_height, 1 at nominal standing height.
    Ball gate:   1 when ball XY is within ball_inner_radius of the paddle centre,
                 linearly decreasing to 0 at ball_max_radius.  This eliminates the
                 'move sideways out of the way' exploit — the robot earns no alive
                 reward once the ball is far from its paddle.

    Returns values in [0, 1]; apply with a large positive weight.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    trunk_z = robot.data.root_pos_w[:, 2]
    height_gate = torch.clamp((trunk_z - min_height) / (nominal_height - min_height), 0.0, 1.0)

    if ball_cfg is None:
        return height_gate

    ball: RigidObject = env.scene[ball_cfg.name]
    trunk_pos_w = robot.data.root_pos_w
    trunk_quat_w = robot.data.root_quat_w

    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

    dist_xy = torch.norm(ball.data.root_pos_w[:, :2] - paddle_pos_w[:, :2], dim=-1)
    span = max(ball_max_radius - ball_inner_radius, 1e-6)
    ball_gate = torch.clamp(1.0 - (dist_xy - ball_inner_radius) / span, 0.0, 1.0)

    return height_gate * ball_gate


def early_termination_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-shot penalty when the episode ends due to a non-timeout termination
    (i.e. ball_off_paddle or ball_below_paddle).

    Terminations are computed before rewards in Isaac Lab's step(), so
    env.termination_manager.dones is valid here.

    Returns:
        Tensor of shape (num_envs,) — 1.0 on bad-termination steps, else 0.0.
        Apply with a large negative weight (e.g. -50.0).
    """
    bad_done = env.termination_manager.dones & ~env.termination_manager.time_outs
    return bad_done.float()


def base_height_penalty(
    env: ManagerBasedRLEnv,
    min_height: float = 0.28,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise the robot for letting its trunk drop below a minimum height.

    Returns 0 when the trunk is at or above ``min_height`` (normal standing is
    ~0.38 m).  Once the trunk drops below that floor the penalty grows linearly
    with the deficit, making it costly to collapse or squat all the way down.

    Args:
        env: The RL environment.
        min_height: Z threshold in world frame (metres).  Trunk below this
            value incurs a penalty proportional to the shortfall.
        robot_cfg: Scene entity config for the robot.

    Returns:
        Tensor of shape (num_envs,) — positive values, apply with negative weight.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    trunk_z = robot.data.root_pos_w[:, 2]            # world-frame trunk height
    return torch.clamp(min_height - trunk_z, min=0.0)


def base_height_max_penalty(
    env: ManagerBasedRLEnv,
    max_height: float = 0.45,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise the robot for raising its trunk above a maximum height ceiling.

    Prevents the policy from gaming height-gated rewards by fully extending all
    four legs (reaching 0.47-0.50 m) rather than maintaining a natural standing
    pose (~0.40 m).  The ceiling is set above the nominal height to allow normal
    postural adjustments and active juggling thrust dynamics without penalty.

    Args:
        env: The RL environment.
        max_height: Z ceiling in world frame (metres).  Juggling task: 0.45 m
            (5 cm above nominal, allows active trunk thrust without penalty).
            Balance task: 0.43 m (3 cm above nominal; no active thrust needed).
        robot_cfg: Scene entity config for the robot.

    Returns:
        Tensor of shape (num_envs,) — positive values, apply with negative weight.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    trunk_z = robot.data.root_pos_w[:, 2]
    return torch.clamp(trunk_z - max_height, min=0.0)


def feet_off_ground_penalty(
    env: ManagerBasedRLEnv,
    foot_contact_cfg: SceneEntityCfg = SceneEntityCfg("foot_contact_forces"),
    min_force: float = 1.0,
) -> torch.Tensor:
    """Penalise feet that are not in contact with the ground.

    Returns the count of feet (0–4) whose net contact force magnitude is below
    ``min_force`` Newtons.  A value of 0 means all four feet are planted; a
    value of 1 is the "kickstand" exploit (one leg splayed sideways for lateral
    stability while the other three carry the robot).

    Academic basis: Ji et al. DribbleBot (ICRA 2023) requires stable
    foot-ground contact during all manipulation phases.  A quadruped cannot
    generate a controlled ground-reaction force for juggling thrust without
    planted feet; the kickstand leg both wastes a DOF and destabilises the
    platform under ball impact.

    Requires a ContactSensorCfg in the scene named ``foot_contact_forces``
    matching the four foot prim paths (e.g. ``.*_foot``).

    Args:
        env: The RL environment.
        foot_contact_cfg: Scene entity config pointing to the foot contact sensor.
        min_force: Contact force threshold in Newtons below which a foot is
            considered airborne.  1.0 N is well below normal stance load
            (~30 N per foot for a 12 kg robot) but above sensor noise.

    Returns:
        Tensor of shape (num_envs,) — float count in [0, 4], apply with
        negative weight.
    """
    sensor: ContactSensor = env.scene[foot_contact_cfg.name]
    # net_forces_w: (num_envs, num_feet, 3) — contact force in world frame
    force_mags = torch.norm(sensor.data.net_forces_w, dim=-1)  # (N, 4)
    airborne   = (force_mags < min_force).float()               # (N, 4)
    return airborne.sum(dim=-1)                                  # (N,)