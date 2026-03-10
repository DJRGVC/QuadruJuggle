"""Custom observation terms for the ball-balance task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_pos_in_paddle_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    paddle_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.18),
) -> torch.Tensor:
    """Ball XYZ position expressed in the paddle frame.

    The paddle frame is rigidly attached to the robot trunk at ``paddle_offset_b``
    (body-frame offset from the trunk origin, in metres).  We return the 3-D
    position of the ball relative to that frame so the policy can see how far off-
    centre the ball is.

    Args:
        env: The RL environment.
        ball_cfg: Scene entity config for the ball rigid object.
        robot_cfg: Scene entity config for the robot articulation.
        paddle_offset_b: XYZ offset of the paddle centre from the trunk origin,
            expressed in the trunk body frame (forward, left, up).

    Returns:
        Tensor of shape (num_envs, 3) — (dx, dy, dz) ball offset from paddle centre.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Trunk root pose: position (N,3) and quaternion wxyz (N,4)
    trunk_pos_w = robot.data.root_pos_w          # (N, 3)
    trunk_quat_w = robot.data.root_quat_w        # (N, 4)  wxyz

    # Paddle centre in world frame = trunk_pos_w + rotate(paddle_offset_b, trunk_quat_w)
    offset_b = torch.tensor(paddle_offset_b, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    paddle_pos_w = trunk_pos_w + math_utils.quat_apply(trunk_quat_w, offset_b)

    # Ball position in world frame
    ball_pos_w = ball.data.root_pos_w            # (N, 3)

    # Vector from paddle centre to ball, rotated into trunk frame
    diff_w = ball_pos_w - paddle_pos_w
    diff_b = math_utils.quat_apply_inverse(trunk_quat_w, diff_w)

    return diff_b


def ball_vel_in_paddle_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball linear velocity expressed in the robot trunk frame.

    Args:
        env: The RL environment.
        ball_cfg: Scene entity config for the ball.
        robot_cfg: Scene entity config for the robot.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_quat_w = robot.data.root_quat_w        # (N, 4) wxyz
    ball_vel_w = ball.data.root_lin_vel_w        # (N, 3)

    return math_utils.quat_apply_inverse(trunk_quat_w, ball_vel_w)
