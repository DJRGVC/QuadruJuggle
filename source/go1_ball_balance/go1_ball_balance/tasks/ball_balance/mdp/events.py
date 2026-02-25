"""Custom event terms for the ball-balance task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_paddle_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    paddle_cfg: SceneEntityCfg = SceneEntityCfg("paddle"),
    offset_b: tuple[float, float, float] = (0.0, 0.0, 0.13),
) -> None:
    """Reset the kinematic paddle to the default standing position for the given envs.

    Called on episode reset.  We approximate the trunk world position from the
    env origin + standing height (0.40 m) + body-frame offset, with an identity
    quaternion (robot resets upright).  The interval event will correct any
    deviation from the first physics step onward.
    """
    paddle: RigidObject = env.scene[paddle_cfg.name]

    env_origins = env.scene.env_origins[env_ids]          # (N, 3)
    # Robot resets to ~0.40 m above ground with trunk upright
    trunk_pos = env_origins + torch.tensor([0.0, 0.0, 0.40], device=env.device)

    off = torch.tensor(offset_b, device=env.device, dtype=torch.float32)
    paddle_pos = trunk_pos + off.unsqueeze(0)              # offset in world frame (upright)

    quat_identity = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32
    ).unsqueeze(0).expand(len(env_ids), -1)

    pose = torch.cat([paddle_pos, quat_identity], dim=-1)  # (N, 7)
    paddle.write_root_pose_to_sim(pose, env_ids=env_ids)


def update_paddle_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    paddle_cfg: SceneEntityCfg = SceneEntityCfg("paddle"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset_b: tuple[float, float, float] = (0.0, 0.0, 0.13),
) -> None:
    """Continuously track the robot trunk with the kinematic paddle (every policy step).

    Runs as an interval event with interval_range=(1,1) so it fires every step for
    all envs.  We update ALL envs (not just env_ids) so the paddle never lags behind
    a moving trunk.
    """
    paddle: RigidObject = env.scene[paddle_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    trunk_pos_w = robot.data.root_pos_w     # (num_envs, 3)
    trunk_quat_w = robot.data.root_quat_w   # (num_envs, 4) wxyz

    off = torch.tensor(offset_b, device=env.device, dtype=torch.float32)
    off_w = math_utils.quat_rotate(trunk_quat_w, off.unsqueeze(0).expand(env.num_envs, -1))
    paddle_pos_w = trunk_pos_w + off_w

    pose = torch.cat([paddle_pos_w, trunk_quat_w], dim=-1)  # (num_envs, 7)
    paddle.write_root_pose_to_sim(pose)
