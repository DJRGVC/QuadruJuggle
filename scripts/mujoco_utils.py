"""Shared MuJoCo utilities for Go1 scripts.

build_reindex(model)
    Builds the REINDEX permutation array at runtime by querying joint names
    from the loaded MuJoCo model, instead of hardcoding [3,4,5, 0,1,2, ...].

    If you change the MJCF file (e.g. reorder actuators, add joints), this
    function adapts automatically — no manual index updates needed.

How REINDEX works
-----------------
Isaac Lab order  : type-grouped — all hips first, then thighs, then calves
  [0]FL_hip [1]FR_hip [2]RL_hip [3]RR_hip
  [4]FL_thigh [5]FR_thigh [6]RL_thigh [7]RR_thigh
  [8]FL_calf  [9]FR_calf  [10]RL_calf [11]RR_calf
MJCF order (XML) : leg-grouped — FR(0-2), FL(3-5), RR(6-8), RL(9-11)

REINDEX[i] = Isaac index of the joint at MJCF actuator position i.
Usage:
    isaac_array = mjcf_array[REINDEX]   # MJCF → Isaac Lab order
"""

import numpy as np
import mujoco

# Joint names in actual Isaac Lab articulation order (type-grouped, confirmed by
# test_joint_cmd_isaaclab.py querying robot.joint_names at runtime).
# ALL hips first, then ALL thighs, then ALL calves — NOT leg-grouped.
ISAAC_JOINT_NAMES = [
    "FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
]


def build_reindex(model: mujoco.MjModel) -> np.ndarray:
    """Build REINDEX permutation array from MuJoCo model metadata.

    Queries each actuator's transmission joint name from the model,
    then maps those names to Isaac Lab joint indices.

    Args:
        model: Loaded MuJoCo model (mujoco.MjModel).

    Returns:
        reindex: int array of shape (12,) where reindex[i] is the Isaac Lab
                 index of the joint at MJCF actuator position i.

    Raises:
        KeyError: If a joint name in the model is not found in ISAAC_JOINT_NAMES.
                  This means ISAAC_JOINT_NAMES needs updating.
    """
    # Get joint name for each actuator in MJCF order
    mjcf_joint_names = [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            model.actuator_trnid[i, 0],   # transmission joint id for actuator i
        )
        for i in range(model.nu)
    ]

    # Map Isaac name → Isaac index
    isaac_name_to_idx = {name: i for i, name in enumerate(ISAAC_JOINT_NAMES)}

    # For each MJCF actuator position, find the corresponding Isaac index
    try:
        reindex = np.array([isaac_name_to_idx[name] for name in mjcf_joint_names])
    except KeyError as e:
        raise KeyError(
            f"Joint name {e} found in MJCF but not in ISAAC_JOINT_NAMES.\n"
            f"  MJCF joints  : {mjcf_joint_names}\n"
            f"  Isaac joints : {ISAAC_JOINT_NAMES}"
        )

    return reindex
