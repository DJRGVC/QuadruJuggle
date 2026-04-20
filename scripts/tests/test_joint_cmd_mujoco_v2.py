"""Minimal Go1 joint movement test — MuJoCo version.

Press keys in the MuJoCo viewer window.

Keys:
    0-9, A, B   — select joint (0-11)
    ]           — increase selected joint +0.05 rad
    [           — decrease selected joint -0.05 rad
    R           — reset all joints to default
    ESC         — quit

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    python scripts/tests/test_joint_cmd_mujoco_v2.py
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# ---------------------------------------------------------------------------
# Constants (must match play_mujoco.py)
# ---------------------------------------------------------------------------

XML_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../mujoco/go1_juggle.xml"))

SIM_DT = 0.005   # matches XML timestep (0.005s = Isaac Lab physics dt)
KP     = 100.0
KD     = 3.0
EFFORT_LIMIT = 23.7

# MJCF joint order: FR, FL, RR, RL
JOINT_NAMES = [
    "FR_hip",  "FR_thigh", "FR_calf",
    "FL_hip",  "FL_thigh", "FL_calf",
    "RR_hip",  "RR_thigh", "RR_calf",
    "RL_hip",  "RL_thigh", "RL_calf",
]

DEFAULT_JOINT_POS = np.array([
    -0.1,  0.8, -1.5,   # FR
     0.1,  0.8, -1.5,   # FL
    -0.1,  1.0, -1.5,   # RR
     0.1,  1.0, -1.5,   # RL
])

NUM_JOINTS = 12
STEP_SIZE  = 0.05

# ---------------------------------------------------------------------------
# Keyboard state
# ---------------------------------------------------------------------------

joint_offsets  = np.zeros(NUM_JOINTS)
selected_joint = 0

# glfw keycodes
KEY_0   = 48;  KEY_9 = 57   # '0'-'9' are ASCII 48-57
KEY_A   = 65
KEY_B   = 66
KEY_R   = 82
KEY_LSQ = 91   # [
KEY_RSQ = 93   # ]
KEY_ESC = 256

quit_flag = False


def key_callback(keycode):
    global selected_joint, joint_offsets, quit_flag

    if keycode == KEY_ESC:
        quit_flag = True

    elif KEY_0 <= keycode <= KEY_9:
        selected_joint = keycode - KEY_0

    elif keycode == KEY_A:
        selected_joint = 10

    elif keycode == KEY_B:
        selected_joint = 11

    elif keycode == KEY_RSQ:   # ]
        joint_offsets[selected_joint] += STEP_SIZE

    elif keycode == KEY_LSQ:   # [
        joint_offsets[selected_joint] -= STEP_SIZE

    elif keycode == KEY_R:
        joint_offsets[:] = 0.0

    print_state()


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def print_state():
    print("\033[2J\033[H", end="")   # clear terminal
    print("=" * 50)
    print("  Go1 Joint Keyboard Controller — MuJoCo")
    print("  (press keys in the MuJoCo viewer window)")
    print("=" * 50)
    print("  0-9 / A / B  — select joint")
    print("  ]            — +0.05 rad")
    print("  [            — -0.05 rad")
    print("  R            — reset all")
    print("  ESC          — quit")
    print("=" * 50)
    print(f"  {'#':<4} {'Joint':<12} {'Default':>9} {'Offset':>9} {'Target':>9}")
    print(f"  {'-'*48}")
    for i, name in enumerate(JOINT_NAMES):
        default = DEFAULT_JOINT_POS[i]
        offset  = joint_offsets[i]
        target  = default + offset
        marker  = "  ◄" if i == selected_joint else ""
        print(f"  {i:<4} {name:<12} {default:>+.3f}   {offset:>+.3f}   {target:>+.3f}{marker}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

# Set initial pose
mujoco.mj_resetData(model, data)
data.qpos[0:3]  = [0.0, 0.0, 0.35]      # robot base position
data.qpos[3:7]  = [1.0, 0.0, 0.0, 0.0]  # robot base quaternion (upright)
data.qpos[7:19] = DEFAULT_JOINT_POS      # joint positions
# Move ball out of the way (go1_juggle.xml includes a ball)
if model.nq > 19:
    data.qpos[19:22] = [0.0, 0.0, 5.0]  # ball high above, won't interfere
mujoco.mj_forward(model, data)

# Warm up — hold default pose for 200 steps so robot settles before control
print("Settling robot pose...")
targets = DEFAULT_JOINT_POS.copy()
for _ in range(200):
    pos     = data.qpos[7:19]
    vel     = data.qvel[6:18]
    torques = np.clip(KP * (targets - pos) + KD * (-vel), -EFFORT_LIMIT, EFFORT_LIMIT)
    data.ctrl[:] = torques
    mujoco.mj_step(model, data)

print_state()

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running() and not quit_flag:
        t0 = time.perf_counter()

        # PD control toward default + offset
        targets = DEFAULT_JOINT_POS + joint_offsets
        pos     = data.qpos[7:19]
        vel     = data.qvel[6:18]
        torques = np.clip(KP * (targets - pos) + KD * (-vel), -EFFORT_LIMIT, EFFORT_LIMIT)

        data.ctrl[:] = torques
        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, SIM_DT - elapsed))
