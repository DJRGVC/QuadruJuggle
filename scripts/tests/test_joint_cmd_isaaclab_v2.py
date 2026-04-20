"""Minimal Go1 joint movement test with keyboard control.

Press keys in the Isaac Sim WINDOW (not terminal).

Keys:
    0-9, A, B   — select joint (0-11)
    ]           — increase selected joint +0.05 rad
    [           — decrease selected joint -0.05 rad
    R           — reset all joints to zero

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    python scripts/tests/test_joint_cmd_isaaclab_v2.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal Go1 joint keyboard control.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import carb
import omni

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class MySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ---------------------------------------------------------------------------
# Minimal MDP
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    joint_pos = JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True,
    )

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene:        MySceneCfg      = MySceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions:      ActionsCfg      = ActionsCfg()
    rewards:      RewardsCfg      = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 9999.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation


# ---------------------------------------------------------------------------
# Register + create env
# ---------------------------------------------------------------------------

gym.register(
    id="My-Go1-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": MyEnvCfg},
)

env_cfg = MyEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env = gym.make("My-Go1-Test-v0", cfg=env_cfg)
device = env.unwrapped.device

JOINT_NAMES = [
    "FL_hip",   "FL_thigh", "FL_calf",
    "FR_hip",   "FR_thigh", "FR_calf",
    "RL_hip",   "RL_thigh", "RL_calf",
    "RR_hip",   "RR_thigh", "RR_calf",
]
NUM_JOINTS = 12
STEP_SIZE  = 0.05

# ---------------------------------------------------------------------------
# Keyboard handler using Omniverse carb.input (keys in Isaac Sim window)
# ---------------------------------------------------------------------------

joint_cmds     = [0.0] * NUM_JOINTS
selected_joint = 0

# Map key names → joint indices
_KEY_JOINT_MAP = {
    carb.input.KeyboardInput.KEY_0: 0,
    carb.input.KeyboardInput.KEY_1: 1,
    carb.input.KeyboardInput.KEY_2: 2,
    carb.input.KeyboardInput.KEY_3: 3,
    carb.input.KeyboardInput.KEY_4: 4,
    carb.input.KeyboardInput.KEY_5: 5,
    carb.input.KeyboardInput.KEY_6: 6,
    carb.input.KeyboardInput.KEY_7: 7,
    carb.input.KeyboardInput.KEY_8: 8,
    carb.input.KeyboardInput.KEY_9: 9,
    carb.input.KeyboardInput.A:     10,
    carb.input.KeyboardInput.B:     11,
}

def _on_key(event, *args):
    global selected_joint, joint_cmds
    if event.type != carb.input.KeyboardEventType.KEY_PRESS:
        return True

    k = event.input

    if k in _KEY_JOINT_MAP:
        selected_joint = _KEY_JOINT_MAP[k]

    elif k == carb.input.KeyboardInput.RIGHT_BRACKET:   # ]
        joint_cmds[selected_joint] += STEP_SIZE

    elif k == carb.input.KeyboardInput.LEFT_BRACKET:    # [
        joint_cmds[selected_joint] -= STEP_SIZE

    elif k == carb.input.KeyboardInput.R:
        joint_cmds = [0.0] * NUM_JOINTS

    _print_state()
    return True


def _print_state():
    print("\033[2J\033[H", end="")   # clear terminal
    print("=" * 50)
    print("  Go1 Joint Keyboard Controller")
    print("  (press keys in the Isaac Sim window)")
    print("=" * 50)
    print("  0-9 / A / B  — select joint")
    print("  ]            — +0.05 rad")
    print("  [            — -0.05 rad")
    print("  R            — reset all")
    print("=" * 50)
    print(f"  {'#':<4} {'Joint':<14} {'Cmd (rad)':>10}")
    print(f"  {'-'*32}")
    for i, (name, val) in enumerate(zip(JOINT_NAMES, joint_cmds)):
        marker = "  ◄ selected" if i == selected_joint else ""
        print(f"  {i:<4} {name:<14} {val:>+.3f}{marker}")
    print("=" * 50)


# Subscribe to keyboard events from the Omniverse window
_appwindow = omni.appwindow.get_default_app_window()
_input     = carb.input.acquire_input_interface()
_keyboard  = _appwindow.get_keyboard()
_sub       = _input.subscribe_to_keyboard_events(_keyboard, _on_key)

env.reset()
_print_state()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

try:
    while simulation_app.is_running():
        actions = torch.tensor([joint_cmds], dtype=torch.float32, device=device)
        env.step(actions)

except KeyboardInterrupt:
    print("\nStopped.")

env.close()
simulation_app.close()
