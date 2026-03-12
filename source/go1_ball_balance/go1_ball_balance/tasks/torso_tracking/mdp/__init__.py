"""MDP components for the torso-tracking task.

Reuses base MDP terms from isaaclab + shared helpers from ball_balance.
Adds torso-tracking-specific commands, observations, rewards, and events.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Reuse shared helpers from ball_balance
from go1_ball_balance.tasks.ball_balance.mdp.rewards import *       # noqa: F401, F403
from go1_ball_balance.tasks.ball_balance.mdp.terminations import *  # noqa: F401, F403
from go1_ball_balance.tasks.ball_balance.mdp.events import *        # noqa: F401, F403

# Torso-tracking-specific additions
from .commands import *      # noqa: F401, F403  — resample_torso_commands, update_circle_commands
from .observations import *  # noqa: F401, F403  — torso_command_obs
from .rewards import *       # noqa: F401, F403  — 8 tracking reward functions
from .events import *        # noqa: F401, F403  — update_target_marker_pose, update_velocity_arrows_pose
