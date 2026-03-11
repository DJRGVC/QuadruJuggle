"""MDP components for the hierarchical ball-juggle task.

Reuses all MDP terms from the flat ball_juggle task.
The only difference is the action space (6D torso commands via TorsoCommandAction
instead of 12D joint targets).
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Reuse all ball_balance MDP terms
from go1_ball_balance.tasks.ball_balance.mdp.events import *        # noqa: F401, F403
from go1_ball_balance.tasks.ball_balance.mdp.observations import *  # noqa: F401, F403
from go1_ball_balance.tasks.ball_balance.mdp.terminations import *  # noqa: F401, F403
from go1_ball_balance.tasks.ball_balance.mdp.rewards import *       # noqa: F401, F403

# Juggling-specific additions
from go1_ball_balance.tasks.ball_juggle.mdp.rewards import *       # noqa: F401, F403
from go1_ball_balance.tasks.ball_juggle.mdp.observations import *  # noqa: F401, F403
