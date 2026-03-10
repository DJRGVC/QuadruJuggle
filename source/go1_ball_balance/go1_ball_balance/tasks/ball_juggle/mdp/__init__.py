"""MDP components for the ball-juggle task.

Reuses all events, observations, terminations, and helper rewards from
ball_balance.mdp.  Adds juggling-specific reward and observation terms.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Reuse ball_balance event / observation / termination / reward helpers
from go1_ball_balance.tasks.ball_balance.mdp.events import *        # noqa: F401, F403
from go1_ball_balance.tasks.ball_balance.mdp.observations import *  # noqa: F401, F403
from go1_ball_balance.tasks.ball_balance.mdp.terminations import *  # noqa: F401, F403
from go1_ball_balance.tasks.ball_balance.mdp.rewards import *       # noqa: F401, F403

# Juggling-specific additions (override nothing from above)
from .rewards import *       # noqa: F401, F403  — ball_apex_height_reward
from .observations import *  # noqa: F401, F403  — target_apex_height_obs
