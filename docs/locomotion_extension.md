# Locomotion Extension: Juggling While Walking

Notes on extending the hierarchical architecture to support velocity/heading commands alongside ball juggling.

## Current Architecture

pi1 (6D) --> pi2 (12D joint targets)
- pi2 tracks: h, h_dot, roll, pitch, omega_roll, omega_pitch
- pi2 obs: 39D (6 command + 3 lin_vel + 3 ang_vel + 3 gravity + 12 joint_pos + 12 joint_vel)

## Proposed Extension: 6D --> 9D Interface

| Dim | Name | Range | Units |
|-----|------|-------|-------|
| 0 | h_target | [0.20, 0.50] | m |
| 1 | h_dot_target | [-1.5, 1.5] | m/s |
| 2 | roll_target | [-0.5, 0.5] | rad |
| 3 | pitch_target | [-0.5, 0.5] | rad |
| 4 | omega_roll | [-4.0, 4.0] | rad/s |
| 5 | omega_pitch | [-4.0, 4.0] | rad/s |
| **6** | **vx** | **[-0.5, 0.5]** | **m/s** |
| **7** | **vy** | **[-0.3, 0.3]** | **m/s** |
| **8** | **omega_yaw** | **[-0.5, 0.5]** | **rad/s** |

pi2 obs grows from 39D to 42D.

## Feasibility

- **Go1 joints**: 30 rad/s velocity limit per joint, 23.7 Nm effort limit. Not saturated by either task alone.
- **Robot is free to move**: ball_juggle_hier scene has no position constraints on the base.
- **Isaac Lab has existing velocity tracking infra** that can be reused directly.

## Reusable Isaac Lab Components

| Component | Location |
|-----------|----------|
| `UniformVelocityCommandCfg` | `isaaclab/envs/mdp/commands/commands_cfg.py` |
| `track_lin_vel_xy_yaw_frame_exp` | `isaaclab_tasks/.../locomotion/velocity/mdp/rewards.py` |
| `track_ang_vel_z_world_exp` | same |
| Go1 velocity task reference | `isaaclab_tasks/.../locomotion/velocity/config/go1/` |

## Files to Modify

1. `torso_tracking/mdp/commands.py` -- add vx, vy, omega_yaw to command buffer + ranges
2. `torso_tracking/mdp/observations.py` -- add 3 dims to _NORM, _OFFSET
3. `torso_tracking/action_term.py` -- expand _CMD_SCALES/_CMD_OFFSETS to 9D
4. `torso_tracking/torso_tracking_env_cfg.py` -- add velocity tracking rewards
5. `torso_tracking/mdp/rewards.py` -- add velocity tracking reward functions (or import from Isaac Lab)
6. `train_torso_tracking.py` -- expand curriculum to include velocity ranges
7. `ball_juggle_hier/ball_juggle_hier_env_cfg.py` -- expand action space to 9D, add velocity obs/rewards
8. `ball_juggle_hier/agents/rsl_rl_ppo_cfg.py` -- update input dims

## Potential Gotchas

- Curriculum complexity: 3 objectives (pose tracking + velocity tracking + juggling) needs careful staging. Probably learn pose first, then add velocity, then juggle.
- Velocity ranges should be conservative while juggling (Go1 max ~0.8 m/s, but juggling likely limits to ~0.3-0.5 m/s).
- Foot contact penalty may need retuning -- walking naturally lifts feet.
- May need to increase env_spacing to prevent collisions between walking robots.
