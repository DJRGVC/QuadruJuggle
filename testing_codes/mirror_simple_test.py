import numpy as np

def mirror_law_torso(
    ball_pos,      # (3,)
    ball_vel,      # (3,)
    paddle_pos,    # (3,)
    robot_R,       # (3,3) world -> body rotation
    K=2.0,
    h=0.3,
    e=0.8,
    h_nominal=0.38
):
    """
    Returns:
        torso_cmd: (6,) -> [h, h_dot, roll, pitch, 0, 0]
    """

    # 1. relative position
    p_rel = ball_pos - paddle_pos

    # 2. desired outgoing velocity
    v_out = np.array([
        -K * p_rel[0],
        -K * p_rel[1],
        np.sqrt(2 * 9.81 * h)
    ])

    # 3. restitution correction
    v_out_eff = v_out / e

    # 4. mirror law normal in world frame
    n_raw = v_out_eff - ball_vel
    n = n_raw / (np.linalg.norm(n_raw) + 1e-8)

    # 5. make sure normal points upward
    if n[2] < 0:
        n = -n

    # 6. rotate into body frame
    n_b = robot_R @ n
    nx, ny, nz = n_b

    # 7. compute pitch and roll
    nz_safe = max(nz, 0.15)
    pitch = np.arctan2(nx, nz_safe)
    roll = np.arctan2(-ny, nz_safe)

    # 8. clip
    pitch = np.clip(pitch, -0.4, 0.4)
    roll = np.clip(roll, -0.4, 0.4)

    # 9. simple h_dot
    if ball_vel[2] < 0:
        h_dot = 0.2
    else:
        h_dot = 0.1

    # 10. final torso command
    torso_cmd = np.array([h_nominal, h_dot, roll, pitch, 0.0, 0.0])

    return torso_cmd

import numpy as np

def mirror_law_torso(
    ball_pos,
    ball_vel,
    paddle_pos,
    robot_R,
    K=2.0,
    h=0.3,
    e=0.8,
    h_nominal=0.38
):
    p_rel = ball_pos - paddle_pos

    v_out = np.array([
        -K * p_rel[0],
        -K * p_rel[1],
        np.sqrt(2 * 9.81 * h)
    ])

    v_out_eff = v_out / e
    n_raw = v_out_eff - ball_vel
    n = n_raw / (np.linalg.norm(n_raw) + 1e-8)

    if n[2] < 0:
        n = -n

    n_b = robot_R @ n
    nx, ny, nz = n_b

    nz_safe = max(nz, 0.15)
    pitch = np.arctan2(nx, nz_safe)
    roll = np.arctan2(-ny, nz_safe)

    pitch = np.clip(pitch, -0.4, 0.4)
    roll = np.clip(roll, -0.4, 0.4)

    if ball_vel[2] < 0:
        h_dot = 0.2
    else:
        h_dot = 0.1

    torso_cmd = np.array([h_nominal, h_dot, roll, pitch, 0.0, 0.0])
    return torso_cmd


def main():
    ball_pos = np.array([0.1, 0.0, 0.5])
    ball_vel = np.array([0.0, 0.0, -2.0])
    paddle_pos = np.array([0.0, 0.0, 0.0])
    robot_R = np.eye(3)

    torso_cmd = mirror_law_torso(ball_pos, ball_vel, paddle_pos, robot_R)
    print("torso_cmd =", torso_cmd)


if __name__ == "__main__":
    main()