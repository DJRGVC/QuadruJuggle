# INBOX

---
[2026-04-09 18:15 UTC] Daniel G → perception
MSG: How close are we to the vision-based model described here: ✅ **fix-keyboard-vision-control-ui** — Built keyboard teleop UI for sim control  **What was missing** - No script for keyboard-based target apex height control (only body velocity existed on perception branch) - No visual UI panel running alongside IsaacSim - The `vel_cmd` module only existed on `agent/perception`, not in the main worktree  **What I built**  **`scripts/rsl_rl/play_teleop_ui.py`** — new teleop script - **UP / DOWN arrows** → raise / lower target apex height ±0.05 m (range 0.05–1.00 m) - **WASD** → body velocity vx/vy (via UserVelocityInput + CommandMixer) - **R** → reset target to 0.30 m | **ESC/Q** → quit - Live-injects `env._target_apex_heights` every step so pi1 immediately reacts  **OpenCV UI window** (640×420, 15 Hz, `DISPLAY=:1`) - Left: top-down ball XY view relative to paddle circle - Right: scrolling height timeline — ball Z vs target line (last 5 s) - Middle: target apex height bar (colour-coded cyan→orange→red) - Bottom: telemetry (ball height, peak this episode, robot vel, ball vz) + key hints  **`source/go1_ball_balance/go1_ball_balance/vel_cmd/`** — ported from perception branch - `CommandMixer`, `UserVelocityInput` — needed by play_teleop_ui.py  **How to run** (DISPLAY=:1 required for OpenCV window) ``` DISPLAY=:1 uv run --active python scripts/rsl_rl/play_teleop_ui.py \     --task Isaac-BallJuggleHier-Go1-Play-v0 \     --pi2-checkpoint <path/to/pi2_checkpoint.pt> \     --num_envs 1 --real-time ```  **Result**: All unit tests pass (vel_cmd, TeleopState, _draw_ui rendering). UI preview image committed to `images/fix-keyboard-vision-control-ui/teleop_ui_preview_iter_001.png`.  **One remaining step**: full "vision-based" camera feed (EKF + D435i) requires the perception branch to be merged. Once merged, add `--noise-mode d435i` to route ball observations through the EKF instead of ground truth.

---
[2026-04-09 18:15 UTC] Daniel G → perception
MSG: And, when it’s ready, can you make sure to update the ui/opencv thing to show the camera feed? And not just a random top-down view?

---
[2026-04-09 18:16 UTC] Daniel G → perception
MSG: Or maybe add it in addition. Your choice.
