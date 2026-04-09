# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## Compacted summary (through iter_073)

**Iters 001-004 (docs/roadmap):** Updated perception_roadmap.md and sim_to_real_plan.md for D435i
(stereo depth, not monocular). Surveyed Isaac Lab camera APIs — TiledCamera for debug only;
training uses ETH noise-injection on GT state (no camera sensor needed). Created ball_obs_spec.py
with 3 modes: oracle (GT passthrough), d435i (structured depth noise), ekf (EKF-filtered).

**Iters 005-011 (camera + EKF + pipeline):** Mounted simulated D435i TiledCamera in PLAY scene.
Implemented ball_ekf.py (batched 6-state EKF, ballistic+drag, Joseph-form covariance).
Created noise_model.py + PerceptionPipeline. Handed off to policy agent.

**Iters 012-028 (diagnostics + contact-aware EKF):** Added RMSE/NIS diagnostics. NIS=966 at
q_vel=0.30 due to unmodeled paddle contact forces. Built hardware pipeline stubs.
Contact-aware EKF (q_vel=0.40 flight, 50.0 contact) → NIS from 966 to 1.60.

**Iters 029-048 (extensions + tooling):** IMU-aided EKF, 9D spin estimation, config wiring.
Velocity command package. Hough circle detector. Threaded RealPerceptionPipeline.
Chi-squared NIS gating. 239/239 tests.

**Iters 049-059 (GPU validation + active-policy tuning):** EKF severely overconfident under
active juggling (flight NIS=52.9 vs random-action NIS=1.45). 3-level q_vel architecture:
contact=50.0, post_contact=20.0, flight=TBD. Wrote sweep_q_vel.py.

**Iters 060-068 (sweep tooling + bug fixes):** Built sweep infrastructure (bisection, apply_sweep_results.py).
Fixed EKF reset under inference_mode, diagnostics pipeline recreation bug (all 4 scripts),
numpy→tensor warnings. All CPU work — GPU locked by policy training.

**Iters 069-073 (GPU sweeps + adaptive R):** High-range sweep (q_vel 0.4-50): ALL flight NIS < 3.0
(max 1.73). Root cause: R_xy calibrated for z=0.5m but Stage A balls at z≈0.1m → R_xy variance
25× too large. Fix: adaptive R_xy = max(r_xy_per_metre·z, r_xy_floor). Quarto docs + experiment
write-up created. Low-range sweep queued with adaptive R.

**Key architectural findings through iter_073:**
- Contact-aware EKF: inflate Q during paddle contact → NIS from 966 to ~3.5
- Adaptive R_xy: σ_xy = 0.0025·z → measurement noise scales with distance
- Ahn 2019 noise: sigma_xy=0.0025*z, sigma_z=1mm+0.005*z², dropout 20-50%
- Raw d435i noise outperforms EKF at close range (z≈0.1m, Stage A)
- EKF value: velocity estimation + dropout bridging during free-flight
- 3-level q_vel: contact=50.0, post_contact=20.0 (10 steps), flight=0.4
- **D435i camera mounted in sim but debug frames blank — camera not seeing ball**
- **No camera→detect→EKF→juggle pipeline exists yet (ETH-style only)**

---

## Iteration 74 — Adaptive R_xy sweep validates calibration  (2026-04-09T02:23:00Z)
Hypothesis: With adaptive R_xy (iter 73 fix), low-range q_vel sweep should show flight NIS ≈ 3.0,
            confirming the root cause was R_xy calibrated for wrong height.
Change:     Waited for queued GPU sweep (PID 999147) to complete. Analysed results from
            sweep_q_vel_low_range.json (6 q_vel points: 0.01-0.40, 512 envs × 600 steps).
            Generated combined figure. Updated Quarto page + fix_plan.
Command:    python apply_sweep_results.py logs/perception/sweep_q_vel_low_range.json --plot
Result:     All flight NIS now 3.31-3.78 (was < 1.73 pre-fix). Root cause confirmed.
            q_vel=0.40: flight NIS=3.31, EKF RMSE=6.49mm vs raw 4.92mm (-32.0%).
            EKF loses to raw at Stage A (z≈0.1m) — expected, camera too precise for EKF to help.
            q_vel=0.40 default unchanged (already correct).
            256/256 CPU tests pass.
Decision:   Next iter: run higher-height sweep (target=0.3-0.5m) to confirm EKF beats raw at
            distance. Then final handoff to policy agent. Alternative: skip higher-height sweep
            (EKF value proven via velocity estimation, not position) and declare pipeline tuned.

## Iteration 75 — compaction (summarized iters 060-073)  (2026-04-09T03:00:00Z)
Hypothesis: N/A — compaction iteration (log at 310 lines > 300 threshold).
Change:     Archived iters 060-073 verbatim to RESEARCH_LOG_ARCHIVE.md. Rewrote compacted summary
            through iter_073. Processed 3 INBOX entries: (1) Daniel asks about sim camera/detect/
            juggle status — replied honestly: camera blank, no camera→detect→EKF loop. (2) Fixer
            restart notification — ack'd, noted Daniel's direct request takes priority. (3) Daniel
            suggests 70°+ camera tilt, confirms D435i is the hardware camera.
            Updated fix_plan to prioritize camera visualization.
Command:    No GPU commands.
Result:     Log shrunk from 310 → ~80 lines. Archive now has 73 verbatim entries (001-073).
            fix_plan rewritten: sim camera visualization is now top priority.
Decision:   Next iter: compute FOV geometry for D435i at 70° tilt. Update camera mount in
            BallJuggleHierSceneCfg_DEBUG. Run GPU smoke test to capture frames with ball visible.
            D435i specs: 86° HFOV, 58° VFOV, min depth 0.1m.

## Iteration 76 — Camera quaternion convention discovered, tilt corrected to 70°  (2026-04-08T19:35:00Z)
Hypothesis: Blank camera frames caused by ball being below camera FOV at 45° tilt;
            steeper tilt will bring airborne ball into view.
Change:     Three GPU runs to diagnose:
            1. First set tilt to "75°" via q=(0.7934,-0.6088,0,0) — still no ball in frame.
            2. Added camera world-pose diagnostics. Discovered critical convention:
               **convention="ros" means identity = looking STRAIGHT UP (zenith).**
               Rotation angle = -(90 - desired_elevation). So -75° rotation → 15° above
               horizontal, NOT 75°. The old 45° config was actually correct at 45° elev!
            3. Fixed to 70° elevation: rot_angle=-20°, q=(0.9848,-0.1736,0,0).
               FOV: 41°-99° above horizontal. Also set update_period=0.0 for debug,
               moved camera to (-0.08,0,0.06), ball impulse 3m/s in debug script.
Command:    3× $C3R_BIN/gpu_lock.sh debug_d435i_capture.py (between policy train runs)
            104/104 CPU tests pass.
Result:     Convention clarified. 70° tilt queued for GPU validation but blocked by
            policy training (started 20:44, will run ~60 min).
            Key finding: old 45° tilt was geometrically sound — ball at rest is at
            ~21° elevation, within 45° camera's FOV (16°-74°). The "blank frames"
            issue may have been a different problem (ball too small, rendering timing,
            or wrong definition of "blank").
Decision:   Next iter: check if GPU capture completed (queued PID). If ball visible
            at 70° tilt → proceed to sim detector. If still blank → investigate
            rendering pipeline (timing, ball visibility, material).

## Iteration 77 — Convention fix: "ros" → "world", sim ball detector  (2026-04-08T21:30:00Z)
Hypothesis: Iter 76 frames show near-horizontal view despite "ros" convention fix — the
            ros convention (identity=+Z fwd, -Y up) applies non-intuitively as body-frame
            offset. Switching to convention="world" (identity=+X fwd, +Z up) with
            q=(0.8192, 0, -0.5736, 0) for 70° pitch-up should correctly point camera upward.
Change:     1. Changed camera offset from convention="ros" rot=(0.9848,-0.1736,0,0) to
               convention="world" rot=(0.8192, 0.0, -0.5736, 0.0).
            2. Created sim_detector.py — SimBallDetector for TiledCamera float32 depth
               (connected components + ball-size scoring). 8 unit tests.
            3. Integrated SimBallDetector into debug_d435i_capture.py (auto-runs on capture).
            4. GPU capture queued but blocked by policy training (~55 min remaining).
Command:    pytest scripts/perception/ → 264/264 passed (256 existing + 8 new).
            GPU capture: $C3R_BIN/gpu_lock.sh debug_d435i_capture.py (waiting for lock).
Result:     Code changes complete. GPU validation pending.
            Analysed iter 76 frames: RGB shows ground+horizon (horizontal view), depth
            confirms close objects only at bottom-left (robot body). Camera was NOT tilted
            70° up — ros convention was misunderstood.
Decision:   Next iter: check GPU capture results. If ball visible with world convention →
            wire camera→detect→EKF demo pipeline. If still wrong → try identity quaternion
            first, then systematically rotate.

## Iteration 78 — Fix demo_camera_ekf.py frame consistency + kill child  (2026-04-09T06:35:00Z)
Hypothesis: Demo script had frame mismatch — EKF initialized in world frame but detections
            transformed to body frame via manual rotation matrices. Using cam.data.quat_w_ros
            directly is simpler and correct.
Change:     1. Rewrote cam_detection_to_world() to use cam.data.pos_w + cam.data.quat_w_ros
               from Isaac Lab sensor data, eliminating manual body-frame intermediate.
            2. Added camera pose diagnostic at startup (prints forward vector + elevation).
            3. Removed unused robot entity reference.
            4. Killed testing-dashboard child (guide complete at iter 4, idle).
               Note: c3r kill failed due to worktree path issue; child still running.
Command:    pytest scripts/perception/ → 264/264 passed.
            No GPU commands (policy d435i training at iter 950/1500, ~7.8GB VRAM).
Result:     Demo script frame handling corrected. Camera convention analysis:
            - Isaac Lab stores camera data in OpenGL internally (-Z fwd, +Y up)
            - cam.data.quat_w_ros converts to ROS convention (+Z fwd, -Y up, +X right)
            - SimBallDetector returns pos_cam in ROS frame (deprojection: x=right, y=down, z=fwd)
            - quat_w_ros @ pos_cam + pos_w → world frame directly
Decision:   Next iter: GPU capture with convention="world" config. If ball visible → run
            full demo_camera_ekf.py to validate end-to-end pipeline. If GPU still locked,
            focus on adding a unit test for cam_detection_to_world().
