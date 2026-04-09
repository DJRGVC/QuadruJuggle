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

## Iteration 79 — Quarto: references + experiment write-up + policy ping  (2026-04-09T06:00:00Z)
Hypothesis: Daniel requested references and experiments documentation by tomorrow morning.
            Populating references/perception.qmd and writing adaptive R_xy experiment.
Change:     1. Added 9 references to references/perception.qmd (Rudin, Portela, Zhuang, Hoeller,
               Huang, Margolis, Caluwaerts, Bar-Shalom — total now 15 entries).
            2. Wrote experiments/perception/2026-04-09_adaptive_rxy_sweep.qmd — full write-up
               of the adaptive R_xy fix with before/after comparison tables.
            3. Updated agents/perception.qmd with iters 77-78 camera convention + SimBallDetector.
            4. Pinged policy agent about references + experiments requirement from Daniel.
            5. Processed 3 INBOX entries (testing-dashboard status check, Daniel deadline note,
               Daniel references/experiments request).
Command:    No GPU commands (policy d435i training at ~iter 950/1500).
Result:     Quarto content populated: 15 references, 2 experiment write-ups, agent page current.
            GPU still locked — camera validation remains the blocker for demo.
Decision:   Next iter: check if GPU freed. If yes, run demo_camera_ekf.py smoke test.
            If GPU still locked, work on unit tests for cam_detection_to_world().

## Iteration 80 — frame_transforms module + 18 cam-to-world tests  (2026-04-09T07:15:00Z)
Hypothesis: cam_detection_to_world and quat_to_rotmat need proper unit tests and should
            live in a reusable module (not embedded in the argparse-guarded demo script).
Change:     1. Created perception/frame_transforms.py with quat_to_rotmat() and
               cam_detection_to_world() — clean importable module.
            2. Updated demo_camera_ekf.py to import from frame_transforms.
            3. Created test_cam_to_world.py with 18 tests covering:
               - Identity/axis rotations, orthogonality, determinant checks
               - 70-degree body pitch (D435i config validation)
               - Inverse property (q vs conjugate(q))
               - Roundtrip rotmat->quat->rotmat
               - Identity/translation/rotation transforms
               - 70-deg tilt detection with constructed quat_w_ros
               - World->cam->world roundtrip (10 random poses)
               - Height batch consistency
               - Edge cases: zero detection, -q equivalence, non-unit tolerance
            Key learning: config quaternion != quat_w_ros. Isaac Lab adds body-to-ROS
            fixed rotation. Tests now construct quat_w_ros correctly via rotation matrices.
Command:    pytest scripts/perception/ → 282/282 passed (264 existing + 18 new).
            GPU blocked by policy d435i training (12288 envs, resuming from model_1499.pt).
Result:     All transforms verified. Module extracted cleanly. No regressions.
Decision:   Next iter: GPU capture smoke test (top priority). Policy training should
            finish soon — check nvidia-smi / process list.

## Iteration 81 — Demo summary visualizations + tests  (2026-04-09T08:00:00Z)
Hypothesis: Adding summary plot (trajectory + RMSE) and video compilation to demo_camera_ekf.py
            will make the demo output more useful for Daniel when GPU becomes available.
Change:     1. Added trajectory tracking (GT, EKF, detection positions per step) to demo main loop.
            2. Added _save_summary_plots(): 2-panel matplotlib figure (height trajectory + RMSE).
            3. Added _compile_video(): ffmpeg assembly of annotated frames into demo.mp4.
            4. Created test_demo_summary.py with 4 tests (plot creation, zero-detection edge case,
               no-ffmpeg graceful skip, no-frames graceful skip).
            5. Updated agents/perception.qmd with iters 80-81 summary.
            6. Processed INBOX: testing-dashboard reports GPU locked by policy PID 1118275.
            7. Attempted to kill testing-dashboard child (task done at iter 4) — c3r kill
               failed due to worktree path issue. Child has max-iter cap as safety net.
Command:    pytest scripts/perception/ → 286/286 passed (282 + 4 new).
            GPU: PID 1118275, 7839 MiB, 85% util, etime=5min (just started).
Result:     Demo script now produces summary.png + demo.mp4 automatically after run.
            All tests pass. GPU still locked — camera smoke test remains blocked.
Decision:   Next iter: check GPU status. If cleared, run demo_camera_ekf.py smoke test
            (1 env, 50 steps, headless) to validate camera sees ball. If GPU still locked,
            consider adding XY trajectory panel to summary plot, or pre-compute expected
            ball pixel location for detector tuning.

## Iteration 82 — Sim pipeline integration tests (8 new)  (2026-04-08T17:30:00Z)
Hypothesis: The sim camera pipeline (SimBallDetector → cam_detection_to_world → EKF) needs
            integration tests validating the exact wiring used in demo_camera_ekf.py.
Change:     Created test_sim_pipeline.py with 8 tests covering:
            1. Identity camera (cam = world frame)
            2. Translated camera (offset in X)
            3. 90° Y-axis rotation (cam Z → world +X)
            4. Moderate tilt (30° about Y) — validates in-frame projection check
            5. Full detect→transform→EKF convergence (stationary ball, 30mm tolerance for gravity)
            6. Ballistic trajectory tracking (30 steps, <30mm mean error)
            7. 50% dropout resilience (<80mm bound)
            8. Rotated + translated camera chain (20° Y tilt at cam_pos=[1,0,0])
            Key learnings:
            - EKF gravity model causes ~22mm systematic drift for "stationary" balls
            - 70° camera tilt causes ball to project outside 640x480 image for nearby objects
            - R_y(90°) maps [0,0,1] → [1,0,0] (not [-1,0,0] — sin, not -sin)
Command:    pytest scripts/perception/test_sim_pipeline.py -v → 8/8 passed.
            pytest scripts/perception/ → 294/294 passed (286 + 8 new).
            GPU locked by policy d435i training (PID 1118275, 8 min in, ~75 min remaining).
Result:     Full sim pipeline chain validated CPU-only. No regressions.
            Testing-dashboard child still running (c3r kill failed — worktree path issue).
Decision:   Next iter: check GPU status. If free, run debug_d435i_capture.py smoke test
            to validate camera sees ball in Isaac Lab. If still locked, consider adding
            XY trajectory panel to summary plot or updating Quarto page.

## Iteration 83 — Quarto update + video workflow + child cleanup  (2026-04-09T09:15:00Z)
Hypothesis: Preparing Quarto page and video auto-copy workflow will make GPU demo outputs
            immediately publishable when camera smoke test runs.
Change:     1. Processed INBOX: Daniel requests videos in Quarto for reasonable results — replied.
            2. Killed testing-dashboard child (c3r kill /path testing-dashboard — worked with
               explicit repo path; task was done at iter 4, idle at iter 10).
            3. Updated agents/perception.qmd with iter 82-83 summary (sim pipeline tests, child cleanup).
            4. Added _copy_to_quarto() to demo_camera_ekf.py: auto-copies summary.png to
               images/perception/ and demo.mp4 to videos/perception/ after each run.
            5. Updated fix_plan: marked integration tests done, added video Quarto task.
Command:    pytest scripts/perception/ → 294/294 passed. No GPU commands (PID 1118275, 15min in).
Result:     Quarto page current. Video workflow ready for GPU demo output.
            GPU still locked (~45 min remaining). testing-dashboard child killed successfully.
Decision:   Next iter: check GPU status. If free, run debug_d435i_capture.py smoke test
            immediately (top priority). If GPU still locked, could add expected-pixel-location
            test for camera config validation, or update the test for _copy_to_quarto().
