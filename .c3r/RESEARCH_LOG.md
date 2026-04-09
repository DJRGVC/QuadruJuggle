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

## Iteration 76 — Camera FOV fix: 45° → 75° tilt  (2026-04-08T19:35:00Z)
Hypothesis: Blank camera frames caused by ball being below camera FOV at 45° tilt;
            75° tilt + adjusted position will bring airborne ball into view.
Change:     Updated BallJuggleHierSceneCfg_DEBUG camera mount:
            - Tilt: 45° → 75° (quaternion (0.9239,-0.3827,0,0) → (0.7934,-0.6088,0,0))
            - Position: (-0.05, 0, 0.08) → (-0.08, 0, 0.06) for better geometry
            - FOV now covers 46°-104° above horizontal (was 16°-74°)
            Updated debug_d435i_capture.py: applies 2 m/s upward ball velocity before
            capture so ball is airborne (at ~0.2m elevation ≈ 60°, centred in new FOV).
Command:    Syntax checks pass. 104/104 CPU tests pass. GPU smoke test blocked by
            policy agent training (ETA ~60 min from 19:19 UTC).
Result:     Code changes committed. GPU validation deferred — policy holds gpu_lock.
            FOV geometry: ball at rest (11° elev) → OUT of FOV (expected for juggling);
            ball at 0.2m apex (60°) → IN; ball at 0.5m (80°) → IN; ball at 1.0m (85°) → IN.
Decision:   Next iter: run GPU debug capture when lock frees. If ball visible in frame,
            proceed to Hough circle detector integration (camera→detect→EKF pipeline).
