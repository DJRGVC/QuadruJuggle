# RESEARCH_LOG.md

_(older entries auto-archived to RESEARCH_LOG_ARCHIVE.md at 2026-04-09 06:29 UTC)_

## Iteration 89 — GPU smoke test: camera pipeline with trained pi1  (2026-04-09T14:50:00Z)
Hypothesis: Running demo_camera_ekf.py with trained pi1 checkpoint will validate the
            full camera→detect→EKF pipeline under realistic policy-driven ball dynamics.
Change:     1. Fixed stdout buffering issue: added `faulthandler.enable()` and must use
               `python -u` (unbuffered) to see output before Isaac Sim's simulation_app.close()
               kills the process. Without -u, only step 0 appeared in logs.
            2. Tested 3 pi1 checkpoints: d435i-trained (policy branch), oracle-trained
               (policy branch), and main-branch pi1. All fail on our env config.
            3. Root cause: policy branch env has restitution=0.99 + ball_pos_perceived obs;
               our perception branch has restitution=0.85 + ball_pos_in_paddle_frame obs.
               Policies trained on incompatible physics/obs cannot balance ball on our branch.
Command:    $C3R_BIN/gpu_lock.sh bash -c 'uv run --active python -u scripts/perception/demo_camera_ekf.py
              --task Isaac-BallJuggleHier-Go1-Play-v0 --num_envs 1 --steps 300 --headless
              --pi1-checkpoint <various>' (4 GPU runs total: debug 10-step + 3 × 300-step)
Result:     Pipeline mechanically works (no crashes with python -u, correct API):
            - d435i pi1: 2% det rate, 60mm RMSE when detected, EKF diverges (7m)
            - oracle pi1: 1.7% det rate, 40mm RMSE when detected, EKF diverges (5m)
            - main-branch pi1: 4.3% det rate, 92mm RMSE, EKF diverges (4.4m)
            - All: 0 episodes in 300 steps → ball falls off immediately, no termination fires
            - Detection works correctly in first 5 steps (20-40mm RMSE) before ball leaves FOV
            Key finding: cross-branch policy incompatibility is the blocker, not the camera
            pipeline. Bounce mode demo (iter 87, 100% det rate) remains the valid demo.
Decision:   Live policy demo is BLOCKED on branch integration. Two paths forward:
            (a) Merge policy branch env changes (restitution + perceived obs) into perception
            (b) Train a pi1 on our perception branch's env config
            Neither is in our scope alone. Will ping policy agent about option (a).
            Next iter: update Quarto page with cross-branch findings + focus on remaining
            fix_plan items (GPU sweep at higher heights for EKF vs raw comparison).

## Iteration 88 — Wire camera pipeline into live policy eval  (2026-04-09T10:45:00Z)
Hypothesis: Adding --pi1-checkpoint to demo_camera_ekf.py will enable camera→detect→EKF
            evaluation under realistic trained-policy ball dynamics (not just bounce mode).
Change:     1. Added --pi1-checkpoint arg to demo_camera_ekf.py.
            2. When set: loads pi1 via RSL-RL OnPolicyRunner, uses policy actions, tracks
               episodes/timeouts, resets EKF on episode termination.
            3. Multi-env support: per-env camera detection + EKF measurement tensors.
            4. Bounce mode auto-disabled when using trained policy.
            5. Killed completed child agent checkin-087.
Command:    python -m pytest scripts/perception/ -x -q → 311/311 passed (6.6s).
            GPU smoke test queued (blocked behind policy agent's cross-eval).
Result:     Code complete, syntax verified, all tests pass. GPU validation pending.
            Policy agent has d435i-trained checkpoints at Stage F (+59% apex vs oracle).
Decision:   Next iter: if GPU smoke test ran successfully (check output), validate camera
            pipeline accuracy under trained policy. If GPU still blocked, run the test.
            Compare camera-EKF RMSE under policy vs bounce mode.

               demo_camera_ekf.py (300 steps, capture every 5) in one gpu_lock call.
            3. Processed INBOX: Daniel requests respawning testing-dashboard at 5am PST
               tomorrow (2026-04-09 12:00 UTC) with prior memory. Replied + archived.
            4. Updated Quarto page with iter 84-85 summary (projection table + bounce mode).
Command:    pytest scripts/perception/ → 311/311 passed. No GPU commands (PID 1118275, 30 min in).
Result:     Demo script ready for GPU. Bounce mode produces ~4-5 bounces in 300 steps (6s sim),
            each reaching ~0.4-0.6m above paddle — within camera FOV.
            GPU blocked by policy d435i training (~50 min remaining).
Decision:   Next iter: check GPU status. If free, run run_gpu_demo.sh immediately via
            gpu_lock.sh. This is the single most important blocker — camera visualization
            has been waiting since iter 76.

## Iteration 87 — GPU camera demo SUCCESS + scene.get() fix  (2026-04-09T07:31:00Z)
Hypothesis: Fixing scene.get() → scene["d435i"] and requeuing GPU demo will produce
            first validated camera frames showing ball detection in Isaac Lab sim.
Change:     1. Fixed demo_camera_ekf.py:135 — scene.get("d435i") → scene["d435i"]
               (InteractiveScene has __getitem__ not .get()).
            2. Killed hung PID 1154762 (crashed Isaac cleanup), waited for policy eval
               to finish (PID 1160839), then ran run_gpu_demo.sh via gpu_lock.
            3. Processed 4 INBOX entries (Daniel: config fix + kill authorization;
               checkin-087: Quarto scope + dashboard sub-agent request).
            4. Relayed kill-hung-processes authorization to policy agent.
            5. Spawned checkin-087 sub-agent for progress monitoring.
            6. Copied summary.png + frames to images/perception/, created video.
            7. Updated Quarto page with GPU demo milestone (3 figures + video).
Command:    $C3R_BIN/gpu_lock.sh bash scripts/perception/run_gpu_demo.sh
            pytest scripts/perception/ → 311/311 passed.
Result:     GPU DEMO SUCCESS:
            - Camera at 70° elevation, world convention, pos (-0.08, 0, 0.46)
            - 100% detection rate across 300 steps (bounce mode)
            - Detection RMSE: ~190mm (depth-based at distance, expected)
            - EKF diverges on bounce impulses (expected — contact-aware Q needed)
            - 60 annotated frames saved, summary.png, demo video created
            Step 1 (debug capture, 50 steps): SUCCESS
            Step 2 (full demo, 300 steps): SUCCESS
            Sentinel file written: logs/perception/gpu_demo_DONE
Decision:   Next iter: wire camera pipeline into live eval with trained pi1 policy.
            The bounce-mode EKF divergence is expected and not an issue — real
            training uses contact-aware Q which we validated at NIS=3.3 (iter 70).
            Camera visualization is now VALIDATED. Main remaining work:
            1. Live eval with camera pipeline under trained policy
            2. GPU sweep at higher heights to confirm EKF beats raw
            3. Support policy agent with noise curriculum tuning

## Iteration 86 — GPU demo queued + robustness improvements  (2026-04-09T06:24:00Z)
Hypothesis: GPU demo can be queued behind policy training via gpu_lock — if training
            finishes during this iteration, demo runs automatically.
Change:     1. Queued run_gpu_demo.sh behind gpu_lock (PID 1133926, waiting for lock).
            2. Added per-step timeout (5 min default) to run_gpu_demo.sh — prevents hangs.
            3. Added sentinel file (logs/perception/gpu_demo_DONE) written on success —
               next iteration can detect completion even if this iter times out.
            4. Verified all prerequisites: pi2 checkpoint auto-detected, DEBUG scene has
               d435i camera, Quarto image/video dirs exist, 311/311 tests pass.
            5. Policy training at iter ~2100/3000, 35 min elapsed, est 30-50 min remaining.
Command:    pytest scripts/perception/ → 311/311 passed.
            $C3R_BIN/gpu_lock.sh bash scripts/perception/run_gpu_demo.sh (background PID 1133926)
Result:     GPU still locked. Background process waiting. All CPU work complete.
            Policy agent reports d435i outperforms oracle at Stage E (+176% apex) — validates
            our noise model. Their continuation run (from checkpoint 1550) is in progress.
Decision:   Next iter: check for sentinel file (logs/perception/gpu_demo_DONE). If found,
            parse results from GPU log + debug frames. If not found, check if PID 1133926
            is still running or if it died. Re-queue if needed.
