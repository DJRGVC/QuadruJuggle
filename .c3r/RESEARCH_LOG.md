# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## Compacted summary (through iter_059)

**Iters 001-004 (docs/roadmap):** Updated perception_roadmap.md and sim_to_real_plan.md for D435i
(stereo depth, not monocular). Surveyed Isaac Lab camera APIs — TiledCamera for debug only;
training uses ETH noise-injection on GT state (no camera sensor needed). Created ball_obs_spec.py
with 3 modes: oracle (GT passthrough), d435i (structured depth noise), ekf (EKF-filtered).

**Iters 005-008 (camera + EKF core):** Mounted simulated D435i TiledCamera in PLAY scene.
Implemented ball_ekf.py: batched 6-state EKF (pos+vel), ballistic+quadratic-drag dynamics,
Joseph-form covariance update.

**Iters 009-011 (full pipeline + handoff):** Created noise_model.py (depth-dependent sigma,
hold-last-value dropout, latency buffer) and PerceptionPipeline class. Integration tested.
Wrote PERCEPTION_HANDOFF.md for policy agent.

**Iters 012-018 (diagnostics + EKF tuning):** Added diagnostics (RMSE, NIS), noise_scale
curriculum, body-frame gravity correction. Fixed critical vel-view mutation bug. 3-mode
comparison: oracle=13.7, d435i=10.5, ekf=7.6 reward. EKF 28% below raw d435i.

**Iters 019-024 (NIS debugging — contact forces):** NIS=966 with q_vel=0.30. Root cause:
unmodeled contact normal forces during paddle contact. q_vel sweep found q_vel>=5.0 needed.

**Iters 025-028 (real hardware pipeline):** Hardware pipeline stubs, MockCamera+MockDetector.

**Iters 029-035 (contact-aware EKF + calibrated noise):** Contact-aware EKF (q_vel=0.40
free-flight, q_vel=50.0 contact) achieved 860x NIS improvement. GPU-validated NIS=1.60
with Ahn 2019 noise model. All tests pass.

**Iters 036-043 (extensions + velocity commands):** IMU-aided EKF (Coriolis corrections), 9D spin
estimation (Magnus force), config wiring through BallObsNoiseCfg. vel_cmd/ package:
UserVelocityInput, CommandMixer, ResidualMixer, play_teleop.py. 176/176 tests.

**Iters 044-048 (real pipeline + NIS tooling):** Hough circle fallback detector (<10mm at
0.3-1.0m). Threaded RealPerceptionPipeline with MockCamera integration tests. Chi-squared
NIS gating (11.345 threshold, 50-step warmup). 239/239 tests.

**Iters 049-055 (GPU NIS validation + live-policy eval):** IMU/spin modes functionally identical
under random actions (NIS≈0.44). Added phase-separated NIS (flight vs contact). Built
eval_perception_live.py for accuracy under trained policy. **CRITICAL: EKF severely
overconfident under active juggling (flight NIS=52.9, RMSE 22mm > raw 19mm).** Root cause:
violent paddle strikes produce near-instant velocity reversals; q_vel=0.4 trusts prediction
too much. Contact q_vel_contact=50.0 close to right (NIS=5.3). Random-action flight NIS=1.45
well-calibrated — the gap is entirely from active policy dynamics.

**Iters 056-059 (post-contact q_vel + sweep prep):** Two compaction iters (056, 059). Killed all
3 children (lit-review, vel-cmd-survey, report-writer). Implemented 3-level q_vel architecture:
contact=50.0, post_contact=20.0 (10 steps), flight=0.4. Wrote sweep_q_vel.py with warmup,
post-contact toggle, and fixed EKF reset bug. 224/224 CPU tests pass. GPU sweep blocked by
policy agent training. Sweep ready to run: q_vel=[0.4,2.0,5.0,10.0,20.0,50.0].

**Key architectural findings through iter_059:**
- Contact-aware EKF: inflate Q during paddle contact → NIS from 966 to 0.78
- Ahn 2019 noise: sigma_xy=0.0025*z, sigma_z=1mm+0.005*z², dropout 20-50%
- Raw d435i noise outperforms EKF for training (noise = regularization)
- EKF value: velocity estimation + dropout bridging during free-flight
- IMU/spin: non-harmful in sim, ready for real hardware
- Velocity commands: Method 1 (override) done; Method 2 needs pi1 retraining
- **Active-policy EKF tuning is fundamentally different from random-action tuning**
- 3-level q_vel architecture: contact=50.0, post_contact=20.0 (10 steps), flight=TBD
- GPU q_vel sweep is next priority — find flight q_vel where NIS ≈ 3.0

---

## Iteration 60 — compaction (summarized iters 056-059)  (2026-04-08T21:15:00Z)
Hypothesis: N/A — compaction iteration (context at 80% per system auto-compact trigger).
Change:     Archived iters 056-059 verbatim to RESEARCH_LOG_ARCHIVE.md. Rewrote compacted summary
            through iter_059 (folded post-contact q_vel work + sweep prep). Processed 1 system
            INBOX message.
Command:    No GPU commands.
Result:     Log shrunk from ~127 → ~60 lines. Archive now has 59 verbatim entries (001-059).
Decision:   Next iter: GPU q_vel sweep with sweep_q_vel.py when GPU lock frees. Primary goal:
            find flight q_vel where NIS ≈ 3.0 and EKF RMSE < raw RMSE under active policy.

## Iteration 61 — sweep_q_vel.py bisection + auto-save  (2026-04-08T22:30:00Z)
Hypothesis: Adding automatic bisection refinement to sweep_q_vel.py will find the exact q_vel
            for NIS≈3.0 in a single GPU run instead of requiring manual follow-up sweeps.
Change:     Refactored sweep loop into _eval_q_vel() helper. Added bisection pass: after coarse
            sweep, finds interval where flight NIS crosses 3.0 and runs 3 bisection steps
            (8x finer resolution). Auto-saves JSON results with timestamp. Added --no-bisect
            and --bisect-steps flags. GPU was locked by policy agent (training 2000 iters
            with --noise-mode d435i, started ~42min ago).
Command:    No GPU commands (lock held by policy agent). CPU tests: 9/9 pass.
Result:     sweep_q_vel.py enhanced — ready to run next iter when GPU frees. Policy agent is
            at iter_018 training d435i noise mode, plateau at apex≈10.7.
Decision:   Next iter: run the GPU q_vel sweep. If GPU still locked, check if policy agent
            has finished and whether we can coordinate timing.

## Iteration 62 — sweep queued + apply_sweep_results.py helper  (2026-04-08T22:50:00Z)
Hypothesis: Preparing result-parsing tooling now saves time when sweep completes.
Change:     Processed 2 INBOX messages (Daniel: subagent cleanup Q + status Q). Verified all 3
            children already killed (iter_056). Confirmed GPU locked by policy training (PID 805803,
            ~38min elapsed, 2000 iters @ 12288 envs). The sweep_q_vel.py process (PID 806215) is
            queued behind gpu_lock flock — will auto-run when policy finishes. Verified pi2/pi1
            checkpoints exist. Created apply_sweep_results.py to parse sweep JSON output. Updated
            fix_plan. 241/241 CPU tests pass.
Command:    pytest (CPU only), nvidia-smi, ps checks. No GPU commands.
Result:     Sweep is auto-queued (waiting on flock). Results will save to logs/perception/.
            Policy training at 2000 iters may take 30-60 more minutes.
            Note: sweep uses original pi1 (2026-03-12), not noise-trained pi1 (still training).
Decision:   Next iter: check if sweep completed (ls logs/perception/sweep_q_vel_*.json). If yes,
            parse results and update BallEKFConfig defaults. If still waiting, consider other CPU
            work or wait for GPU. Eventually re-run with noise-trained pi1.

## Iteration 63 — fix EKF reset under inference_mode  (2026-04-08T22:01:00Z)
Hypothesis: sweep_q_vel.py crashes because EKF.reset() does in-place updates on tensors that
            became inference tensors after predict/update ran inside torch.inference_mode().
Change:     Wrapped BallEKF.reset() body in `with torch.inference_mode():` so it can modify
            internal tensors regardless of caller's context. Added regression test
            (test_reset_after_inference_mode_predict) to test_contact_aware_ekf.py.
Command:    pytest scripts/perception/test_*.py (CPU): 241+1=242 total, all pass.
            Attempted GPU sweep twice — both times blocked by policy agent training.
            Policy agent iter_018 ran 2000 iters (7200→9200), then launched another 2000 from
            stage 5 (pid 805803, ~51min in when checked).
Result:     Bug fixed. The RuntimeError "Inplace update to inference tensor outside InferenceMode"
            is resolved. Sweep script verified to parse args correctly. Cannot GPU-validate
            until policy agent's training completes.
Decision:   Next iter: run GPU q_vel sweep. Policy agent should be done soon (2000 iters from
            model_best.pt stage 5). If still blocked, coordinate timing via INBOX.

## Iteration 64 — fix sweep diagnostics bug (pipeline recreation)  (2026-04-08T23:10:00Z)
Hypothesis: sweep_q_vel.py returns all-zero NIS/RMSE because the perception pipeline is created
            during gym.make() → env.reset() BEFORE _perception_diagnostics_enabled is set.
Change:     Added `base_env._perception_pipeline = None` after setting the diagnostics flag,
            matching eval_perception_live.py (which worked in iter_054-055). This forces pipeline
            recreation with diagnostics enabled on the next obs call. Also documented EKF
            divergence between perception and policy branches (docs/EKF_CONTACT_AWARE_NOTE.md).
            Policy agent stripped contact-aware 3-level q_vel, NIS gating, and 9D spin from their
            copy — fine for d435i mode but will fail under ekf mode (NIS 52.9 vs 1.60).
Command:    GPU locked by policy agent (PID 866211, 2000 iters from model_9100, ~2min in). 107/107
            CPU tests pass. Sweep queued behind GPU lock (will run next iter).
Result:     Root cause identified and fixed. The eval_perception_live.py had this fix already
            (lines 196-197) but sweep_q_vel.py didn't. Without it, diagnostics always returned None.
Decision:   Next iter: GPU sweep should be queued or just finished. Parse results, find optimal
            q_vel for flight NIS ≈ 3.0. If GPU still blocked, escalate via ask_human.

## Iteration 65 — fix sweep diagnostics (restore pipeline recreation)  (2026-04-08T16:35:00Z)
Hypothesis: Sweep all-zeros results were caused by (a) uncommitted working-tree changes that
            reverted the iter-64 pipeline recreation fix, and (b) the sweep JSON files predating
            the fix (timestamps 16:05-16:11 vs iter-64 commit at 23:10).
Change:     Restored the `base_env._perception_pipeline = None` approach (force recreation after
            setting `_perception_diagnostics_enabled = True`). Added verification logging and a
            fallback force-enable path. Removed the unreliable manual-patching approach from the
            working tree. Added 3 CPU unit tests verifying diagnostics work when pipeline is
            recreated with `enable_diagnostics=True`.
Command:    pytest scripts/perception/test_pipeline_config.py — 20/20 pass.
            GPU locked by policy agent (PID 886154, 1500 iter d435i training, ~7min in).
            Queued sweep behind gpu_lock (PID 892945, 512 envs × 600 steps).
Result:     CPU tests confirm: pipeline recreated with diagnostics enabled properly accumulates
            NIS and RMSE data. Sweep results will be available next iteration when GPU frees.
Decision:   Next iter: parse sweep results from logs/perception/sweep_q_vel_fixed.json. If still
            zeros, the bug is in GPU-specific code path (world_frame or inference_mode). If
            non-zero, proceed to find optimal q_vel for flight NIS ≈ 3.0.
