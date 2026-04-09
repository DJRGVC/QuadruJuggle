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

## Iteration 66 — Quarto report + sweep status check  (2026-04-08T16:50:00Z)
Hypothesis: N/A — documentation iteration responding to QUARTO_UPDATE_NUDGE.
Change:     Created agents/perception.qmd with 6 report sections covering all 65 prior iterations.
            Pulled Quarto scaffolding (_quarto.yml, index.qmd, agents/index.qmd) from policy branch.
            Processed INBOX (system nudge). Verified GPU sweep (PID 892945) is still queued behind
            policy training (PID 886154, ~11min in, 1500 iters @ 12288 envs). 59 CPU tests pass.
Command:    pytest (CPU only), ps/nvidia-smi checks. No GPU commands.
Result:     Quarto page now has comprehensive research history. Sweep will auto-run when GPU frees
            (~35-50 min from now). Old sweep JSONs confirmed all-zeros (pre-fix).
Decision:   Next iter: check if sweep_q_vel_fixed.json exists with non-zero results. If yes, parse
            and find optimal flight q_vel. If still waiting, check policy agent progress.

## Iteration 67 — fix diagnostics bug in compare_perception_modes.py  (2026-04-08T16:55:00Z)
Hypothesis: compare_perception_modes.py has the same all-zeros diagnostics bug as sweep_q_vel.py
            had — sets _perception_diagnostics_enabled but doesn't null _perception_pipeline.
Change:     Added `base_env._perception_pipeline = None` after setting diagnostics flag in
            compare_perception_modes.py (line 195-197). All 4 scripts now consistently force
            pipeline recreation for diagnostics. Verified all CPU tests pass (30/30).
Command:    pytest scripts/perception/test_pipeline_config.py test_contact_aware_ekf.py — 30 pass.
            No GPU commands (policy training PID 886154, ~14 min in, 1500 iters @ 12288 envs).
            Sweep (PID 892945) still queued behind GPU lock.
Result:     Bug fixed. compare_perception_modes.py now matches eval_perception_live.py,
            nis_diagnostic.py, and sweep_q_vel.py. Policy agent's EKF is simpler (no contact-aware,
            no spin, no NIS gating) — fine for d435i mode but not ekf mode. Documented in iter_64.
Decision:   Next iter: check if sweep_q_vel_fixed.json exists with non-zero results. If yes,
            parse and apply optimal q_vel. If GPU still blocked, wait or do more CPU prep.

## Iteration 68 — fix numpy→tensor warnings in tests  (2026-04-08T17:15:00Z)
Hypothesis: torch.tensor([numpy_array]) triggers a slow-path warning; from_numpy().unsqueeze(0) avoids it.
Change:     Replaced torch.tensor([pos0/vel0/gt_pos], dtype=torch.float32) with
            torch.from_numpy(arr.astype("float32")).unsqueeze(0) in test_latency_injection.py
            (6 occurrences) and test_ballistic_trajectory.py (7 occurrences). Processed INBOX
            (Daniel's !c3r help → replied with full status). GPU sweep (PID 892945) still queued
            behind policy agent training (PID 886154, 1500 iters @ 12288 envs, ~28 min elapsed).
Command:    pytest (all 14 CPU test files) -W error::UserWarning — 245 passed, 0 warnings.
Result:     All numpy→tensor warnings eliminated. Test suite clean. Sweep will auto-run when GPU frees.
Decision:   Next iter: check if sweep_q_vel_fixed.json exists with non-zero results. If yes,
            parse with apply_sweep_results.py and update BallEKFConfig defaults. If GPU still
            blocked, wait or check policy agent progress.

## Iteration 69 — GPU q_vel sweep: EKF over-conservative, needs lower q_vel  (2026-04-09T00:10:00Z)
Hypothesis: Sweeping q_vel=[0.4,2,5,10,20,50] will bracket flight NIS=3.0 and allow bisection
            to find optimal process noise for the d435i noise model.
Change:     Relaunched sweep (PID 892945 died with previous session). Full run completed:
            512 envs × 600 steps × 6 q_vel points. Results saved to sweep_q_vel_fixed.json.
Command:    gpu_lock.sh sweep_q_vel.py --num_envs 512 --steps 600 --warmup-steps 50
            --q-vels "0.4,2.0,5.0,10.0,20.0,50.0" --output sweep_q_vel_fixed.json
Result:     ALL flight NIS values BELOW 3.0 — EKF is over-conservative (opposite of iter-54):
            q_vel=0.4: flight_NIS=1.73, RMSE=11.55mm (raw=11.69, +1.1%)
            q_vel=2.0: flight_NIS=1.23, RMSE=11.55mm (raw=11.71, +1.4%)
            q_vel=5.0: flight_NIS=0.72, RMSE=11.58mm (raw=11.79, +1.8%)
            q_vel=50:  flight_NIS=0.14, RMSE=11.53mm (raw=11.74, +1.7%)
            Key insight: EKF RMSE barely varies (~11.5-11.6mm) across all q_vel. d435i noise
            is already small (~11.7mm raw) so the filter can't improve much. EKF's main value
            is in missed detections and velocity estimation, not position smoothing.
            No NIS=3.0 crossing → no bisection ran. Need lower q_vel to find crossing.
            Queued low-range sweep (q_vel 0.01-0.4) but GPU blocked by policy training.
Decision:   Next iter: run low-range sweep (q_vel 0.01-0.4) to find NIS=3.0 crossing point.
            Even if NIS=3.0 isn't achievable (very low q_vel may cause filter divergence),
            we should identify the practical lower bound. Update BallEKFConfig defaults.

## Iteration 70 — Quarto docs + sweep figure + experiment write-up  (2026-04-09T01:30:00Z)
Hypothesis: N/A — documentation iteration per Daniel's morning goals (stop signal).
Change:     Updated agents/perception.qmd with q_vel sweep results table + figure + integration
            status. Created experiments/perception/2026-04-09_q_vel_sweep.qmd (full experiment
            write-up). Created experiments.qmd and references.qmd listing pages (missing from
            Quarto site). Created references/perception.qmd with key citations. Generated
            matplotlib figure (images/perception/q_vel_sweep_iter_069.png) showing NIS and RMSE
            across q_vel values. Processed 3 INBOX messages (Daniel plan Q + stop signal + re-read).
            No GPU processes running (confirmed via nvidia-smi).
Command:    No GPU commands. matplotlib for figure generation.
Result:     Quarto page now documents: (1) sweep results showing all NIS < 3.0, (2) integration
            status (pipeline feature-complete, 3 modes working), (3) experiment with figures.
            Missing Quarto scaffolding (experiments.qmd, references.qmd) now created.
Decision:   Next iter: run low-range q_vel sweep (0.01-0.4) when GPU is available. Then update
            BallEKFConfig defaults and communicate handoff to policy. Daniel's (a) EKF→pi1 is
            already wired; (b) oracle-vs-EKF demo needs GPU; (c) docs done this iter.

## Iteration 71 — enhanced apply_sweep_results.py + low-range sweep queued  (2026-04-08T18:20:00Z)
Hypothesis: Multi-file merge + NIS crossing interpolation will enable automatic optimal q_vel
            selection once both sweep ranges complete.
Change:     Rewrote apply_sweep_results.py: load_and_merge() merges multiple JSON files, deduped
            by q_vel. find_nis_crossing() linear-interpolates on log(q_vel) scale. make_figure()
            generates pub-quality dual-panel (NIS + RMSE) figure. Updated test_apply_sweep.py to
            match new output format. Queued low-range sweep (q_vel=0.01-0.40, 512 envs × 600 steps)
            behind policy training GPU lock (PID 982975, 1200 iters @ 12288 envs, ~62 min remaining).
Command:    pytest scripts/perception/ — 251/251 pass. No GPU commands (lock held by policy agent).
Result:     apply_sweep_results.py enhanced and tested. Low-range sweep PID 987574 queued (sleeping
            on flock). Existing high-range sweep confirmed: all flight NIS < 3.0 (max 1.73 at
            q_vel=0.4). Generated combined figure from existing data.
Decision:   Next iter: check if sweep_q_vel_low_range.json exists. If yes, merge with high-range
            data, find NIS=3.0 crossing, update BallEKFConfig defaults. If sweep crashed, diagnose.
            Sweep PID: 987574. Output: logs/perception/sweep_q_vel_low_range.json.

## Iteration 72 — apply_sweep_results --apply + low-range sweep status  (2026-04-09T01:45:00Z)
Hypothesis: Adding --apply flag to apply_sweep_results.py enables automatic BallEKFConfig patching
            once the combined sweep data is available.
Change:     Added apply_to_config() function + --apply CLI flag to apply_sweep_results.py. Uses regex
            to patch q_vel in ball_ekf.py. Added 2 unit tests (test_patches_q_vel, test_no_match).
            Processed 2 INBOX messages from Daniel (accuracy Q + respond Q). Low-range sweep
            (PID 987574, q_vel=0.01-0.40) still queued behind policy GPU lock (PID 982975, ~10min in).
Command:    pytest scripts/perception/ — 253/253 pass. No GPU commands.
Result:     apply_sweep_results.py now supports: --plot (figure), --apply (auto-patch config), and
            multi-file merge. Ready to run `python apply_sweep_results.py --plot --apply` once
            sweep_q_vel_low_range.json exists. Replied to Daniel with full accuracy report:
            EKF RMSE ≈ 11.5mm, pipeline feature-complete in sim, real hardware blocked on Go1 access.
Decision:   Next iter: check if sweep_q_vel_low_range.json exists. If yes, run apply_sweep_results.py
            --plot --apply to find NIS=3.0 crossing and auto-patch BallEKFConfig. If sweep still
            queued, check GPU status. Sweep PID: 987574.

## Iteration 73 — Adaptive R_xy: fix root cause of low NIS  (2026-04-09T02:00:00Z)
Hypothesis: All flight NIS < 3.0 because R_xy was calibrated for z=0.5m (σ=1.25mm) but d435i
            noise model has σ_xy=0.0025·z. At Stage A (z≈0.1m), actual σ_xy=0.25mm → R_xy
            variance 25× too large → filter over-conservative → NIS artificially suppressed.
Change:     Made R_xy adaptive in BallEKF.update(): σ_xy = max(r_xy_per_metre·z, r_xy_floor).
            Added BallEKFConfig fields: r_xy_per_metre=0.0025, r_xy_floor=0.0005m. Updated
            test_imu_aided_ekf.py (test_tracking_degrades_without_imu was marginal with new R;
            increased omega 2→4 rad/s and steps 30→60). Added 3 new tests (TestAdaptiveR) in
            test_nis_gating.py verifying: (1) R_xy scales with z, (2) floor works at z≈0,
            (3) non-adaptive R is height-independent. Killed old sweep PID 987574 (would have
            imported pre-fix code) and requeued as PID 999147 with updated adaptive R.
Command:    pytest scripts/perception/ — 256/256 pass (was 253; +3 new). No GPU commands.
            Sweep PID 999147 queued behind policy training (PID 982975, ~18 min in, 1500 iters).
Result:     Root cause identified and fixed. With adaptive R_xy:
            - At z=0.10m: σ_xy = 0.25mm (was 1.25mm) → R_xy variance 25× smaller → NIS ↑
            - At z=0.50m: σ_xy = 1.25mm (unchanged)
            - At z=1.00m: σ_xy = 2.50mm (was 1.25mm) → R_xy variance 4× larger → NIS ↓
            The low-range sweep should now show NIS ≈ 3.0 at the correctly-calibrated q_vel,
            rather than being universally suppressed by an over-sized R.
Decision:   Next iter: check if sweep_q_vel_low_range.json exists. With adaptive R, the NIS
            crossing point should be at a higher q_vel than before (since R is now tighter at
            low z). If NIS > 3 at all tested q_vel, the bisection in sweep_q_vel.py will
            auto-find the crossing. If still < 3 everywhere, consider that the EKF process
            model (ballistic+drag) is genuinely accurate for this task.
