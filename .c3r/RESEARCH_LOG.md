# RESEARCH_LOG.md

_(older entries auto-archived to RESEARCH_LOG_ARCHIVE.md at 2026-04-09 12:53 UTC)_

Result:     Full analysis pipeline ready: anchor ablation script now produces 4 outputs:
            (1) height-binned comparison, (2) phase RMSE detail, (3) anchor-ON timeline,
            (4) anchor-OFF timeline. GPU still occupied by policy agent.
Decision:   Next iter: run anchor ablation on GPU if available. If still blocked,
            consider building a synthetic trajectory generator for offline pipeline
            testing, or check if policy agent needs support.

## Iteration 111 — Height-dependent velocity noise fix  (2026-04-11T05:50:00Z)
Hypothesis: The d435i velocity noise model uses fixed z_nominal=0.5m regardless of
            actual ball height. At Stage G (1.0m above paddle), velocity noise should
            be ~2.7× higher. Fixing this to be height-dependent produces more realistic
            training noise for the policy agent.
Change:     1. Modified _apply_d435i_vel_noise() to accept optional pos_b tensor.
               When provided, computes per-env distance-dependent sigma_xy, sigma_z,
               and dropout — matching the position noise model.
            2. Updated ball_vel_perceived() to pass ground-truth ball position to
               the noise function.
            3. Backward-compatible: pos_b=None falls back to old z_nominal=0.5m.
            4. 6 new tests (test_vel_noise_height_dep.py): height scaling, quadratic z,
               dropout distance dependence, backward compat, nominal match at 0.5m.
            5. Created plot_noise_vs_height.py — 4-panel characterization figure.
            6. Updated Quarto page with figure and explanation.
            7. Pinged policy agent about the fix (their Stage G run uses old code).
Command:    pytest scripts/perception/ → 440/440 passed (9.86s). No GPU needed.
Result:     At z=1.0m (Stage G target): vel noise σ_z = 2.7× old estimate,
            σ_xy = 2.0× old. At z=0.1m (Stage A): identical to old model.
            Position noise was already correct — only velocity was affected.
            Figure: images/perception/noise_vs_height_iter111.png
Decision:   Next iter: run anchor ablation on GPU if available. If policy agent
            confirms Stage G retrain needed with corrected noise, coordinate timing.

## Iteration 112 — Parameterized perception eval script  (2026-04-09T06:05:00Z)
Hypothesis: Hardcoded eval scripts (run_full_eval.sh, run_height_sweep_eval.sh) with
            baked-in checkpoint paths make it hard to evaluate new checkpoints quickly.
            A parameterized script will enable immediate evaluation when Stage G finishes.
Change:     Created run_perception_eval.sh with CLI args: --pi1, --pi2, --targets,
            --noise-mode, --label, --anchor, --camera-scheduling, --steps, --num-envs.
            Produces per-target trajectory.npz + summary table with det%, mean/max height,
            position RMSE. Saves eval_config.txt for reproducibility.
Command:    No GPU needed — script creation + test suite verification (440/440 pass).
            Checked policy training progress: Stage G at ~60 min, ETA 23 min.
            Checkpoints at 2026-04-09_04-56-44/model_best.pt (iter 3000+).
            Training metrics: 44% timeout, ep_len=817, apex=7.4, actively juggling.
Result:     Script ready at scripts/perception/run_perception_eval.sh. Updated fix_plan
            with Stage G eval commands. Policy agent uses pi2 from 09-04-32 (matches
            our default). Their noise model is position-only (no separate vel noise fn),
            so the iter 111 vel noise fix only affects our direct-injection path.
Decision:   Next iter: if GPU free, run Stage G eval immediately:
            `run_perception_eval.sh --pi1 .../2026-04-09_04-56-44/model_best.pt
            --targets "0.10 0.30 0.50 0.70 1.00" --anchor --camera-scheduling --label stage_g`
            If GPU still blocked, run anchor ablation or prepare Quarto experiment page.

## Iteration 113 — Stage G comparison tooling + smoke test  (2026-04-09T11:55:00Z)
Hypothesis: A batch comparison script (oracle vs d435i, anchor ON vs OFF) will enable
            efficient evaluation of Stage G checkpoint across all conditions in one run.
Change:     1. Created run_stage_g_comparison.sh — runs 4 eval variants (d435i+anchor,
               d435i-no-anchor, oracle+anchor, oracle-baseline) across N target heights.
            2. Created plot_stage_comparison.py — 4-panel comparison figure (det rate,
               max height, RMSE pos, RMSE z) with grouped bar charts per variant.
            3. Added test_plot_stage_comparison.py — 7 tests for data loading + plotting.
            4. All 447/447 tests pass (7 new).
            5. Policy agent Stage G training completed: 50.7% timeout, ep_len~750,
               apex=7.0, release_vel=0.8. Checkpoint: 2026-04-09_04-56-44/model_best.pt.
            6. Launched smoke test (20 steps, 2 envs) — still in cold-start shader
               compilation after 20 min. Isaac sim cold start is much slower than expected.
Command:    pytest scripts/perception/ -x -q → 447/447 passed (10.02s).
            GPU smoke test still running in background (PID 1309706).
Result:     Tooling ready. Policy Stage G finished successfully. Smoke test blocked by
            Omniverse cold-start shader compilation (~20 min on first run after idle).
            Need to account for this in next iteration's time budget.
Decision:   Next iter: check if smoke test completed (background PID 1309706). If yes,
            verify trajectory.npz was produced correctly, then run full comparison.
            If cold start takes >25 min total, run comparison with longer time budget.
            Stage G checkpoint: /home/daniel-grant/Research/QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier/2026-04-09_04-56-44/model_best.pt

## Iteration 114 — Camera scheduling starvation fix + Stage G eval  (2026-04-09T14:00:00Z)
Hypothesis: Stage G eval with d435i+anchor+camera-scheduling will show improved EKF
            tracking at higher target heights thanks to the policy agent's mixed-target
            training.
Change:     1. Fixed UnboundLocalError in demo_camera_ekf.py — `robot` and constants
               `_PADDLE_OFFSET_Z`/`_BALL_RADIUS` were used before definition (moved up).
            2. Fixed bash f-string escaping in run_perception_eval.sh summary table.
            3. Added camera scheduling starvation override: when EKF goes >50 steps
               without a measurement, force camera on regardless of phase tracker state.
               Prevents death spiral: wrong phase → no detection → EKF drift → wrong phase.
            4. Ran Stage G eval (d435i, anchor, camera-scheduling, 5 targets, 1500 steps).
Command:    GPU eval: run_perception_eval.sh --targets "0.10 0.30 0.50 0.70 1.00"
            --anchor --camera-scheduling --label stage_g_d435i
Result:     CRITICAL FINDING: Camera scheduling death spiral confirmed.
            Detection rate 1.1-1.4% across all targets. Phase tracker classified 98.5%
            as "contact" because EKF Z was wrong (mean EKF_Z=0.459 vs GT_Z=0.798).
            EKF RMSE 0.10-0.25m — too high for usable obs. Ball reached max 1.09m at
            target=0.50 and 1.28m at target=0.70, confirming Stage G policy works.
            Root cause: camera scheduling uses phase tracker (EKF-based) → EKF has no
            measurements → drifts low → phase tracker says "contact" → camera stays off.
            FIX APPLIED: starvation override (>50 steps → force camera on).
            Re-eval needed to verify fix works.
Decision:   Next iter: re-run Stage G eval with starvation fix to verify detection rate
            improves. Then run oracle baseline for comparison.

## Iteration 115 — Starvation override threshold reduction  (2026-04-09T07:30:00Z)
Hypothesis: The camera scheduling death spiral (iter 114: 1.1% detection rate) is
            caused by SCHED_STARVE_LIMIT=50 being too high. At 50Hz, 50 steps = 1s,
            during which the ball completes a full bounce cycle with no measurements.
            Reducing to 10 steps (0.2s) will break the spiral near episode start
            while preserving contact-phase camera savings.
Change:     1. Reduced default SCHED_STARVE_LIMIT from 50 to 10 steps.
            2. Added --starve-limit CLI arg to demo_camera_ekf.py for tuning.
            3. Wired --starve-limit into run_perception_eval.sh.
            4. Root cause analysis: phase_tracker.reset() → CONTACT, but ball spawns
               above paddle → camera stays off → EKF never gets first measurement.
               With limit=10, camera fires within 0.2s of episode start.
            5. Added 2 tests for starvation override logic (test_phase_tracker.py).
            6. Updated Quarto page with iter 112-114 findings (death spiral writeup).
            7. All 449/449 tests pass.
Command:    pytest scripts/perception/ -x -q → 449/449 passed (9.96s). No GPU (blocked
            by policy agent Stage G training PID 1334185).
Result:     Code ready. GPU eval needed to verify detection rate improvement.
            Expected: detection rate should jump from ~1% to >30% with limit=10,
            since ball spends ~30-40% of episode in flight (at Stage G targets).
Decision:   Next iter: run Stage G eval with starve_limit=10 when GPU free. Compare
            detection rates vs iter 114 (limit=50). If GPU still blocked, consider
            preparing the anchor ablation analysis or Quarto experiment page.

## Iteration 116 — Stage G starvation eval + pi2 default fix  (2026-04-09T14:30:00Z)
Hypothesis: Reducing SCHED_STARVE_LIMIT from 50→10 (iter 115) will break the camera
            scheduling death spiral, raising detection rate from ~1% to >30%.
Change:     1. Fixed pi2 default in run_perception_eval.sh: 09-04-32 → 17-16-01
               (matching the checkpoint the policy agent uses for Stage G training).
            2. Ran full Stage G d435i eval: 5 targets (0.10-1.00), 1500 steps, 4 envs,
               anchor=ON, camera-scheduling=ON, starve_limit=10.
            3. Attempted oracle comparison but GPU blocked by policy training.
Command:    gpu_lock.sh bash run_perception_eval.sh --pi1 .../2026-04-09_07-19-27/model_best.pt
            --targets "0.10 0.30 0.50 0.70 1.00" --anchor --camera-scheduling
            --starve-limit 10 --label stage_g_starve10 --steps 1500 --num-envs 4
Result:     STARVATION FIX: MARGINAL IMPACT. Detection rates still 1.0-2.0% (vs 1.1%
            in iter 114 with limit=50). Root cause is NOT starvation — it's that the
            ball rarely enters flight.

            | Target | Det% | EKF_RMSE | TO%  | Bounces | Flight% | Peak   |
            |--------|------|----------|------|---------|---------|--------|
            | 0.10   | 1.0% | 0.145m   | 75%  | 0.0     | 0.0%    | 0.000m |
            | 0.30   | 1.1% | 0.146m   | 50%  | 0.5     | 2.0%    | 0.662m |
            | 0.50   | 1.0% | 0.142m   | 50%  | 0.2     | 0.9%    | 0.485m |
            | 0.70   | 1.7% | 0.444m   | 0%   | 1.2     | 6.2%    | 0.503m |
            | 1.00   | 2.0% | 0.436m   | 0%   | 2.2     | 23.2%   | 0.701m |

            KEY FINDINGS:
            - Low targets (0.10-0.50): Policy BALANCES (50-75% TO), never bounces.
              EKF RMSE ~0.14m from anchor updates alone. Detection useless.
            - High targets (0.70-1.00): Policy ATTEMPTS juggling but drops ball fast
              (0% TO, 8-19 episodes). Ball reaches 0.5-0.7m (not target). Some detections
              but huge detection RMSE (1.8-2.2m = false positives at distance).
            - Anchor dominates: 87-98% anchor rate. EKF tracks via paddle position.
            - Flight-conditional det rate: at target=1.00, 30 det / ~180 flight steps
              ≈ 17% per flight-step — reasonable for the camera FOV.
            - Core bottleneck is POLICY, not perception. The d435i Stage G checkpoint
              can balance but can't juggle reliably.
            - Policy agent currently training new Stage G run (nohup, PID 1348279,
              from 2026-04-08_22-51-56 checkpoint).
Decision:   Next iter: (1) run oracle comparison when GPU frees to confirm perception
            vs policy split, (2) update Quarto with these findings. The perception
            pipeline is essentially feature-complete — further progress depends on the
            policy agent producing a better juggling policy. Consider running the
            oracle-trained checkpoint (2026-04-08_19-19-41) through our pipeline to see
            if it juggles better (it was oracle-trained Stage F, should balance well).

## Iteration 117 — Quarto update + Stage G bottleneck experiment write-up  (2026-04-09T15:30:00Z)
Hypothesis: The iter 116 Stage G eval results deserve a proper experiment write-up
            and Quarto page update to document the perception/policy boundary finding.
Change:     1. Updated agents/perception.qmd with iter 115-116 findings (starvation
               override disproved, policy is the bottleneck, pipeline feature-complete).
            2. Created experiments/perception/2026-04-09_stage_g_bottleneck_analysis.qmd
               — full experiment write-up with method, results, discussion, reproducibility.
            3. Created scripts/perception/plot_stage_g_bottleneck.py — 2-panel figure
               (flight fraction vs det rate, peak height vs target with identity line).
            4. Generated images/perception/stage_g_bottleneck_iter116.png.
            5. Updated "Current focus" section on agent page to reflect feature-complete status.
Command:    python3 scripts/perception/plot_stage_g_bottleneck.py (no GPU needed)
Result:     Quarto content updated. Experiment write-up published. Figure clearly shows
            the two behavioral regimes (balance vs failed juggle) and that detection rate
            tracks flight fraction, not target height. GPU still occupied by policy Stage G
            training (PID 1348279).
Decision:   Next iter: check if policy Stage G training completed. If new checkpoint
            available, run oracle comparison. If GPU still blocked, consider:
            (1) running oracle-trained Stage F checkpoint through d435i pipeline, or
            (2) preparing real-hardware integration plan items (D435i wrapper, YOLO training data).

## Iteration 118 — Multi-target eval comparison tool  (2026-04-09T16:30:00Z)
Hypothesis: A multi-target comparison tool that processes eval directories will
            streamline oracle vs d435i analysis once GPU frees up.
Change:     Created compare_multi_target.py — takes two eval directories (each with
            target_X_XX/trajectory.npz subdirs), computes per-target metrics via
            existing analyze_eval_trajectory.py, and produces a 4-panel dashboard
            figure (det rate, EKF RMSE, flight fraction, peak height). Also supports
            single-directory mode for quick summaries. Added 8 tests.
Command:    pytest scripts/perception/test_compare_multi_target.py -x -q → 8/8 passed.
            pytest scripts/perception/ -x -q → 457/457 passed (9.95s). No GPU needed.
            Tested against existing Stage G data: eval_stage_g_d435i vs eval_stage_g_starve10.
Result:     Tool works. Dashboard figure clearly shows per-target comparison.
            Data confirms iter 116 finding: detection rates uniformly low (1-2%),
            policy balances rather than juggles at all targets. Peak height identical
            across targets (0.716m) — policy ignores target height command.
            GPU still blocked by policy Stage G training (PID 1348279, ~20 min in).
Decision:   Next iter: check if policy training completed. If new checkpoint available,
            run full oracle vs d435i comparison using the new tool. If GPU still blocked,
            consider updating Quarto with latest status or preparing the oracle eval
            command so it can launch immediately when GPU frees.

## Iteration 119 — Trajectory time-series visualization + oracle comparison tooling  (2026-04-09T17:30:00Z)
Hypothesis: Time-series visualization of ball height will clearly show the
            balancing-vs-juggling behavioral split, and preparing automated
            oracle-vs-d435i comparison tooling will enable immediate launch
            when GPU frees from policy training.
Change:     1. Created plot_trajectory_timeseries.py — multi-panel figure showing
               GT ball height, EKF estimate, detection events, and phase coloring
               per target height. Clear visual evidence of contact-dominated regime.
            2. Created run_oracle_vs_d435i.sh — one-command pipeline that runs oracle
               eval then d435i eval then generates comparison dashboard figure.
            3. Generated trajectory_timeseries_stage_g.png from existing iter 116 data.
            4. Updated agents/perception.qmd with new figure and iter 119 entry.
Command:    python plot_trajectory_timeseries.py --eval-dir eval_stage_g_d435i --out ... (CPU only)
            pytest scripts/perception/ -x -q → 457/457 passed (10.02s)
Result:     TIME SERIES FIGURE: clearly shows ball oscillates at 0.5-0.7m regardless
            of target height. Detections cluster in first ~1s (initial ball drop).
            After that, 98.5% contact phase — ball on paddle. Camera only active
            for 1.5% of steps. EKF tracks via anchor after initial flight window.
            GPU still blocked by policy Stage G training (PID 1348279, model_3200.pt,
            ~68 min remaining). Oracle/d435i checkpoints all verified available.
            READY TO LAUNCH: run_oracle_vs_d435i.sh with:
              --oracle-pi1 .../2026-04-08_19-19-41/model_best.pt
              --d435i-pi1  .../2026-04-09_07-38-27/model_best.pt (or latest Stage G)
Decision:   Next iter: check if GPU freed. If yes, launch oracle vs d435i comparison
            using run_oracle_vs_d435i.sh. If GPU still blocked, consider deeper
            analysis of detection timing (are detections useful even in the first 1s?)
            or preparing real-hardware integration specs.
