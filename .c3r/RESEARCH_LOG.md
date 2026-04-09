# RESEARCH_LOG.md

_(older entries auto-archived to RESEARCH_LOG_ARCHIVE.md at 2026-04-09 15:25 UTC)_

            d435i mode (lower threshold, or easier target distribution).

## Iteration 122 — Live training monitoring + TensorBoard plotting tool  (2026-04-09T21:00:00Z)
Hypothesis: Reading TensorBoard event files directly (vs text log parsing) will
            enable accurate live monitoring of the policy agent's active training.
Change:     1. Created plot_training_tb.py — reads TB event files directly, produces
               5-panel figure (timeout%, ep_length, apex_reward, noise_std, ES counter).
               Handles missing dirs/tags gracefully. 7 tests (485/485 total pass).
            2. Generated stage_g_training_progress_iter122.png from the active training run.
            3. Updated agents/perception.qmd with iter 122 entry + live training figure + table.
Command:    python plot_training_tb.py --logdir .../2026-04-09_07-38-27/ --out ... (CPU only)
            pytest scripts/perception/ -x -q → 485/485 passed (10.96s)
Result:     TRAINING PLATEAU CONFIRMED AND QUANTIFIED:
            - 776 iterations (steps 2764–3539), timeout 54.1% ± 0.2%
            - Zero iterations ever reached 80% threshold
            - ES no-improve counter at 745/1500 (49.7%) — will early stop at ~step 4294
            - At 3.3s/iter, ~42 min until ES triggers (PID 1348279 running 41 min)
            - Noise std stable at 0.2140, apex reward 3.618 ± 0.135
            CHECKPOINTS AVAILABLE for eval:
            - Oracle: .../2026-04-08_19-19-41/model_best.pt (Stage F oracle)
            - D435i:  .../2026-04-09_07-38-27/model_best.pt (Stage G d435i, 54% timeout)
            EVAL COMMAND READY:
            $C3R_BIN/gpu_lock.sh bash scripts/perception/run_oracle_vs_d435i.sh \
              --oracle-pi1 /home/daniel-grant/Research/QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_19-19-41/model_best.pt \
              --d435i-pi1 /home/daniel-grant/Research/QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier/2026-04-09_07-38-27/model_best.pt \
              --targets "0.10 0.30 0.50 0.70 1.00" --label stage_g_comparison --steps 1500 --num-envs 4
Decision:   Next iter: check if GPU freed (ES should trigger by ~step 4294).
            If free, launch oracle vs d435i comparison using the command above.
            If still blocked, continue monitoring or analyze perception noise
            model parameters for potential improvements.

## Iteration 123 — D435i noise profile analysis  (2026-04-09T21:30:00Z)
Hypothesis: The Stage G plateau (53.6% timeout) may be caused by excessive
            d435i noise at low target heights (0.10-0.20m) where SNR is lower.
Change:     Created plot_noise_profile.py — comprehensive D435i noise model
            visualizer (6 panels: position/velocity noise, dropout, SNR, per-target
            summary table). 12 new tests (497/497 total pass). Updated Quarto page.
Command:    python plot_noise_profile.py --out images/perception/d435i_noise_profile_iter123.png
            python plot_training_tb.py --logdir .../2026-04-09_07-38-27/ --out ...
            pytest scripts/perception/ -x -q → 497/497 passed (10.98s)
Result:     NOISE IS NOT THE BOTTLENECK:
            - Position noise: 1.01–1.31mm across all targets (negligible)
            - Velocity noise: 0.043–0.056 m/s (SNR > 30 everywhere)
            - Dropout: flat 20% (all heights < 0.5m threshold)
            - Policy reached 73% timeout at Stage F (single target, full noise)
            - Drop to 53.6% at Stage G is purely energy modulation (mixed targets)
            TRAINING STATUS (updated):
            - 892 iterations, timeout 53.6% ± 0.2% (unchanged from iter 122)
            - ES no-improve: 861/1500 (57.4%) — ~36 min to early stop
            - Noise std: 0.2123, apex reward: 3.511
Decision:   Next iter: GPU should be free (ES triggers ~step 4264). Launch oracle
            vs d435i comparison eval. No perception-side changes needed — the noise
            model is well-calibrated for the 0.10–0.50m range.

## Iteration 124 — D435i vs oracle training history dashboard  (2026-04-09T22:00:00Z)
Hypothesis: A multi-run training history visualization will reveal the full d435i
            training trajectory and help identify where improvement opportunities lie.
Change:     Created plot_training_history.py — reads TensorBoard events from 4 key runs
            (Oracle A→F, D435i E, D435i F, D435i G) and produces 4-panel dashboard
            (timeout%, episode length, apex reward, policy noise std).
Command:    python plot_training_history.py --out images/perception/training_history_iter124.png
            (CPU only; GPU blocked by policy Stage G training, PID 1348279)
Result:     TRAINING HISTORY CAPTURED:
            - Oracle (A→F): 1500 iters, step 2572, timeout 62.9% ± 0.3%
            - D435i Stage E: 1500 iters, step 1499, timeout 71.9% ± 0.5%
            - D435i Stage F: 1500 iters, step 3049, timeout 54.6% ± 0.3%
            - D435i Stage G: 1026 iters, step 3789, timeout 53.4% ± 0.2%
            Stage G ES counter at 955/1500 (63.7%) — ~30 min to early stop.
            D435i noise std stable at 0.21, oracle at 0.18.
            Key pattern: D435i achieves high apex reward but can't modulate energy
            for mixed targets. Timeout degrades at each harder stage (72→55→53%).
            Updated Quarto page with dashboard figure and summary table.
            All 497 tests still pass.
Decision:   Next iter: GPU should be free (ES ~step 4330). Launch oracle vs d435i
            comparison eval via run_oracle_vs_d435i.sh. Ready to execute immediately.

## Iteration 125 — Oracle vs D435i Stage G final comparison eval  (2026-04-09T23:10:00Z)
Hypothesis: Running the oracle (Stage F) and d435i (Stage G) checkpoints through
            matched perception evals will quantify the perception gap vs policy gap.
Change:     Waited for Stage G early stop (ES 1500/1500 at step 4294), then ran
            run_oracle_vs_d435i.sh with 5 targets (0.10-0.50m), 1500 steps, 4 envs.
            Updated Quarto page with comparison table and dashboard figure.
Command:    $C3R_BIN/gpu_lock.sh bash scripts/perception/run_oracle_vs_d435i.sh \
              --oracle-pi1 .../2026-04-08_19-19-41/model_best.pt \
              --d435i-pi1 .../2026-04-09_07-38-27/model_best.pt \
              --targets "0.10 0.20 0.30 0.42 0.50" --label stage_g_final \
              --steps 1500 --num-envs 4
Result:     STAGE G TRAINING COMPLETED: 1531 iters, 53.6% timeout, ES early-stopped.
            COMPARISON RESULTS:
            Oracle (Stage F): 0% timeout ALL targets, 0 bounces, 15-19% flight, 1.7% det
            D435i (Stage G):  50% timeout at 0.42/0.50m, 0.2-1.0 bounces, 19-39% flight
            EKF RMSE COMPARISON MISLEADING:
            - Oracle: 5.6-5.8m (no paddle anchor → EKF diverges in predict-only mode)
            - D435i: 74-442mm (paddle anchor keeps EKF bounded during contact)
            RMSE difference reflects pipeline config, NOT perception quality.
            POLICY PERFORMANCE:
            - D435i Stage G clearly outperforms Oracle Stage F on juggling metrics
            - But this is unfair: different training stages (G vs F)
            - Neither model truly juggles — ball rarely above 200mm for camera detection
            Figure: images/perception/oracle_vs_d435i_stage_g_final.png
            Eval data: logs/perception/eval_stage_g_final_{oracle,d435i}/
Decision:   Need oracle Stage G training for fair comparison (same curriculum).
            Will ping policy agent to request oracle Stage G run. Meanwhile, explore
            running oracle checkpoint through d435i pipeline (cross-eval: does the
            noise model degrade oracle performance?).

## Iteration 126 — Dual trajectory comparison analysis  (2026-04-09T23:45:00Z)
Hypothesis: Side-by-side trajectory visualization will reveal whether oracle and
            d435i models differ in ball dynamics (bouncing vs balancing pattern).
Change:     Created plot_dual_trajectory.py — side-by-side oracle vs d435i
            trajectory figure (rows=targets, cols=modes). 9 new tests, 506/506 pass.
            Generated dual_trajectory_iter126.png from Stage G eval data.
            Updated Quarto page with figure and summary table.
Command:    python plot_dual_trajectory.py --oracle-dir .../eval_stage_g_final_oracle
              --d435i-dir .../eval_stage_g_final_d435i --out .../dual_trajectory_iter126.png
            pytest scripts/perception/ -x -q → 506/506 passed (10.86s)
Result:     NEITHER MODEL JUGGLES — BALANCING ONLY:
            - Peak ball height: 0.246m across ALL targets (0.10–0.50m) and BOTH modes
            - Flight time: 2–9% (almost entirely contact phase)
            - Detection: 15–29 events per 1000 steps (during brief bounces)
            - D435i shows slightly more flight at 0.30–0.42m (noise dislodges ball)
            - Policy cannot modulate launch energy at all
            Policy agent's Stage G d435i training completed (early-stopped 53.6%).
            Policy agent now evaluating early_stop checkpoint (PID 1391572 on GPU).
Decision:   Perception pipeline is mature and waiting for policy improvement.
            Next: monitor policy agent's Stage G eval results. If policy still can't
            juggle, the gap is pure policy skill (energy modulation), not perception.
            May need to coordinate on reward structure changes to unlock juggling.
