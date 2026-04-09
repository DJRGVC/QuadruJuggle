# RESEARCH_LOG.md

_(older entries auto-archived to RESEARCH_LOG_ARCHIVE.md at 2026-04-09 07:23 UTC)_

            Continuation run in progress: step ~32/1500, ETA 85 min
            Key checkpoints:
            - D435i: logs/.../2026-04-08_21-16-05/model_best.pt (Stage E)
            - Oracle: logs/.../2026-04-08_19-19-41/model_best.pt (Stage F)
            - Pi2: .../2026-03-12_17-16-01/model_best.pt
Decision:   Next iteration: check continuation run progress. If Stage F reached, run
            formal eval with eval_juggle_hier.py for publication-quality comparison table.
            Also need to update fix_plan.md to mark d435i training as done.

## Iteration 29 — Fix eval partial-episode bug + corrected cross-eval  (2026-04-09T08:00Z)
Hypothesis: The iter 28 cross-eval metrics (mean_len=111, timeout=70% for d435i) were
            internally inconsistent (86.7% timeout but 92 mean steps — impossible if
            timeout means 1500-step episodes). Suspected eval_juggle_hier.py bug.
Change:     Fixed eval_juggle_hier.py: after flush loop, step_counts=0 but env's internal
            episode_length_buf was mid-episode. Envs that reset early in flush were deep
            into their next episode when collection started, so first recorded episodes
            had tiny step_counts with stale timeout flags. Fix: track `fresh_env` flag,
            only record episodes that completed a full cycle within the collection loop.
Command:    gpu_lock.sh bash scripts/rsl_rl/cross_eval_fixed.sh (4 combos × 5 targets × 30 eps)
Result:     CORRECTED cross-eval results:
            Oracle→Oracle: 100% timeout, 1500 steps at all targets (perfect stability)
            Oracle→D435i:  60-94% timeout, 1110-1451 mean_len (moderate noise degradation)
            D435i→D435i:   0-73% timeout, 310-1219 mean_len (overshoots easy targets)
            D435i→Oracle:  0-83% timeout, 442-1346 mean_len (better than d435i obs)

            Per-target breakdown (d435i→d435i):
            target=0.10: timeout=0%,   apex=10.33, len=310  — massive overshoot
            target=0.20: timeout=0%,   apex=6.49,  len=439  — still overshoots
            target=0.30: timeout=40%,  apex=4.28,  len=886  — starting to match
            target=0.42: timeout=73%,  apex=2.92,  len=1187 — good performance
            target=0.50: timeout=70%,  apex=2.43,  len=1219 — good performance

            KEY REVISION: d435i model is MUCH better than iter 28 reported.
            Main issue is target overshoot at easy targets (0.10-0.20m), not
            general instability. At harder targets the model performs well.
            Results dir: experiments/iter_029_cross_eval_fixed/
Decision:   The d435i model is viable but overshoots at easy targets. Two possible fixes:
            (1) Retrain with ball_release_vel weight reduced (currently 8.0, very high)
            (2) The overshoot may be a natural consequence of training to Stage F
            (target=0.80) — the policy learned high-energy launches that are overkill
            for low targets. This might be acceptable if we only care about hard targets.
            Next: investigate whether the overshoot matters for the real task. If so,
            reduce ball_release_vel weight from 8.0 → 3.0 and retrain.

## Iteration 28 — Cross-eval: d435i vs oracle under swapped obs  (2026-04-09T07:40Z)
Hypothesis: Cross-evaluating d435i-trained and oracle-trained checkpoints under both noise
            modes will reveal whether noise training creates noise-robust or noise-dependent
            policies, and quantify the sim-to-real perception gap.
Change:     Ran eval_juggle_hier.py in 4 combinations (2 checkpoints × 2 noise modes),
            5 target heights (0.10-0.50m), 30 episodes each. Created cross_eval.sh and
            parse_cross_eval.py scripts.
Command:    gpu_lock.sh bash scripts/rsl_rl/cross_eval.sh
Result:     Oracle→Oracle: timeout=100%, apex=3.68, len=1496 (never drops ball)
            Oracle→D435i:  timeout=81%,  apex=3.58, len=476  (drops more, apex same)
            D435i→D435i:   timeout=70%,  apex=5.65, len=111  (high apex, frequent drops)
            D435i→Oracle:  timeout=79%,  apex=5.74, len=127  (similar to d435i→d435i)
            KEY FINDINGS:
            1. D435i training creates a MORE AGGRESSIVE strategy: +53% apex reward per step
               but episodes are 13× shorter (111 vs 1496 steps).
            2. Oracle model is noise-robust on APEX (-2.7%) but not on SURVIVAL (100%→81%).
            3. D435i model transfers cleanly to oracle obs (+1.5% apex) — not noise-dependent.
            4. The d435i model's short episodes (70% timeout) suggest it learned to throw
               hard but can't consistently catch — a catch reward or survival pressure needed.
            Results: experiments/iter_028_cross_eval/
Decision:   The d435i model juggles well (high apex) but drops frequently. Next step:
            investigate if increasing survival pressure (alive weight or longer curriculum
            sustain) can keep the high apex while extending episodes. This is a reward
            shaping problem — the d435i model should be retrained with stronger ball_on_paddle
            weight to incentivize catch-and-sustain rather than launch-and-drop.

## Iteration 27 — D435i reaches Stage F: full comparison  (2026-04-09T07:15Z)
Hypothesis: The d435i continuation run (from model_1499.pt, start-stage 4) will reach
            Stage F and provide a controlled comparison against oracle at matched difficulty.
Change:     Analysis iteration — no code changes. Updated Quarto page + experiment write-up
            with Stage F results. Created comparison figure. Processed 4 INBOX messages.
Command:    TensorBoard analysis of continuation run (2026-04-08_22-51-56, 3049 steps total).
Result:     **D435i reached Stage F (final stage, 6/6).** Completed 1500 continuation iters.
            Stage F comparison (avg last 50 iters):
              Oracle: reward=40.3, apex=0.88, timeout=62.9%, ball_below=37.0%, release_vel=0.54
              D435i:  reward=38.7, apex=1.40, timeout=54.6%, ball_below=45.4%, release_vel=0.43
            D435i OUTPERFORMS oracle on apex height: +59% at Stage F.
            Trade-off: more drops (45% vs 37%) but better target accuracy when juggling.
            ball_low=-0.011 in both = active juggling confirmed.
            Checkpoints:
            - D435i (Stage F): logs/.../2026-04-08_22-51-56/model_best.pt
            - Oracle (Stage F): logs/.../2026-04-08_19-19-41/model_best.pt
            Figure: images/policy/d435i_vs_oracle_stage_f.png
            GPU occupied by perception demo (PID 1154762); video capture deferred.
Decision:   Next iteration: (1) capture play.py video of d435i checkpoint for Quarto
            (Daniel requested), (2) cross-eval: run noise-trained checkpoint with oracle obs
            and vice versa to measure noise robustness vs overfitting. Both need GPU.

## Iteration 30 — Cross-eval analysis + Stage G preparation  (2026-04-09T10:20Z)
Hypothesis: The d435i target overshoot at easy targets (0.10-0.20m) is caused by the model
            never training on low targets under noise — it only saw target=0.50 in Stages E-F.
            Stage G (mixed targets 0.10-0.50) should fix energy modulation.
Change:     Analysis iteration (GPU occupied by perception). Created cross-eval comparison
            figure. Updated Quarto page with corrected results table and figure. Prepared
            Stage G training command in fix_plan.md.
Command:    python scripts/plot_cross_eval.py (figure generation, no GPU)
Result:     CONFIRMED: d435i apex reward HIGHER than oracle at ALL targets (10.33 vs 7.79
            at 0.10m, 6.49 vs 4.81 at 0.20m, etc.) — the model throws too hard.
            At target=0.42-0.50m, d435i achieves 70-73% timeout (reasonable).
            At target=0.10-0.20m, d435i achieves 0% timeout (catastrophic overshoot).
            Root cause: d435i training skipped Stage G (mixed targets), only trained
            at target=0.50 under noise.
            Figure: images/policy/cross_eval_iter029.png
Decision:   Next iteration: launch Stage G continuation training from d435i model_best.pt
            (start-stage 6, noise-mode d435i). This is the highest-priority item.
            GPU was blocked this entire iteration by perception agent's demo runs.
