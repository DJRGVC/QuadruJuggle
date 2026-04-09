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
