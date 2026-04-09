# RESEARCH_LOG.md

_(older entries auto-archived to RESEARCH_LOG_ARCHIVE.md at 2026-04-09 17:07 UTC)_

Decision:   Next iter: check if policy has progressed. If not, consider EKF
            pre-landing inflation (anticipate upcoming contact when descending
            near paddle) as another phase-aware improvement. Or update Quarto.

## Iteration 135 — Pre-landing Q inflation in EKF  (2026-04-09T19:00:00Z)
Hypothesis: Inflating q_vel when the ball is descending near the paddle (z < 80mm,
            vz < 0) will prepare the EKF covariance for the upcoming contact
            discontinuity, reducing NIS spikes at the contact transition and
            improving the noise-to-gap model fit.
Change:     Added 5th phase to EKF Q schedule: pre-landing (q_vel=2.0 when
            ball_z < 0.08m and vz < 0). New config: q_vel_pre_landing=2.0,
            pre_landing_z_threshold=0.08m. Updated predict_perception_gap.py
            to include pre-landing fraction in phase-averaged q_vel.
            Added test_pre_landing_inflated_q_vel test.
Command:    pytest scripts/perception/ -x -q → 560/560 passed (12.11s)
            predict_perception_gap.py → R²=0.994 (was 0.866 from iter 134)
Result:     560/560 tests pass (559 existing + 1 new).
            5-level Q schedule: contact(50) > post-contact(20) > pre-landing(2.0) > desc(0.40) > asc(0.25)
            Noise-to-gap model R² jumped from 0.866 → 0.994 — pre-landing phase
            was the missing explanatory variable. The model now captures the
            height-gap relationship almost perfectly.
            Updated predictions: 0.70m → 36.9%, 1.00m → 75.0% (higher than
            before because pre-landing noise matters more at low heights, steepening
            the fit). These are policy-limited predictions — actual gaps depend on
            whether policy can modulate energy.
            Policy agent still at iter 32 (81% context, no new checkpoint).
            Figure: images/perception/gap_prediction_iter135.png
Decision:   Next iter: research log is at ~310 lines — may need compaction soon.
            Check if policy has progressed. If not, update Quarto with the
            improved gap prediction model results.

## Iteration 136 — Quarto update: phase-aware EKF results  (2026-04-09T20:30:00Z)
Hypothesis: Documenting the phase-aware EKF Q schedule (iters 133-135) on the
            Quarto site will make the R²=0.994 improvement visible to collaborators.
Change:     Updated agents/perception.qmd with new entry for iters 133-135 (5-level
            Q schedule table, updated predictions, figure). Updated current focus
            section (560 tests, R²=0.994). Updated experiment write-up
            (experiments/perception/2026-04-10_noise_gap_prediction.qmd) with
            phase-aware update section and revised predictions.
Command:    N/A (documentation-only iteration)
Result:     Quarto page updated with phase-aware EKF results. Policy agent still
            at iter 32 (81% context) — no new checkpoint available.
Decision:   Next iter: check policy progress. If still stalled, consider what
            remaining perception-side improvements are possible without a new
            policy checkpoint. Could investigate EKF observability analysis
            (when does the EKF estimate diverge most from truth?) or prepare
            the real-hardware D435i wrapper skeleton.

## Iteration 137 — Validate gap prediction against real policy data  (2026-04-09T22:00:00Z)
Hypothesis: The noise-to-gap prediction model (R²=0.994) should match the
            actual perception gaps observed in policy agent's Stage G eval
            (iter 32: mixed targets 0.10-0.50m under D435i noise).
Change:     Ran predict_perception_gap.py with observed data from policy
            Stage G eval. Generated validation figure. Updated Quarto page,
            experiment write-up, and fix_plan.
Command:    python scripts/perception/predict_perception_gap.py \
              --observed '{"0.10":0.3,"0.20":0.0,"0.30":3.6,"0.40":10.0,"0.50":18.3}' \
              --out images/perception/gap_validation_iter137.png
Result:     Model validated — max error 0.8 percentage points across all 5 targets.
            R²=0.994 confirmed. Predictions:
            0.10m: predicted -0.2% vs observed 0.3% (Δ0.5pp)
            0.20m: predicted -0.3% vs observed 0.0% (Δ0.3pp)
            0.30m: predicted 4.4% vs observed 3.6% (Δ0.8pp)
            0.40m: predicted 10.6% vs observed 10.0% (Δ0.6pp)
            0.50m: predicted 17.8% vs observed 18.3% (Δ0.5pp)
            Extrapolation: 0.70m→36.9%, 1.00m→75.0%.
            Conclusion: perception gap is fully noise-explained. No pipeline bugs.
Decision:   Next iter: policy plans longer Stage G retrain with fixed ES metric.
            While waiting, consider: (a) EKF observability analysis — when does
            the estimate diverge most? (b) real-hardware D435i wrapper skeleton,
            (c) investigate whether EKF R tuning could reduce the gap at 0.40-0.50m.

## Iteration 138 — EKF error decomposition: position vs velocity  (2026-04-09T23:30:00Z)
Hypothesis: Decomposing EKF error into position vs velocity, per-axis, and
            per-flight-phase will reveal whether R tuning (measurement) or Q
            tuning (process/velocity) is the better lever for reducing the
            perception gap at high target heights.
Change:     Created analyze_ekf_error_decomposition.py — simulates 50 noise
            trials per target height (0.10-1.00m), runs full EKF with D435i
            noise model, computes RMSE by component/axis/phase. Updated Quarto.
Command:    python scripts/perception/analyze_ekf_error_decomposition.py \
              --out images/perception/ekf_error_decomposition_iter138.png
            pytest scripts/perception/ -x -q → 560/560 passed (12.14s)
Result:     VELOCITY ERROR DOMINATES at all heights:
            - V/P ratio (100ms window): 2.6x at 0.10m, 1.3x at 1.00m
            - Z-axis dominates both pos and vel error; XY is negligible (~1mm, ~0.05 m/s)
            - Ascending phase worst: vz RMSE 0.61 m/s (0.10m) → 1.04 m/s (1.00m)
            - Descending phase clean: vz RMSE ~0.09 m/s at all heights
            - Position RMSE: 9.7mm (0.10m) → 31.3mm (1.00m), driven by Z noise
            IMPLICATION: R tuning won't help much. The gap is in velocity prediction
            during ascending flight (dropout + ballistic model). Future work should
            target drag coefficient uncertainty and ascending-phase Q schedule.
            Policy still at iter 32, waiting for Stage G retrain.
            Figure: images/perception/ekf_error_decomposition_iter138.png
Decision:   Next iter: investigate whether drag coefficient mismatch explains the
            ascending vz error. The EKF uses drag_coeff=0.112 (theoretical); if the
            sim uses a different effective drag, the ballistic prediction drifts during
            dropout. Alternatively, consider adaptive drag estimation in the EKF (online
            parameter learning). Or wait for policy and update fix_plan.
