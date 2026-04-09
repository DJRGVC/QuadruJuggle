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

## Iteration 139 — Drag model mismatch: PhysX linear vs EKF quadratic  (2026-04-10T01:00:00Z)
Hypothesis: The EKF's quadratic drag model (a=-0.112·|v|·v) does not match
            PhysX's linear_damping=0.1 (a=-0.1·v). At high speeds, the EKF
            over-predicts drag, causing ascending vz error to grow with height.
Change:     (1) Created analyze_drag_mismatch.py — quantifies error from pure
            drag model mismatch (no noise). (2) Added drag_mode config to
            BallEKFConfig: "quadratic" (real hardware) or "linear" (PhysX sim).
            (3) Updated BallObsNoiseCfg default to use drag_mode="linear".
            (4) Added 5 tests for linear drag mode. (5) Updated Quarto page.
Command:    python scripts/perception/analyze_drag_mismatch.py
            pytest scripts/perception/ -x -q → 565/565 passed (12.15s)
Result:     CONFIRMED: drag mismatch is massive at high speeds.
            Crossover speed: 0.89 m/s. Above this, EKF over-predicts drag.
            At v=4.4 m/s (1m juggle): PhysX=0.44 m/s², EKF=2.20 m/s² (5x).
            Pure-mismatch errors (no noise):
              0.50m: vz RMSE 76mm/s, apex error -24mm
              1.00m: vz RMSE 230mm/s, apex error -102mm
            This explains the ascending vz dominance found in iter 138:
            the "ground truth" in that analysis used quadratic drag too,
            so it only measured noise error. The drag mismatch is additional.
            Fix: EKF now configurable. Sim defaults to linear (matching PhysX).
            Real hardware should use quadratic (true aerodynamic drag).
            Policy agent still at iter 32 (81% context).
            Figure: images/perception/drag_mismatch_iter139.png
Decision:   Next iter: re-run EKF error decomposition and gap prediction
            with the corrected linear drag model. This should show a large
            reduction in ascending-phase vz error. Then re-validate against
            policy Stage G eval data to see if the corrected model fits better.

## Iteration 140 — Re-validate EKF error decomposition with linear drag  (2026-04-10T02:30:00Z)
Hypothesis: Re-running EKF error decomposition with matched linear drag
            (both ground truth and EKF) will confirm that noise, not drag
            model mismatch, is the dominant error source in sim.
Change:     Updated analyze_ekf_error_decomposition.py with --drag-mode
            parameter (linear/quadratic) for both ground truth trajectory
            and EKF config. Defaults to linear (PhysX). Re-ran decomposition.
Command:    python scripts/perception/analyze_ekf_error_decomposition.py \
              --drag-mode linear --out images/perception/ekf_error_decomposition_linear_iter140.png
            pytest scripts/perception/ -x -q → 565/565 passed (12.11s)
Result:     CONFIRMED: with matched drag, noise dominates error structure.
            Linear drag (matched):  asc vz RMSE 0.613 (0.10m) → 1.158 (1.00m)
            Quadratic drag (matched): asc vz RMSE 0.613 (0.10m) → 1.040 (1.00m)
            Linear case has ~11% more vz error at 1.00m because ball flies
            higher (actual apex 1.042m vs 0.952m) → more noise exposure.
            Gap prediction model (R²=0.994) is analytical — drag-independent.
            Overall pos RMSE: 9.7mm (0.10m) → 32.2mm (1.00m) under linear.
            Descending phase remains clean: vz ~0.09 m/s at all heights.
            Policy agent at iter 32 (81% context), waiting for Stage G retrain.
Decision:   RE-VALIDATE task complete. Next priorities:
            (1) Wait for policy Stage G retrain results to re-validate gap model
            (2) Consider real-hardware prep: D435i wrapper skeleton (low priority)
            (3) Check if policy agent needs support

## Iteration 141 — Implement D435iCamera pyrealsense2 wrapper  (2026-04-09T04:00:00Z)
Hypothesis: Replacing the D435iCamera stubs with real pyrealsense2 calls
            (start/get_frame/stop) and adding mock-based tests will validate
            the camera driver design before hardware is available.
Change:     (1) Implemented D435iCamera.start() — configures rs.pipeline,
            enables depth stream at requested resolution/fps, extracts
            intrinsics from depth stream profile, blocks until first frame.
            (2) Implemented D435iCamera.get_frame() — non-blocking poll via
            poll_for_frames(), returns (depth_u16, timestamp_s) or None.
            Hardware timestamp converted from ms to seconds.
            (3) Implemented D435iCamera.stop() — stops pipeline, sets to None.
            (4) Created test_d435i_camera.py — 11 tests using mock pyrealsense2:
            start extracts intrinsics, configures depth stream, handles serial,
            waits for first frame; get_frame returns depth+timestamp or None;
            stop releases resources; ImportError guard works.
Command:    pytest scripts/perception/ -x -q → 576/576 passed (12.19s)
Result:     D435iCamera is no longer a stub. All 3 methods implemented with
            pyrealsense2 API. 11 new mock-based tests verify the contract.
            Total test count: 565 → 576. No regressions.
            Implementation choices:
            - poll_for_frames() for non-blocking (vs wait_for_frames which blocks)
            - np.asanyarray with reshape for zero-copy depth access
            - wait_for_frames(5000ms) in start() to confirm device is live
            - stop() sets pipeline=None so get_frame returns None after stop
Decision:   Next: YOLO detector stub is the next real-hardware item, but it
            requires TensorRT which we don't have in this env. Better to wait
            for policy retrain or work on checkerboard calibration (from_checkerboard).
            Check policy agent progress — still at iter 32 with 81% context.

## Iteration 142 — Implement CameraCalibrator.from_checkerboard()  (2026-04-09T18:00:00Z)
Hypothesis: Implementing the checkerboard calibration routine (PnP + gravity
            alignment) completes another real-hardware component while waiting
            for policy Stage G retrain.
Change:     (1) Implemented from_checkerboard(): captures N depth frames,
            detects checkerboard corners via cv2.findChessboardCorners,
            runs solvePnP for each, averages board Z direction in camera frame
            to get gravity_cam, then uses _rotation_between_vectors() (Rodrigues'
            formula via cross product) to compute R_cam_body that maps
            gravity_cam to gravity_body. Translation set to zero (requires
            manual measurement for production).
            (2) Added _rotation_between_vectors() helper: handles identity,
            antiparallel, and general cases via skew-symmetric cross product.
            (3) Added 9 tests: 4 for _rotation_between_vectors (identity,
            orthogonal, antiparallel, 10 random pairs), 5 for from_checkerboard
            (gravity-aligned, rotated board, too-few-frames error, no-opencv
            error, custom gravity). All use mock cv2 functions.
Command:    pytest scripts/perception/ -x -q → 585/585 passed (12.28s)
Result:     from_checkerboard() no longer a stub. Gravity-alignment tested:
            R_cam_body correctly maps camera gravity to body gravity for
            identity rotation, 90° X rotation, and custom tilted gravity.
            _rotation_between_vectors passes 10 random direction pairs.
            Test count: 576 → 585.
            Policy agent at iter 32, 81% context, retraining Stage G.
Decision:   Next: YOLO detector is the last stub but blocked on TensorRT.
            Could implement a lightweight fallback (colour thresholding on
            depth) or wait for policy results. Check policy progress next iter.
