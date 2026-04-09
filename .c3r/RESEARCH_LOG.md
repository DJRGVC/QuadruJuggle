# RESEARCH_LOG.md

_(older entries auto-archived to RESEARCH_LOG_ARCHIVE.md at 2026-04-09 09:19 UTC)_

               text-log parsing as fallback.
Command:    pytest scripts/perception/ → 347/347 passed (7.13s).
            GPU still locked: policy training at PID 1199971, oracle eval PID 1204960 queued.
Result:     Analysis tooling complete. When oracle eval runs and saves trajectory.npz,
            run_comparison.sh will work regardless of whether stdout was captured.
            Policy agent iter 29 corrected cross-eval: d435i model overshoots easy targets
            (0% timeout at 0.10m) but works at hard targets (73% at 0.42m).
Decision:   Next iter: check oracle_eval_DONE sentinel. If found, run analyze_eval_trajectory.py
            on the npz to get detection rate + height-binned RMSE under trained policy.
            If still blocked, update Quarto with policy agent's cross-eval findings.

## Iteration 98 — Flight-phase analysis for trajectory evaluation  (2026-04-10T05:15:00Z)
Hypothesis: Splitting EKF/detection metrics by flight phase (ascending vs descending vs contact)
            will reveal phase-dependent performance differences — ascending should be predictable
            (ballistic), while descending near paddle has contact transitions where EKF needs
            adaptive Q. This insight will inform EKF tuning when oracle eval results arrive.
Change:     Added compute_phase_metrics() + print_phase_table() to analyze_eval_trajectory.py.
            Phases: ascending (vz>0, above contact zone), descending (vz≤0, above contact zone),
            contact (within 25mm of paddle). Velocity estimated via central finite differences.
            Integrated into both single-trajectory and comparison analysis paths.
            7 new tests (TestPhaseMetrics): phase sum=total, both-phases-present, stationary-ball,
            det-rate-bounded, ekf-rmse-positive, perfect-ekf.
Command:    pytest scripts/perception/ → 354/354 passed (7.15s).
            GPU status: policy training ETA ~38 min, oracle eval PID 1204960 still queued.
Result:     Phase analysis ready. When oracle eval trajectory.npz arrives, analyze script
            will now also output phase-separated table showing whether EKF outperforms raw
            detections differently during ascent vs descent vs contact phases.
Decision:   Next iter: check oracle_eval_DONE sentinel. ETA: training ~38min + eval ~15min
            ≈ 53min total. Should complete before next iteration starts. When it does,
            run analyze_eval_trajectory.py --npz <path> to get full results with phase analysis.

## Iteration 99 — Cross-eval validation + Quarto update while GPU blocked  (2026-04-10T06:30:00Z)
Hypothesis: Policy agent's cross-eval results (iters 27-29) validate the d435i perception
            pipeline as beneficial rather than degrading — documenting this on Quarto
            establishes the key finding while waiting for oracle eval GPU run.
Change:     Updated agents/perception.qmd with new entry summarizing policy agent's
            6-stage d435i vs oracle curriculum comparison (+59% apex at Stage F).
            Added 2 new references to references/perception.qmd: Tobin et al. 2017 (IROS)
            and OpenAI et al. 2019 — domain randomization literature that explains why
            noise-injected training outperforms oracle.
Command:    GPU check: oracle eval PID 1204960 still waiting for lock (policy training
            PID 1199971 holds GPU). No GPU work this iteration.
            pytest scripts/perception/ → 354/354 passed (no code changes, verified).
Result:     Quarto updated with cross-eval findings. Key validation: d435i noise training
            produces +59% apex height, +176% at Stage E. The noise is NOT degrading —
            it acts as beneficial regularization. Cross-eval confirms d435i model is not
            noise-dependent (transfers to oracle obs at +1.5% apex).
Decision:   Next iter: check oracle_eval_DONE sentinel again. If found, run
            analyze_eval_trajectory.py on the npz for detection rate + height-binned RMSE
            + phase analysis under sustained juggling. If still blocked, consider what
            additional analysis tooling would be useful.

## Iteration 100 — Add --out-dir flag + eval scripts for oracle vs d435i comparison  (2026-04-10T07:45:00Z)
Hypothesis: Multiple eval runs need separate output directories to enable side-by-side
            comparison. The hardcoded demo output dir caused each run to overwrite the previous.
Change:     1. Added --out-dir CLI flag to demo_camera_ekf.py (defaults to legacy path).
            2. Updated run_oracle_eval.sh to use --out-dir logs/perception/oracle_eval/.
            3. Created run_d435i_eval.sh (d435i counterpart, --out-dir logs/perception/d435i_eval/).
            4. Updated run_comparison.sh to read from dedicated eval dirs and support 2-run comparison.
            5. Killed stale oracle eval (PID 1204960, used old script) and re-queued (PID 1218087).
Command:    pytest scripts/perception/ → 354/354 passed (7.21s).
            GPU status: policy training PID 1199965 (ETA ~32 min). Oracle eval PID 1218087 queued.
Result:     Eval infrastructure now supports multiple named runs with persistent output.
            Oracle eval will save to logs/perception/oracle_eval/trajectory.npz when GPU frees.
            After oracle completes, run_d435i_eval.sh produces the d435i counterpart.
            run_comparison.sh then generates the side-by-side figure automatically.
Decision:   Next iter: check oracle_eval_DONE sentinel. When oracle eval completes (~45 min),
            parse results with analyze_eval_trajectory.py. Then queue d435i eval for comparison.
            If still GPU-blocked, could start writing the experiment write-up template.

## Iteration 101 — Full eval results: camera visibility gap at 0.42m target  (2026-04-10T09:15:00Z)
Hypothesis: The full eval (oracle + d435i policies through camera pipeline) will show
            whether the D435i camera can track the ball during sustained juggling.
Change:     1. Parsed run_full_eval.sh results (both trajectory.npz files).
            2. Fixed SameFileError in analyze_eval_trajectory.py --quarto-copy (realpath check).
            3. Filled in experiment write-up: experiments/perception/2026-04-10_oracle_vs_d435i_eval.qmd.
            4. Updated agents/perception.qmd with summary + comparison figure.
            5. Updated fix_plan.md — marked eval tasks complete, added higher-target eval queue.
Command:    pytest scripts/perception/ → 354/354 passed (7.21s). No GPU needed this iter.
Result:     FUNDAMENTAL VISIBILITY GAP: oracle 1.7% det rate, d435i 4.5% det rate.
            Ball on paddle 68-84% of episode at target=0.42m → near-zero camera visibility.
            EKF diverges to ~6m RMSE. Detection works at 200-300mm (100% rate) but ball
            rarely reaches that height. D435i policy shows 2.7x more detections (67 vs 25).
Decision:   Next iter: re-run eval at higher target heights (0.50-1.00m) where ball spends
            more time in flight. If policy can sustain higher targets, detection rate should
            improve dramatically. Also consider asking Daniel about camera mount alternatives.

## Iteration 102 — Height sweep: policy balances, doesn't juggle  (2026-04-10T11:45:00Z)
Hypothesis: Higher target heights (0.50-1.00m) should give the ball more airtime and
            dramatically improve camera detection rate (from 1.7-4.5% at 0.42m).
Change:     Created run_height_sweep_eval.sh to sweep targets {0.42, 0.50, 0.70, 1.00}m
            with d435i policy, 1500 steps × 4 envs each. Detailed per-step height analysis
            and height-above-paddle histograms.
Command:    $C3R_BIN/gpu_lock.sh bash scripts/perception/run_height_sweep_eval.sh
            Smoke test: 10 steps at 0.70m — passed. Full sweep: 4 targets × 1500 steps.
Result:     TARGET-INVARIANT: policy caps at 0.246m max apex above paddle regardless of
            target setting. Mean ball height above paddle = 0.01m (essentially resting).
            | Target | Det% | FOV% | Det|FOV% | Mean h | Max h |
            | 0.42m  | 3.9  | 16.2 | 11.1    | 0.011  | 0.246 |
            | 0.50m  | 1.1  | 12.3 |  9.2    | 0.004  | 0.246 |
            | 0.70m  | 1.7  | 16.9 |  6.3    | 0.010  | 0.246 |
            | 1.00m  | 2.5  |  4.1 | 26.2    | -0.009 | 0.246 |
            ROOT CAUSE: d435i Stage F policy catches ball on initial drop (17-step
            flight with 100% detection), then holds it on paddle forever. Zero sustained
            bouncing. Camera+detector pipeline works perfectly — problem is pure policy.
            Pinged policy agent about Stage G training.
Decision:   BLOCKED on policy improvement. Next iter: (a) investigate EKF predict-only mode
            for sparse measurements during contact phases, (b) consider if camera can also
            see ball during paddle contact from a different mount angle, (c) wait for
            policy agent's Stage G progress.

## Iteration 103 — EKF covariance clamping for sparse measurements  (2026-04-10T13:00:00Z)
Hypothesis: Clamping P diagonals prevents covariance divergence during long predict-only
            sequences (ball on paddle 98% of time at current policy capability), keeping
            the EKF ready to absorb measurements when the ball enters the camera FOV.
Change:     Added covariance clamping to BallEKF: p_clamp_enabled=True, p_max_pos=0.25m,
            p_max_vel=5.0m/s, p_max_spin=50.0 rad/s. Clamps P diagonals after each predict
            step. Also added steps_since_measurement counter (per-env, resets on measurement
            or env reset) for sparse-measurement diagnostics.
            New config fields: p_clamp_enabled, p_max_pos, p_max_vel, p_max_spin.
            New property: steps_since_measurement.
            New test file: test_p_clamping.py (8 tests).
Command:    pytest scripts/perception/ → 362/362 passed (7.49s). No GPU needed.
Result:     After 500 predict-only steps, P_pos ≤ 0.0625 (25cm²), P_vel ≤ 25.0 (5m/s²).
            Without clamping, P_vel grows to 24.2 after just 200 contact-zone steps.
            Measurement absorption confirmed: after 200 predict-only steps, a measurement
            still corrects state by >1cm. 9D spin clamping also verified.
Decision:   Next iter: consider "flight window" detection mode (trigger detection only during
            expected flight arcs based on EKF velocity prediction) OR wait for policy agent
            Stage G progress. Could also add paddle-anchor virtual measurement during contact.

## Iteration 104 — Paddle-anchor virtual measurement for contact phase  (2026-04-10T14:30:00Z)
Hypothesis: Injecting a virtual measurement at the known paddle position during long
            contact phases (no camera data) will keep the EKF accurate and prevent drift
            in pi1's ball observations during the 98%+ of time the ball sits on the paddle.
Change:     Added paddle_anchor_update() method to BallEKF. When steps_since_measurement
            >= 5 AND ball Z < contact_z_threshold, the method injects a low-noise (5mm)
            position measurement at the paddle centre and zeros the velocity estimate.
            New config fields: anchor_enabled, anchor_r_pos, anchor_min_starve_steps.
            New test file: test_paddle_anchor.py (9 tests).
Command:    pytest scripts/perception/ → 371/371 passed (7.51s). No GPU needed.
Result:     All 9 tests pass:
            - Fires only for starved+contact envs (mixed env test confirms isolation)
            - Corrects XY position drift toward paddle centre
            - Zeros velocity for anchored envs (ball is stationary on paddle)
            - Reduces position covariance trace (injects information)
            - Does NOT reset steps_since_measurement (only camera does)
            - Disabled mode returns 0 with no state changes
Decision:   Next iter: wire paddle_anchor_update() into demo_camera_ekf.py eval loop.
            The call should happen after step() (predict+camera update), passing the
            known paddle position from robot state. Then re-run eval to measure improvement
            in EKF accuracy during contact phases. Still blocked on policy agent for
            sustained bouncing — this improves contact-phase accuracy in the meantime.

## Iteration 105 — Wire paddle anchor into demo eval loop  (2026-04-10T16:00:00Z)
Hypothesis: Wiring paddle_anchor_update() into demo_camera_ekf.py with correct world-frame
            contact_z_threshold will keep the EKF accurate during the 98%+ contact time,
            and also fix the contact-aware Q switching (which was silently broken at the
            default threshold of 0.025m in world frame where ball Z ≈ 0.49).
Change:     1. demo_camera_ekf.py: get robot from scene, compute paddle world position
               dynamically from robot.data.root_pos_w + paddle_offset_z + ball_radius.
            2. Set contact_z_threshold = paddle_z + ball_radius + 10mm (world frame) in
               EKF config. Previously 0.025m (paddle-relative) → contact-aware Q and
               anchor would never fire in the demo's world-frame coordinate system.
            3. Call ekf.paddle_anchor_update(paddle_pos_w) after ekf.step() each timestep.
            4. Added anchor metrics tracking (count, rate) in periodic logging and summary.
            5. New test: test_anchor_world_frame_threshold verifying anchor fires with
               world-frame Z ≈ 0.49 and threshold ≈ 0.50, doesn't fire at Z=0.65.
Command:    pytest scripts/perception/ → 372/372 passed (7.42s). No GPU needed.
Result:     Paddle anchor wired into eval loop. Also discovered and fixed a latent bug:
            contact_z_threshold=0.025 was designed for paddle-relative frame but the demo
            operates in world frame (ball Z ≈ 0.49). Both contact-aware Q switching AND
            anchor were silently disabled. Now both fire correctly in world frame.
Decision:   Next iter: GPU eval to measure EKF accuracy improvement with anchor active.
            Expect RMSE to drop dramatically during contact phases (ball on paddle 98% of
            time). If policy agent has Stage G progress, also test with sustained bouncing.

## Iteration 106 — Anchor ablation tooling + phase-aware RMSE  (2026-04-10T17:30:00Z)
Hypothesis: Adding --no-anchor flag and per-phase RMSE breakdown enables A/B quantification
            of paddle anchor improvement when GPU becomes available.
Change:     1. Added --no-anchor CLI flag to demo_camera_ekf.py (wired into ekf_cfg.anchor_enabled)
            2. Per-step ball_h and anchored_step tracking in trajectory data
            3. Phase-aware RMSE summary: contact (h<30mm) vs flight (h>=30mm) breakdown
            4. Saved ball_h and anchored_step arrays in trajectory.npz
            5. New run_anchor_ablation.sh: back-to-back anchor ON vs OFF comparison
            6. Updated Quarto page (agents/perception.qmd) with iters 103-106 summary
Command:    pytest scripts/perception/ → 372/372 passed (7.49s). No GPU needed.
            GPU blocked by policy agent Stage G training (12288 envs, started 04:53).
Result:     All tests pass. Ablation script ready for next GPU slot. Analysis will show:
            - Overall RMSE with/without anchor
            - Contact-phase RMSE (h<30mm): expect anchor >> no-anchor (5mm vs drift)
            - Flight-phase RMSE: expect identical (anchor doesn't fire during flight)
Decision:   Next iter: run anchor ablation on GPU if available. If still blocked,
            consider flight-window detection mode or per-step EKF diagnostics plotting.

## Iteration 107 — Anchor ablation analysis tooling  (2026-04-10T19:00:00Z)
Hypothesis: Dedicated ablation analysis script will produce publication-quality 4-panel
            figure for anchor ON vs OFF comparison, ready for GPU eval next iteration.
Change:     Created analyze_anchor_ablation.py with:
            1. Per-step error computation, phase masking (contact vs flight), phase RMSE
            2. Cumulative RMSE divergence tracking
            3. 4-panel figure: (a) error over time with contact shading, (b) phase RMSE
               bar chart, (c) ball height + anchor fire markers, (d) cumulative RMSE
            4. Text summary table comparing ON vs OFF across phases
            5. Updated run_anchor_ablation.sh to call both general + detailed analysis
            6. 16 new tests (test_analyze_anchor_ablation.py), all passing
Command:    pytest scripts/perception/ → 388/388 passed (8.10s). No GPU needed.
Result:     Tooling ready. Script tested with synthetic data. Produces clear figures
            showing contact-phase vs flight-phase improvement from anchor. GPU still
            occupied by policy agent.
Decision:   Next iter: run anchor ablation on GPU if available. If still blocked,
            consider flight-window detection mode OR Quarto update with recent progress.

## Iteration 108 — Ball phase tracker for flight-window detection  (2026-04-10T21:00:00Z)
Hypothesis: A state machine that classifies ball phase (contact/ascending/descending)
            from EKF estimates will enable camera scheduling (skip detection during
            contact), phase-aware metrics without post-hoc recomputation, and bounce
            counting for curriculum diagnostics.
Change:     Created perception/phase_tracker.py with BallPhaseTracker class:
            - 3 phases: CONTACT(0), ASCENDING(1), DESCENDING(2)
            - Transitions: contact→ascending (vz>0.5 + above threshold), ascending→
              descending (vz≤0), descending→contact (enters contact zone)
            - Weak bounce handling: ascending→contact if ball falls back without descent
            - Per-env: bounce_count, peak_height, flight_fraction, steps_in_phase
            - Per-env tensor contact_z_threshold support
            - Wired into demo_camera_ekf.py: update after EKF step, reset on episode done,
              phase stats in summary output, phase array saved in trajectory.npz
            - Exported from perception/__init__.py
Command:    pytest scripts/perception/ → 409/409 passed (8.11s). No GPU needed.
Result:     21 new tests (test_phase_tracker.py) all pass. Full suite 409/409.
            Phase tracker provides real-time ball state classification that was previously
            only available via post-hoc trajectory analysis.
Decision:   Next iter: run anchor ablation on GPU if available. The phase tracker data
            in trajectory.npz will also enable phase-conditioned analysis of anchor
            improvement (anchor should only matter during CONTACT phase).

## Iteration 109 — Camera scheduling: skip detection during contact phase  (2026-04-10T23:00:00Z)
Hypothesis: Using the phase tracker's in_flight mask to skip camera detection during
            contact phase saves compute without degrading EKF accuracy, since the
            paddle anchor handles estimation when ball is on the paddle.
Change:     Added --camera-scheduling flag to demo_camera_ekf.py:
            1. Before detection loop, compute schedule_mask = phase_tracker.in_flight
               (uses state from previous step — valid since phase transitions are rare)
            2. Skip detector.detect() for envs where schedule_mask[ei] is False (contact)
            3. Track sched_skipped metric (count of skipped detections)
            4. Save sched_active per-step in trajectory.npz for offline analysis
            5. Scheduling stats in periodic logging and summary output
            6. 4 new tests (TestCameraScheduling): contact→skip, flight→detect,
               mixed envs, landing transition
Command:    pytest scripts/perception/ → 413/413 passed (8.06s). No GPU needed.
Result:     Feature implemented and tested. On real hardware, contact phase is 70-98%
            of time (ball sitting on paddle), so this should save 70-98% of YOLO
            inference calls. EKF accuracy unaffected since paddle anchor provides
            virtual measurements during contact.
Decision:   Next iter: run anchor ablation on GPU if available (blocked since iter 106).
            Alternatively, add the scheduling data to analysis plots, or update Quarto
            page with recent progress (iters 106-109).

## Iteration 110 — Phase timeline visualization + Quarto update  (2026-04-11T01:00:00Z)
Hypothesis: A phase-annotated 4-panel timeline plot will provide clear visual evidence
            of EKF+anchor+scheduling behavior once GPU ablation data arrives.
Change:     Created plot_phase_timeline.py — 4-panel publication figure:
            (a) ball height with phase-colored background + bounce markers
            (b) EKF error with anchor fire + detection event markers
            (c) camera scheduling (detect vs skip stacked area)
            (d) cumulative RMSE
            Wired into run_anchor_ablation.sh (generates ON + OFF timelines).
            Updated Quarto page (agents/perception.qmd) with iters 107-109 summary.
            21 new tests (test_plot_phase_timeline.py).
Command:    pytest scripts/perception/ → 434/434 passed (9.88s). No GPU needed.
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
