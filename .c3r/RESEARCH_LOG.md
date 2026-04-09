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
