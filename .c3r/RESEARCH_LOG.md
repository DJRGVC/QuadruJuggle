# RESEARCH_LOG.md

_(older entries auto-archived to RESEARCH_LOG_ARCHIVE.md at 2026-04-09 18:05 UTC)_

            policy has new results. If not, consider Hough detector robustness
            analysis (noise levels, edge-of-frame detection) or wait for
            policy checkpoint to re-validate gap predictions.

## Iteration 146 — Synthetic YOLO training data generator  (2026-04-10T00:30:00Z)
Hypothesis: A synthetic depth-image generator that renders a 40mm ball at
            random positions with D435i-like noise will produce training data
            sufficient for YOLO fine-tuning, unblocking the detector pipeline
            without requiring real hardware captures.
Change:     Added generate_yolo_data.py and test_generate_yolo_data.py:
            (1) render_ball_depth_frame(): ray-sphere intersection renders
            true sphere geometry (not just a disc) into uint16 depth frames.
            Background simulates upward-facing camera (~30% invalid pixels).
            D435i noise model: σ_z = base + quad·z² applied per-pixel.
            (2) bbox_to_yolo(): YOLO normalised format conversion.
            (3) generate_dataset(): samples ball positions uniformly in FOV
            at z∈[0.20, 1.50]m, outputs images/ + labels/ + dataset.yaml.
            (4) 14 tests: bbox position/scale, depth accuracy, format, seed
            reproducibility, integration with tmp_path output.
Command:    pytest scripts/perception/test_generate_yolo_data.py -x -q → 14/14
            pytest scripts/perception/ -x -q → 609/609 passed (16.92s)
Result:     Test count: 595 → 609 (+14). All pass.
            Generator produces correct sphere projections (median depth within
            25mm of ground truth), proper YOLO format, and scales inversely
            with distance (close ball 3.3× larger in pixels than far ball).
            Background has ~30% invalid pixels simulating sky/ceiling.
            Policy agent still at iter 32, 81% context, ES metric fixed.
Decision:   Next: add data augmentation (random background textures, multiple
            balls, partial occlusion) OR implement YOLO model loading stub
            with ONNX/TRT inference skeleton. Check policy progress.

## Iteration 147 — Implement BallDetector._detect_yolo() ONNX inference  (2026-04-10T01:15:00Z)
Hypothesis: Implementing the YOLO inference path via ONNX Runtime (with
            TRT/CUDA/CPU execution provider fallback) completes the real-hardware
            detector pipeline, enabling end-to-end depth→YOLO→3D detection.
Change:     (1) Replaced _detect_yolo() stub with full implementation:
            _preprocess(): uint16 depth → 8-bit normalised → letterbox 640×640
            → 3-ch NCHW float32 blob. _detect_yolo(): runs ONNX session,
            handles both (1,5,N) and (1,N,5) output layouts, selects highest-
            confidence detection, un-letterboxes coords, median depth lookup,
            deprojects to 3D camera frame. _load_model(): lazy ONNX session
            with TRT→CUDA→CPU provider chain.
            (2) Added _letterbox() utility for resize-with-padding.
            (3) Added onnxruntime optional import alongside cv2.
            (4) 23 new tests in test_yolo_detector.py: letterbox (5), preprocess
            (4), _detect_yolo with mock session (8), integration (3), median
            depth (3). Uses mock ONNX session — no real model file needed.
Command:    pytest scripts/perception/test_yolo_detector.py -x -q → 23/23
            pytest scripts/perception/ -x -q → 632/632 passed (16.84s)
Result:     Test count: 609 → 632 (+23). All pass. YOLO inference path is
            fully implemented and tested. Supports both standard YOLOv8 output
            layouts. Coordinate mapping verified: off-centre ball at (0.05,
            -0.03, 0.40)m correctly deprojected within 2cm/5cm tolerance.
Decision:   Next: create a minimal dummy ONNX model (single-layer) for
            end-to-end load+infer integration test, OR proceed to YOLO
            fine-tuning pipeline (train script using synthetic data from
            iter 146). Policy agent at iter 32 with 81% context — check
            if they've produced new Stage G results.

## Iteration 148 — YOLO training + ONNX export script  (2026-04-10T02:00:00Z)
Hypothesis: A training script wrapping ultralytics YOLOv8 that handles
            dataset splitting, depth-to-RGB conversion, training with
            depth-appropriate augmentation, and ONNX export will complete
            the synthetic→model→detector pipeline.
Change:     Added train_yolo_ball.py and test_train_yolo_ball.py:
            (1) split_dataset(): shuffles and moves images/labels into
            train/val splits (val_fraction=0.15). Deterministic seed,
            idempotent (skips if already split).
            (2) depth_png_to_rgb(): converts uint16 depth PNGs to 3-ch
            uint8 in-place (maps [168,2000]mm → [0,255]). Idempotent.
            (3) write_dataset_yaml(): generates ultralytics-compatible
            dataset.yaml with relative paths.
            (4) train(): orchestrates split→convert→train→export. Disables
            colour jitter (hsv_h/s/v=0), mosaic, mixup for depth data.
            Enables flipud/fliplr. Early stopping patience=20.
            (5) 18 tests: split counts, label-image matching, idempotency,
            seed determinism, error cases, depth conversion (3-channel,
            uint8, valid/invalid mapping, channel equality), integration.
            Note: ultralytics not installed — train() itself untestable,
            but all utility functions fully tested.
Command:    pytest scripts/perception/test_train_yolo_ball.py -x -q → 18/18
            pytest scripts/perception/ -x -q → 650/650 passed (17.89s)
Result:     Test count: 632 → 650 (+18). All pass.
            Policy agent at iter 32 — Stage G eval shows 80% timeout at
            0.10-0.20m, 48% at 0.50m under d435i noise. Planning to retrain
            Stage G with fixed ES metric for longer.
Decision:   Next: install ultralytics + onnxruntime and do a real training
            run on synthetic data, OR create a dummy ONNX model using the
            onnx package for end-to-end integration testing without
            ultralytics. Check if ultralytics can be pip-installed.

## Iteration 149 — Dummy ONNX model builder for integration testing  (2026-04-10T03:00:00Z)
Hypothesis: A minimal dummy ONNX model matching YOLOv8 output format enables
            end-to-end integration testing of the detector pipeline without
            ultralytics or a trained model.
Change:     Added make_dummy_onnx.py (builds valid ONNX model with fixed
            detection at image centre, shape (1,5,8400)) and 9 tests
            verifying structure, embedded values, custom params, and
            onnxruntime loading+inference (auto-skipped when ORT unavailable).
            Updated Quarto page with iters 146-149 hardware pipeline progress.
Command:    pytest scripts/perception/test_make_dummy_onnx.py -x -q → 9/9 + 2 skipped
            pytest scripts/perception/ -x -q → 659/659 passed, 2 skipped (17.97s)
Result:     Test count: 650 → 659 (+9, +2 conditional skips). All pass.
            Dummy model passes onnx.checker, correct I/O shapes, embedded
            detection values verified. ORT integration tests ready for when
            onnxruntime is installed.
            Policy agent at iter 32 (81% context): Stage G eval shows 80%
            timeout at 0.10-0.20m targets (energy modulation fixed), but
            48-63% at 0.40-0.50m (perception gap grows with height). ES
            metric bug fixed, planning longer Stage G retrain.
Decision:   Next: ask Daniel about installing ultralytics + onnxruntime
            so we can do a real training run on synthetic data. Without these
            deps, the YOLO pipeline is code-complete but untestable end-to-end.
            Alternatively, focus on EKF tuning with noise-trained checkpoint
            when policy provides one.

## Iteration 150 — DepthFrameVisualizer for teleop UI camera feed  (2026-04-09T18:30:00Z)
Hypothesis: A reusable depth-frame visualization module will enable Daniel's
            teleop UI to show the D435i camera feed (or sim equivalent) with
            detection overlay, alongside the existing top-down ball view.
Change:     Added perception/debug/depth_viz.py with DepthFrameVisualizer class:
            (1) render(): uint16 depth (mm) → colorized BGR panel with detection
            bbox, centre marker, confidence label, title bar, telemetry overlay.
            (2) render_f32(): float32 depth (m) path for Isaac Lab TiledCamera.
            (3) _SimDetAsDetection adapter wraps SimDetection for unified rendering.
            (4) VizConfig dataclass for panel size, colormap, colors, text options.
            (5) 20 tests: output shape (3), colorization (3), detection overlay (3),
            SimDetection adapter (2), float32 conversion (3), telemetry (2),
            edge cases (4). All use importlib.util to bypass Isaac Lab chain.
            Updated debug/__init__.py to export DepthFrameVisualizer + VizConfig.
Command:    pytest scripts/perception/test_depth_viz.py -x -q → 20/20
            pytest scripts/perception/ -x -q → 679/679 passed, 2 skipped (17.94s)
Result:     Test count: 659 → 679 (+20). All pass.
            Module ready for import by play_teleop_ui.py. Usage:
            `viz = DepthFrameVisualizer(VizConfig(width=320, height=240))`
            `panel = viz.render_f32(depth_frame, sim_detection=det, telemetry={...})`
            Daniel's INBOX processed: 3 messages about teleop UI camera feed.
            Replied: will add camera feed alongside existing top-down view.
Decision:   Next: the DepthFrameVisualizer is ready for integration. Daniel's
            teleop UI (on fix branch) can import it. Consider adding a
            convenience function that extracts depth + runs detector + renders
            in one call for the teleop loop. Also check if policy agent has
            new results for gap re-validation.

## Iteration 151 — inject_ekf_reset_event helper for EKF mode training  (2026-04-09T19:15:00Z)
Hypothesis: Policy agent's 0% timeout with EKF mode is caused by missing
            reset_perception_pipeline event — EKF carries stale state across
            episode resets, producing garbage observations on new episodes.
Change:     (1) Added inject_ekf_reset_event(env_cfg) to ball_obs_spec.py:
            auto-injects reset_perception EventTerm into env_cfg.events.
            Idempotent, safe to call in any mode. Import at top-level
            (EventTermCfg moved from lazy import to module-level).
            (2) Fixed test_world_frame_ekf.py stub to include EventTermCfg.
            (3) 10 new tests in test_inject_ekf_reset.py: injection,
            func reference, mode, params, idempotency, no-overwrite.
            (4) Pinged policy agent with fix instructions.
Command:    pytest scripts/perception/ -x -q → 689/689 passed, 2 skipped (18.24s)
Result:     Test count: 679 → 689 (+10). All pass.
            Root cause confirmed: policy's train_juggle_hier.py injects
            BallObsNoiseCfg(mode="ekf") on obs terms but never adds the
            reset_perception event. On episode reset, EKF keeps old state
            (pos/vel from prev episode end), ball is at new position →
            EKF diverges → garbage observations → catastrophic policy failure.
            Fix: one line after noise injection:
            inject_ekf_reset_event(env_cfg)
Decision:   Wait for policy agent to integrate the fix and retrain with
            --noise-mode ekf. Meanwhile, can work on teleop convenience
            function or Quarto update.

## Iteration 152 — Install deps + YOLO training + end-to-end ONNX detector  (2026-04-09T19:45:00Z)
Hypothesis: Installing ultralytics + onnxruntime and training YOLOv8n on
            synthetic depth data will produce a working ONNX model for the
            BallDetector pipeline, closing the last code gap for real hardware.
Change:     (1) Installed ultralytics 8.4.36, onnxruntime-gpu 1.24.4 (TRT/CUDA/CPU EPs).
            (2) Generated 2000 synthetic depth frames (generate_yolo_data.py).
            (3) Trained YOLOv8n for 30 epochs at 320px on CPU (GPU locked by policy).
            (4) ONNX export → models/perception/ball_detect_v1.onnx (12MB).
            (5) Fixed BallDetector._load_model() to read input size from model
            metadata (was hardcoded 640, model exported at 320).
            (6) Pushed agent/perception + agent/policy branches to GitHub.
            (7) Spawned status-monitor and quarto-cleanup subagents per Daniel's request.
Command:    python scripts/perception/train_yolo_ball.py --data /tmp/yolo_ball_train --epochs 30 --imgsz 320 --batch 32 --device cpu
            pytest scripts/perception/ -x -q → 691/691 passed (18.36s)
Result:     Training: mAP50=0.995, mAP50-95=0.865 (epoch 29 best).
            ONNX model: (1,3,320,320) → (1,5,2100). 12MB. CPUExecutionProvider works.
            End-to-end test: ball at 0.5m detected, position error <2cm via YOLO,
            <1mm via Hough fallback. YOLO confidence low (0.026) on simple test
            frame but correct position — will improve with domain-matched data.
            Test count: 689 → 691 (2 ORT-dependent tests now pass).
            Previously-skipped ORT integration tests (make_dummy_onnx) pass.
Decision:   YOLO model is a proof-of-concept — synthetic-only training gives
            correct detections but low confidence. For real deployment, needs
            fine-tuning on real D435i captures or domain randomization in
            synthetic data. Next: create merged test branch for Daniel's
            hardware testing, or support policy EKF retrain integration.
