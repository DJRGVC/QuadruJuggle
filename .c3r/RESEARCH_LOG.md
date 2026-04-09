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
