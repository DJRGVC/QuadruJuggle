# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## iter_001 — D435i roadmap audit  (2026-04-07T00:00:00Z)
Hypothesis: The existing perception_roadmap.md was written for a mono USB camera; updating it for the D435i will expose concrete differences in noise model parameters, detection pipeline, and EKF timing that will guide subsequent simulation work.
Change:     Updated docs/perception_roadmap.md to reflect D435i sensor throughout: Phase 3 camera selection (D435i chosen, ELP rejected with reasoning), Phase 4 detection (depth from D435i depth frame not apparent ball size), Phase 2 noise model (σ_base 3mm→1-2mm, σ_dist 5mm/m→2mm/m, latency ~1 policy step), EKF timing (60-120Hz→30-90Hz depth), deployment stack (D435i USB3 + predict-every-5ms EKF).
Command:    Edit docs/perception_roadmap.md (6 targeted edits, no code written)
Result:     Roadmap now accurately reflects D435i. Key implication: depth accuracy is ~3mm at 1m vs ~65mm/m for mono — noise model is ~10× better at long range, which matters for Stage E-G (0.6-1.0m apex). Dropout mode changes from "ball too small to detect" to "IR reflection off shiny surface".
Decision:   Next iter: task 2 — survey Isaac Lab camera sensor API (RayCasterCamera vs TiledCamera) and document choice for D435i semantics simulation.

---

## iter_002 — Isaac Lab camera sensor survey → CAMERA_CHOICE.md  (2026-04-07T00:30:00Z)
Hypothesis: Surveying RayCasterCamera vs TiledCamera vs base Camera will reveal a clear winner for D435i semantics, with a key architectural finding that the ETH noise-injection approach requires no camera at all during training.
Change:     Created `source/go1_ball_balance/go1_ball_balance/perception/__init__.py` (perception package stub) and `perception/CAMERA_CHOICE.md` (full sensor survey + decision doc).
Command:    Read isaaclab/sensors/ray_caster/ray_caster_camera.py, tiled_camera.py, camera.py, sensor_base_cfg.py; wrote CAMERA_CHOICE.md.
Result:     Key finding: RayCasterCamera DISQUALIFIED — only works on static meshes, cannot see the moving ball. TiledCamera is the correct debug option (RGB + depth, dynamic objects, parallel). But the training pipeline uses NO camera sensor — ETH architecture injects noise directly on GT state, batched across 12288 envs on GPU. TiledCamera config spec written for debug/play use (30Hz, 640×480, 45° upward mount).
Decision:   Next iter: task 3 — create `perception/ball_obs_spec.py` stub returning oracle ball state with explicit zero-noise flag, consumed by policy agent's pi1 training script.

---

## iter_003 — ball_obs_spec.py: ObsTerm-compatible perception interface  (2026-04-07T01:00:00Z)
Hypothesis: Creating ObsTerm-compatible wrapper functions with an explicit noise_cfg parameter will let the policy agent swap oracle→noisy observations without changing env_cfg wiring, just the noise_cfg dataclass.
Change:     Created `perception/ball_obs_spec.py` with `ball_pos_perceived()` and `ball_vel_perceived()` — drop-in replacements for `mdp.ball_pos_in_paddle_frame` / `mdp.ball_vel_in_paddle_frame`. Three modes: "oracle" (GT passthrough), "d435i" (structured depth-dependent noise), "ekf" (NotImplementedError stub). D435i noise model: XY σ=2mm base, Z σ=3mm+2mm/m, 2% dropout, velocity noise derived from finite-differenced position at 30Hz. Updated `__init__.py` to export public API (`BallObsNoiseCfg`, `D435iNoiseParams`, `ball_pos_perceived`, `ball_vel_perceived`).
Command:    AST parse + dataclass construction test (isaaclab not available outside sim env)
Result:     Module parses cleanly, dataclasses construct with defaults and custom overrides. Full sim integration test deferred to camera mount iteration (task 5). Key design: `BallObsNoiseCfg(mode="oracle")` is the default → zero behavior change when policy agent swaps in these functions.
Decision:   Next iter: task 4 — survey docs/sim_to_real_plan.md for camera mount assumptions and update for rear-paddle-mounted 45° upward D435i orientation.

---

## iter_004 — sim_to_real_plan.md D435i update  (2026-04-07T02:00:00Z)
Hypothesis: The sim_to_real_plan.md still references a generic "global shutter USB cam" with monocular depth and 120Hz; updating it for D435i will align the hardware preparation track (Track 3), latency budget, comms architecture, and risk assessment with the actual sensor choice.
Change:     13 targeted edits to docs/sim_to_real_plan.md: (1) perception gap table → "noisy D435i depth", (2) latency table → D435i 11ms readout at 90Hz, (3) Track 2 summary → "D435i stereo depth lookup", (4) Track 3A camera row → D435i rear-paddle-mounted 45° upward with mass/USB3 details, (5) added camera-paddle transform row, (6) updated cabling for USB3, (7) added mass → 170g assembly, (8) comms diagram → D435i 30-90Hz, (9) compute stack → librealsense2 SDK, (10) DR base mass note → D435i ~90g, (11) calendar → D435i-specific tasks, (12) risk: motion blur → D435i IR global shutter, (13) risk: detection → D435i stereo depth advantage. Added 45° mount rationale paragraph with FoV geometry analysis.
Command:    Read + Edit (no code, doc-only iteration)
Result:     sim_to_real_plan.md now consistent with perception_roadmap.md on D435i choice. Key additions: 45° mount geometry rationale (FoV spans 0m to >1m), camera-paddle rigid transform spec (~50mm behind, ~30mm below, 45° Y-rotation), assembly mass estimate (170g vs prior ~25g mono), librealsense2 in software stack.
Decision:   Next iter: task 5 — mount a simulated D435i in the Isaac Lab scene env_cfg and take a debug snapshot. This requires reading the existing scene setup and adding a TiledCamera sensor (per CAMERA_CHOICE.md decision).
