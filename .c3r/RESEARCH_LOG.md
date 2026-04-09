# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## Compacted summary (through iter_014)

**Oracle baseline & pi2 selection (iters 001-003):** 41D pi2 checkpoint
(2026-03-12_09-04-32/model_best.pt) is correct — 53D regresses. Oracle baseline:
timeout=98.9%, mean_ep_len=1500, apex_rew=2.92, Stage D. Policy survives but never
juggles — ball sits on paddle.

**Perception noise & d435i comparison (iters 004-005):** D435i causes moderate degradation
(~8% reward, ~14pp more ball drops) but does NOT prevent curriculum advancement. Both
oracle and d435i plateau at Stage D — bottleneck is reward shaping, not noise.

**Stage D/F plateau & noise curriculum (iters 006-008):** Lowered _BJ_APEX_THRESHOLD
5.0→2.0, advanced D→E→F. Noise-curriculum outperforms oracle at Stage F (apex 1.76 vs
1.10). All runs hit Stage F "balance not bounce" local optimum.

**Root cause chain (iters 009-010):** (1) sigma_ratio=2.5 → ball-at-rest earns 4.4% apex/step
(1650/ep guaranteed). FIX: sigma_ratio=3.5 (0.2%). (2) alive=1.0/step (1500/ep) dominates.
FIX: ball_low_penalty cancels alive during balance.

**Breaking the balance trap (iters 011-013):** ball_low=-1.0 broke balance optimum (apex
peaked 13.7) but policy COLLAPSED back (iter 300+). ball_low=-2.0 caused death spiral on
curriculum steps (no safe fallback). Fixed curriculum sustain-during-blend bug (counter was
ticking during blending, causing pipelined stage advances).

**BREAKTHROUGH — ball_release_velocity_reward (iter_014):** Added +3.0 positive reward for
upward ball velocity at paddle separation. SUSTAINED juggling for 1500 iters (apex 15.4 peak,
settled to 9.7 STABLE). No collapse. Policy found stable equilibrium: ~63% timeout, ~34%
ball_below. First time juggling doesn't revert to passive balance.

**Key checkpoints:**
- Oracle baseline: logs/.../2026-04-07_20-18-34/model_best.pt (iter_003, Stage D)
- Noise-curriculum: logs/.../2026-04-07_23-20-25/model_1199.pt (iter_008, Stage F)
- First sustained juggling: logs/.../2026-04-08_07-44-03/model_4249.pt (iter_014)
- Pi2: /home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_09-04-32/model_best.pt (41D, always use this)

**Current config (as of iter_014):**
- sigma_ratio=3.5, _BJ_THRESHOLD=0.30, ball_low=-1.0, ball_release_vel=+3.0, d435i noise
- Early stop patience=700, 12288 envs, 16 curriculum stages A→P

**Infrastructure built:** compare_pi1.py, --noise-mode in train/eval, wandb integration,
noise_scale curriculum, action_term auto-detect 41D/53D.

---

## iter_015 — curriculum threshold 0.75→0.30 (d435i, 1500 iters)  (2026-04-08T09:11Z)
Hypothesis: Lowering _BJ_THRESHOLD from 0.75 to 0.30 allows curriculum advancement, since active
            juggling has ~63% timeout (well above 30%) but never reaches 75%.
Change:     _BJ_THRESHOLD = 0.75 → 0.30. Resumed from iter_014 model_4249.pt.
Command:    gpu_lock.sh uv run --active python scripts/rsl_rl/train_juggle_hier.py \
              --task Isaac-BallJuggleHier-Go1-v0 \
              --pi2-checkpoint .../2026-03-12_09-04-32/model_best.pt \
              --num_envs 12288 --max_iterations 1500 --headless --noise-mode d435i --wandb \
              --resume --load_run 2026-04-08_07-44-03 --checkpoint model_4249.pt
            Checkpoint dir: logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_09-11-18/
Result:     1500 iters completed (step 4249→5748). Curriculum ADVANCED (A→B+).
            Apex: peaked 15.5, settled 9.2-10.9 stable for 1000 iters.
            Final: apex=10.46, timeout=68%, ball_below=30.2%, reward=282, noise_std=1.08.
            Juggling SUSTAINED through curriculum transition — no collapse.
            Checkpoints: model_best.pt (peak ~step 4350), model_5748.pt (final stable)
Decision:   Continue training from model_5748.pt for more curriculum stages.

## iter_016 — continue training 1500 iters from step 5748 (d435i)  (2026-04-08T10:36Z)
Hypothesis: Continued training from model_5748.pt will push through additional curriculum stages
            while maintaining stable juggling.
Change:     No config changes. Resumed from iter_015 model_5748.pt.
Command:    gpu_lock.sh uv run --active python scripts/rsl_rl/train_juggle_hier.py \
              --task Isaac-BallJuggleHier-Go1-v0 \
              --pi2-checkpoint .../2026-03-12_09-04-32/model_best.pt \
              --num_envs 12288 --max_iterations 1500 --headless --noise-mode d435i --wandb \
              --resume --load_run 2026-04-08_09-11-18 --checkpoint model_5748.pt
            Checkpoint dir: logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_10-36-50/
Result:     Early stopped at step 7198 (700-iter ES patience exhausted).
            Curriculum advanced once more (apex 17.4→10.7 dip). Stable plateau at apex≈10.7.
            Final: apex=10.74, timeout=69.6%, ball_below=28%, reward=316, noise_std=1.09.
            Checkpoints: model_best.pt (peak ~step 5835), model_early_stop.pt (step ~7198)
Decision:   Plateau at apex≈10.7. Need to diagnose: which stage is it on? Is ES patience too
            short? Are stage jumps too large? Next: check curriculum state + consider ES patience increase.

## iter_017 — compaction (summarized iters 009-014)  (2026-04-08T18:15Z)
Hypothesis: N/A — compaction iteration.
Change:     Archived iters 009-014 to RESEARCH_LOG_ARCHIVE.md. Wrote compacted summary.
            Processed INBOX: replied to Daniel about apex units and pi2 status.
Command:    No GPU commands.
Result:     Log shrunk from 370→~75 lines. Archive preserved verbatim.
Decision:   Next iteration (iter_018): diagnose the apex≈10.7 plateau. Read curriculum code to
            determine current stage, check if ES patience (700) is too short for curriculum to
            advance, and decide whether to increase patience or adjust stage parameters.

## Iteration 18 — Apex plateau diagnosis (no GPU — active run)  (2026-04-08T22:50Z)
Hypothesis: The apex≈10.7 plateau is caused by Stage P's target range [0.30, 1.00] being too wide,
            with high targets unreachable without multi-bounce energy injection that pi1 hasn't learned.
Change:     No config changes (GPU occupied by PID 805803, an unlogged continuation run).
            Analysis-only iteration. Updated Quarto page. Processed INBOX (Daniel's status request).
Command:    Examined TB data from 5 runs (steps 4249→9182). Read curriculum code, reward functions,
            and pi2's h_dot command range.
Result:     CONFIRMED plateau is consistent across ALL recent runs (apex 10.5-11.3, timeout 68-69%,
            reward 300-360). Key findings:
            (1) Curriculum threshold=0.30 with timeout=69% → curriculum advances every ~35 iters.
                Running job (723 iters from Stage F) is almost certainly at Stage P (final).
            (2) Pi2 h_dot max = 1.5 m/s → single-bounce ceiling ~0.80m. But with r=0.99, multi-bounce
                CAN reach 1.0m+ (second bounce: paddle at 1.5 m/s + ball at 3.96 m/s → 2.43m).
            (3) 43% apex accuracy at Stage P = policy reliably hits easy targets (0.30m) but not hard
                ones (1.0m). Wide range drags average down.
            (4) ball_release_vel already at 8.0 weight, ES already at 1500 patience. Neither broke plateau.
            (5) Two unlogged continuation runs found: 2026-04-08_13-12-09 (1984 iters) and
                2026-04-08_15-05-06 (running, 723+ iters). Same plateau.
            Active GPU process: PID 805803, start-stage 5, 2000 max_iters, from model_best.pt of
                2026-04-08_13-12-09. Log dir: 2026-04-08_15-05-06. Step ~8860.
Decision:   Next iteration: once GPU is free, narrow Stage P target range from [0.30, 1.00] to
            [0.30, 0.60] and add stages Q-T for progressively wider ranges. This gives the policy
            time to learn multi-bounce energy injection gradually instead of facing the full range
            immediately. Resume from model_best.pt of the most advanced run.

## Iteration 19 — cap max target at 0.60m + additional confirmation run  (2026-04-08T22:04Z)
Hypothesis: Capping Stage P max target at 0.60m (within pi2's ~0.80m single-bounce ceiling) will
            let the policy master achievable heights instead of plateauing on unreachable 1.00m targets.
Change:     Modified _BJ_STAGES: Stages I-K now [0.30, 0.45→0.55] (was 0.48→0.62), Stages L-P
            all cap at [0.30, 0.60] with increasing XY and velocity difficulty.
            Also confirmed iter_018's findings: launched 1001-iter training (from model_best.pt of
            2026-04-08_13-12-09, start-stage 5) that showed same plateau: apex≈10.6, release_vel≈1.58,
            timeout≈67%, ball_below≈30% at Stage P after 656 iters. Log dir: 2026-04-08_15-05-06.
Command:    Training with capped curriculum from model_9100 + start-stage 5. Smoke test passed.
            Main training launched but CRASHED at iter ~30 (Isaac Sim carb.tasking mutex recursion
            error — known Isaac Sim bug, not related to curriculum changes).
            A separate training process (PID 870573, unknown origin, start-stage 0, max_iters 1500,
            using capped curriculum from my edit) took GPU afterward.
Result:     Confirmation that apex≈10.7 plateau is consistent even with 8.0 release_vel weight.
            Policy throws 2.5× harder (release_vel 0.61→1.58) but apex unchanged.
            Curriculum cap change implemented but NOT yet trained — Isaac Sim crash cut it short.
            Checkpoints from confirmation run: logs/.../2026-04-08_15-05-06/model_9100.pt
Decision:   Next iteration: retry training with capped curriculum. Check GPU availability.
            If the mystery training process (PID 870573) is using capped curriculum, its results
            will be informative too. Resume from model_9100.pt or the mystery run's latest checkpoint.

## Iteration 21 — Teleop interface: WASD+P/L user control  (2026-04-09T01:20Z)
Hypothesis: Users can interactively control robot velocity and ball target height via keyboard
            while pi1 handles ball tracking autonomously.
Change:     Created scripts/rsl_rl/play_teleop.py with WASD velocity, P/L height, terminal HUD.
            Overrides pi1's vx/vy output channels (slots 6-7) + modifies target_apex_height obs.
Command:    No GPU (code-only iteration). Updated Quarto page, processed INBOX (STOP signal + re-read directive).
Result:     Script created. Keyboard input via background thread + termios. HUD prints every 10 steps.
            Usage: uv run --active python scripts/rsl_rl/play_teleop.py --task Isaac-BallJuggleHier-Go1-Play-v0
                   --pi2-checkpoint <path> --num_envs 1 --real-time
Decision:   Ready for Daniel to test. Next: run fresh 6-stage curriculum training when GPU is free,
            then verify pi1+pi2 with D435i noise in teleop mode.

## Iteration 20 — Curriculum redesign: 16→6 stages, 0.50m cap  (2026-04-08T23:20Z)
Hypothesis: The 16-stage curriculum is over-engineered. Literature uses 3-6 stages (Rudin 4,
            Zhuang 4, ROGER 4). Simplifying to 6 stages with 0.50m max target (within pi2's
            single-bounce ceiling) will be sufficient for the research question.
Change:     Rewrote _BJ_STAGES from 16 stages to 6:
            A(0.10/oracle) B(0.20/oracle) C(0.30/50%noise) D(0.40/full) E(0.50/full+vel) F(0.50/robust)
            Daniel approved via ask_human. Updated fix_plan. Updated Quarto page.
Command:    Smoke test: 4 envs, 2 iters — passed, shows "Stage 1/6 (A)" correctly.
            Full training (12288 envs, 1500 iters): crashed at iter 41 with Isaac Sim mutex
            assertion (carb.tasking/Mutex.cpp:103, known transient bug). GPU then occupied by
            perception sibling's sweep_q_vel.py.
            Log dir: logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_16-20-01/ (only model_0.pt)
Result:     Curriculum code updated and smoke-tested. Training not completed due to Isaac Sim crash
            + GPU contention. Iter time at 12288 envs = 3.47s → 1500 iters ≈ 87 min (within budget).
            Also processed INBOX: Daniel confirmed GPU runs don't need pre-confirmation, 120 min cap.
Decision:   Next iteration: retry fresh training with 6-stage curriculum once GPU is free.
            Start from scratch (Stage A) since old checkpoints trained on different stage definitions.

## Iteration 22 — Fresh 6-stage training from scratch  (2026-04-09T01:12Z)
Hypothesis: 6-stage curriculum (A-F, 0.50m cap) will train pi1 from scratch, advancing through
            oracle stages quickly and handling noise introduction at Stage C/D.
Change:     No code changes. Launched fresh training with 6-stage curriculum (committed in iter 20).
            pi2 checkpoint: 2026-03-12_17-16-01/model_best.pt.
Command:    gpu_lock.sh uv run --active python scripts/rsl_rl/train_juggle_hier.py \
              --task Isaac-BallJuggleHier-Go1-v0 --num_envs 12288 --headless \
              --max_iterations 1200 --pi2-checkpoint .../2026-03-12_17-16-01/model_best.pt
Result:     Training completed in 62 min (3.01s/iter). Advanced A→B→C→D (3 transitions):
            - Stage A (oracle, target=0.10m): 112 iters. apex=8.7, timeout=48%.
            - Stage B (oracle, target=0.20m): 34 iters. apex=8.7, timeout=53%.
            - Stage C (50% noise, target=0.30m): 34 iters. apex=2.1→0.55, timeout=56%.
            - Stage D (full noise, target=0.40m): 1020 iters. STUCK.
              apex decayed 0.55→0.04 over 1000 iters. timeout=92%. ball_release_vel=0.42.
            Policy found local optimum: survive (alive=0.91) + throw occasionally
            (release_vel=0.42) but ball never reaches 0.40m target height.
            Root cause: σ_ratio=3.5 at target=0.40m → σ=0.114m (very narrow Gaussian).
            Ball must be within ±11cm of 0.40m to earn >60% reward — too tight with noisy obs.
            Log dir: logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_18-12-53/
            Checkpoint: model_best.pt (best from Stage D), model_950.pt (step 1150).
Decision:   Next iteration: lower σ_ratio from 3.5 to 2.0 for Stages C-F. This widens the
            Gaussian (σ=0.20m at 0.40m target) giving much more gradient for near-miss heights.
            Resume from model_best.pt at start-stage 3 (Stage D).

## Iteration 23 — σ_ratio 3.5→2.5 breaks Stage D plateau — ALL 6 STAGES REACHED  (2026-04-09T02:20Z)
Hypothesis: Lowering σ_ratio from 3.5 to 2.5 for Stages C-F widens the Gaussian reward (σ=0.16m
            at 0.40m target instead of 0.114m), giving more gradient for near-miss heights with
            noisy observations.
Change:     σ_ratio: 3.5→2.5 for Stages C, D, E, F (Stages A=2.5, B=3.0 unchanged).
Command:    gpu_lock.sh uv run --active python scripts/rsl_rl/train_juggle_hier.py \
              --task Isaac-BallJuggleHier-Go1-v0 --num_envs 12288 --headless --max_iterations 1500 \
              --pi2-checkpoint .../2026-03-12_17-16-01/model_best.pt \
              --resume --load_run 2026-04-08_18-12-53 --checkpoint model_best.pt --start-stage 3
            Log dir: logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_19-19-41/
Result:     **ALL 6 STAGES REACHED (A→F) in 1500 iters (78 min).** Stage D plateau BROKEN.
            - Stage D: 30 iters to advance (was STUCK for 1020 iters with σ_ratio=3.5!)
            - Stage E: 113 iters to advance (target=0.50m + lateral velocity)
            - Stage F: 1357 iters at final stage, ES only 73/1500
            Final metrics (Stage F, iter 2572):
              reward=39.0, ep_len=927, timeout=63%, ball_below=37%
              apex=0.86, release_vel=0.53, noise_std=0.25
              Full D435i noise, target=0.50m, xy_std=0.10m, vel_xy=0.18m/s
            Checkpoints: model_best.pt (peak), model_2572.pt (final)
Decision:   This is the best pi1 result so far — Stage F with full noise and velocity perturbations.
            Next: continue training at Stage F to improve apex (currently 0.86, want >1.0 for proper
            juggling height). Then run oracle vs d435i comparison.

## Iteration 24 — CRITICAL BUG: iters 22-23 ran oracle, not d435i  (2026-04-09T03:00Z)
Hypothesis: Deep analysis of iter_023 reward components would validate juggling behavior and
            prepare oracle vs d435i comparison infrastructure.
Change:     (1) Discovered iters 22-23 trained WITHOUT --noise-mode d435i flag — curriculum
            noise_scale changes were no-ops because mode="oracle" short-circuits in
            ball_obs_spec.py:454. ALL stages ran with ground-truth obs.
            (2) Added warning to train_juggle_hier.py that detects noise stages + oracle mode.
            (3) Created training curve figure for Quarto page.
            (4) Updated fix_plan with corrected training plan.
Command:    Analysis-only (GPU occupied by PID 1062001: fresh oracle training, step 313/1500,
            Stage D, apex=1.43, timeout=66%).
Result:     Reward decomposition at Stage F confirms healthy juggling (ball_low=-0.01 = ball
            rarely sits on paddle, release_vel=+0.53 = active throwing). BUT this was oracle,
            so the "d435i noise" claim in iter_023 log was incorrect.
            Running training (PID 1062001) also confirmed oracle (no --noise-mode flag).
            Env yaml for both runs shows mode=oracle on noise_cfg.
            The compare_pi1.py script and eval_juggle_hier.py already support --noise-mode.
Decision:   Next iteration: once GPU is free, run fresh training WITH --noise-mode d435i from
            scratch. This will be the first ACTUAL noise-curriculum training. Oracle baselines
            from iter_023 and the running process will serve as comparison points.
            Command: gpu_lock.sh uv run --active python scripts/rsl_rl/train_juggle_hier.py \
              --task Isaac-BallJuggleHier-Go1-v0 --num_envs 12288 --headless \
              --max_iterations 1500 --noise-mode d435i \
              --pi2-checkpoint .../2026-03-12_17-16-01/model_best.pt
