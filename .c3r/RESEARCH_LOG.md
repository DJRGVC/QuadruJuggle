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
