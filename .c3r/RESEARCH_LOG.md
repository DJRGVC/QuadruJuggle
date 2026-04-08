# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## Compacted summary (through iter_008)

**Oracle baseline & pi2 selection (iters 001-003):** Established that the 41D pi2 checkpoint
(2026-03-12_09-04-32/model_best.pt) is the correct one — the 53D pi2 regresses badly (Stage C
vs Stage D). Best oracle baseline: iter_003 (41D pi2, 12288 envs, 500 iters) → timeout=98.9%,
mean_ep_len=1500, apex_rew=2.92, Stage D. The policy survives near-perfectly but never actively
juggles — ball sits on paddle, earning survival reward.

**Perception noise integration & d435i comparison (iters 004-005):** Integrated perception
agent's ball_obs_spec.py into pi1 env. Added --noise-mode oracle|d435i CLI flag. D435i noise
causes moderate degradation (~8% reward, ~14pp more ball drops) but does NOT prevent curriculum
advancement. Both oracle and d435i reach Stage D and plateau identically — the bottleneck is
reward shaping, not perception noise.

**Stage D/F plateau diagnosis & noise curriculum (iters 006-008):** Wrote noise curriculum plan
(3-phase: oracle→scaled d435i→EKF). Lowered _BJ_APEX_THRESHOLD from 5.0→2.0 to break Stage D
plateau — curriculum advanced D→E→F. Implemented noise_scale column in _BJ_STAGES (0→100%
across curriculum). KEY FINDING: noise-curriculum outperforms pure oracle at Stage F (apex 1.76
vs 1.10), suggesting noise acts as implicit regularization. All runs plateau at Stage F with
"balance not bounce" local optimum — policy earns more from passive survival than risky juggling.

**Root cause chain (iters 009-010):** (1) sigma_ratio=2.5 gives ball-at-rest 4.4% of max apex
reward per step (1650/ep guaranteed). FIX: sigma_ratio→3.5 (0.2% at rest, 82/ep). (2) Even
with tight sigma, alive=1.0/step (1500/ep) still dominates — balancing is always safer than
juggling. FIX: ball_low_penalty=-1.0/step when ball h≤threshold, cancelling alive during balance.

**Key checkpoints:**
- Oracle baseline: logs/.../2026-04-07_20-18-34/model_best.pt (iter_003, Stage D)
- Noise-curriculum: logs/.../2026-04-07_23-20-25/model_1199.pt (iter_008, Stage F)
- Pi2: /home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_09-04-32/model_best.pt (41D, always use this)

**Infrastructure built:** compare_pi1.py, --noise-mode in train/eval, wandb integration,
noise_scale curriculum, action_term auto-detect 41D/53D.

---

## iter_009 — fresh d435i 1500-iter with threshold=1.5  (2026-04-08T01:00Z)
Hypothesis: Fresh run with _BJ_APEX_THRESHOLD=1.5 (lowered from 2.0) and d435i noise will
advance further than iter_007 oracle run (Stage F plateau, apex≈1.1).
Change:     User ran fresh training with --noise-mode d435i, 12288 envs, 1500 max iters,
            _BJ_APEX_THRESHOLD=1.5 (lowered automatically from train_juggle_hier.py state).
            Checkpoint dir: logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_00-55-03/
Command:    gpu_lock.sh uv run --active python scripts/rsl_rl/train_juggle_hier.py \
              --task Isaac-BallJuggleHier-Go1-v0 \
              --pi2-checkpoint .../2026-03-12_09-04-32/model_best.pt \
              --num_envs 12288 --max_iterations 1500 --headless --noise-mode d435i --wandb
Result:     Early stopped at iter 1127 (700-iter patience exhausted at Stage F).
            Curriculum: A→B→C→D (fast, first ~300 iters), then stuck at Stage F.
            Final metrics:
              mean_episode_length: 1381, timeout: 94.7%, apex_rew: 1.09, ball_below: 5.3%
              mean_reward: 55.3, noise_std: 0.375
            Same Stage F plateau at apex_rew≈1.09 — identical to iter_007 oracle (apex=1.10).
            
            ROOT CAUSE IDENTIFIED: sigma_ratio=2.5 means ball AT REST earns
            exp(-sigma_ratio²/2) = exp(-3.125) = 4.4% of max apex reward per step.
            With weight=25 and 1500 steps: ball at rest earns 25×0.044×1500=1650 apex reward.
            The policy converges to balance (certain survival+1650 apex) vs risky juggling.
            
            FIX: sigma_ratio=3.5 gives exp(-3.5²/2) = 0.0022 (0.2%) per step.
            Ball-at-rest apex contribution = 25×0.0022×1500 = 82.5 (down from 1650, 20× less).
            Now juggling is worth ~20× more relative to balancing → should break plateau.
            Also lowered _BJ_APEX_THRESHOLD to 0.5 (appropriate for tighter sigma).
Decision:   Next iter_010: fresh run with sigma_ratio=3.5 (all stages) + d435i noise.
            Checkpoint dir for iter_009: logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_00-55-03/model_best.pt

## iter_010 — sigma_ratio=3.5 fresh run (1500-iter d435i)  (2026-04-08T03:30Z)
Hypothesis: sigma_ratio=2.5→3.5 reduces ball-at-rest apex reward from 4.4% to 0.2% per step,
forcing active throwing to earn meaningful reward.
Change:     sigma_ratio changed from 2.5 to 3.5 (Stage C-P), 3.0 (Stage B), 2.5 (Stage A kept wider for bootstrap).
            _BJ_APEX_THRESHOLD lowered to 0.5.
Command:    gpu_lock.sh uv run --active python scripts/rsl_rl/train_juggle_hier.py \
              --task Isaac-BallJuggleHier-Go1-v0 \
              --pi2-checkpoint .../2026-03-12_09-04-32/model_best.pt \
              --num_envs 12288 --max_iterations 1500 --headless --noise-mode d435i --wandb
            Checkpoint dir: logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_03-28-26/
Result:     At iter 524 (still running, converged): timeout=99.4%, apex_rew=0.07, mean_reward=23.1
            SAME balance local optimum as before.
            
            ROOT CAUSE 2 IDENTIFIED: alive reward (weight=1.0) × 1500 steps = 1500/ep is too dominant.
            With sigma_ratio=3.5, ball at rest earns 0.07/step apex reward. But alive=1500/ep
            regardless of throwing behavior. Policy still earns ~1507/ep from balance vs risk of
            -200 (early_termination) + potential apex upside. Balancing wins.
            
            FIX: ball_low_penalty=-1.0/step when ball h≤0.03m. With this:
            - Balance: alive=1.0 - low_penalty=1.0 = 0/step (break-even vs random)
            - Juggling: alive=1.0 + apex>0 - low_penalty=0 = 1+apex/step (positive)
            This forces policy to choose between 0/step (balance) vs variable (juggle+survive).
Decision:   Next: iter_011 with ball_low_penalty added. Expected: policy must explore throwing
            to avoid earning ~0 from passive balancing.

## iter_011 — ball_low_penalty + sigma_ratio=3.5 (d435i, 456 iters)  (2026-04-08T04:04Z)
Hypothesis: ball_low_penalty=-1.0/step when h≤threshold forces balance reward to ~0/step,
            making juggling strictly better than passive balancing for the first time.
Change:     Added ball_low_penalty() to rewards.py (returns 1.0 when h≤threshold, 0 otherwise).
            Dynamic threshold = 40% of target height per stage.
            Added ball_low RewTerm(weight=-1.0) to ball_juggle_hier_env_cfg.py.
Command:    gpu_lock.sh uv run --active python scripts/rsl_rl/train_juggle_hier.py \
              --task Isaac-BallJuggleHier-Go1-v0 \
              --pi2-checkpoint .../2026-03-12_09-04-32/model_best.pt \
              --num_envs 12288 --max_iterations 1500 --headless --noise-mode d435i --wandb
            Checkpoint dir: logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_04-04-38/
Result:     Run completed 456 iterations (killed by wall-clock cap).
            TRAJECTORY: apex_rew peaked at iter 200 (13.7!) then COLLAPSED to 0.18 by iter 400+.
              iter   0: apex=0.11, timeout=1.5%,  ball_low=-0.015
              iter  50: apex=1.30, timeout=0.3%,  ball_low=-0.185
              iter 100: apex=4.27, timeout=1.4%,  ball_low=-0.313
              iter 150: apex=9.70, timeout=59.6%, ball_low=-0.500
              iter 200: apex=13.7, timeout=71.6%, ball_low=-0.473  ← PEAK (active juggling!)
              iter 250: apex=10.4, timeout=61.9%, ball_low=-0.310
              iter 300: apex=0.67, timeout=74.0%, ball_low=-0.410  ← COLLAPSE begins
              iter 350: apex=0.56, timeout=81.1%, ball_low=-0.493
              iter 400: apex=0.19, timeout=89.9%, ball_low=-0.570
              iter 456: apex=0.18, timeout=97.5%, ball_low=-0.565  ← back to passive balance
            
            Final metrics (iter 456):
              mean_reward: -27.1  (NEGATIVE — penalties dominate)
              alive: 0.988, ball_low: -0.566, foot_contact: -0.778
              ball_below: 2.5%, ball_off: 0%
              noise_std: 0.541
            
            KEY FINDING: ball_low_penalty successfully broke the balance optimum early (iter 50-250,
            apex peaked at 13.7). But the policy then UNLEARNED juggling and reverted to passive
            survival (iter 300+, timeout→97.5%, apex→0.18). Classic exploration-exploitation:
            the policy explored juggling but found the variance too high and collapsed back to
            safe passive balance. The ball_low penalty is incurred (57% of time) but the policy
            accepts it rather than risk ball_below termination from juggling.
            
            Checkpoint: logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_04-04-38/model_best.pt
            NOTE: model_best.pt likely captured during the peak (iter ~150-200); model_450.pt
            is the latest but passive policy.
Decision:   ball_low_penalty=-1.0 is insufficient — exactly as lit-review predicted (need -2.0).
            At weight=-1.0, passive balance earns alive(1.0)-low(1.0)=0/step, but juggling risks
            ball_below (-200 terminal). Policy rationally prefers 0/step certain over variable.
            
            OPTIONS for next iteration:
            (1) Increase ball_low weight to -2.0 (lit-review recommendation) → balance earns -1/step
            (2) Add ball_release_velocity_reward (DribbleBot/JuggleRL pattern) → positive reward for
                upward ball velocity at paddle separation
            (3) Warm-start from model_best.pt (peak at iter ~200) to preserve juggling behavior
            (4) Gate apex_height_reward on is_airborne (PBRS approach)
            
            RECOMMENDED: Try (1) weight=-2.0 first (simplest change), warm-start from model_best.pt.

## iter_012 — compaction (summarized iters 001-008)  (2026-04-08T15:30Z)
Hypothesis: N/A — compaction iteration.
Change:     Archived iters 001-008 to RESEARCH_LOG_ARCHIVE.md. Wrote compacted summary covering
            oracle baseline, perception noise integration, Stage D/F plateau diagnosis, noise
            curriculum, and root cause chain. Pruned fix_plan.md.
Command:    No GPU commands.
Result:     Log shrunk from 340→~130 lines. Archive preserved verbatim.
            Processed INBOX: lit-review's ball_low_penalty=-2.0 recommendation — already applied
            -1.0 in iter_011, observed peak then collapse. -2.0 is queued for iter_013.
            
            iter_011 FINAL RESULTS (not previously logged — run completed after last context):
              456 iterations total. Apex peaked at iter 200 (13.7) then collapsed to 0.18.
              Policy discovered juggling but unlearned it in favor of passive survival.
              model_best.pt likely from peak period; model_450.pt is passive.
Decision:   Next iteration (iter_013): increase ball_low weight to -2.0, warm-start from
            iter_011 model_best.pt to preserve the juggling behavior from the peak.
