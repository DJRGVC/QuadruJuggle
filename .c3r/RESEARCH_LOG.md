# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## iter_001 — oracle pi1 baseline (500-iter, oracle ball obs)  (2026-04-07T19:30 PDT)
Hypothesis: Training pi1 for 500 iterations with oracle (noise-free) ball observations will advance the curriculum to at least Stage C–D and establish a clean reference metric for noise-injection comparison.
Change:     Launched fresh pi1 training run from scratch (no warm start); oracle obs; 4096 envs; 500 iters.
Command:    uv run --active python scripts/rsl_rl/train_juggle_hier.py \
              --task Isaac-BallJuggleHier-Go1-v0 \
              --pi2-checkpoint /home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_09-04-32/model_best.pt \
              --num_envs 4096 --max_iterations 500 --headless
            # pi2 is the 41D (no last_action) checkpoint — matches current action_term.py
            # checkpoint saved to: logs/rsl_rl/go1_ball_juggle_hier/2026-04-07_19-11-34/model_best.pt
Result:     Training time 798s (13.3 min).  Reached Stage D (target=0.20 m) after 211 iters; spent
            remaining 289 iters in Stage D.
            At iter 499 (tbdump metrics):
              Train/mean_episode_length  : 1470.8 steps  (1500 max = 98% timeout)
              Episode_Termination/time_out: 92.4 %
              Episode_Termination/ball_below: 7.6 %  (ball_off: 0 %)
              Episode_Reward/ball_apex_height: 3.107  (needs 5.0 to advance to Stage E)
              Episode_Reward/alive:  0.976
              Train/mean_reward: 69.2
              Policy/mean_noise_std: 0.548  (converging, started at 0.5)
              Perf/total_fps: 125 K env-steps/s
            Stage advancement threshold at D: timeout≥75% (✓ 92%) AND apex≥5.0 (✗ 3.11).
            Policy survives near-perfectly (mean_len≈1470) but ball is NOT yet being juggled
            to the 20 cm apex; ball staying near paddle surface rather than being tossed.
            NOTE: Pre-existing pi1 checkpoint (2026-03-12_10-21-52, 150 iters) only worked
            at Stage A (0.05 m) when evaled. The 500-iter policy is clearly better (Stage D
            survival) but still short of Stage D graduation.
            NOTE: Latest pi2 checkpoint (2026-03-12_17-16-01) is 53D (includes last_action)
            and INCOMPATIBLE with current action_term.py (builds 41D obs).  Bug tracked in
            .c3r/fix_plan.md — needs action_term.py update to add last joint targets.
Decision:   Task 1 complete.  Next: Task 3 — build compare_pi1.py eval infrastructure that
            wraps eval_juggle_hier.py for systematic oracle-vs-noise comparison.  Will also
            note the action_term.py 41→53D bug for a separate fix iteration.

## iter_002 — Fix pi2 obs dim (41→53D) + re-run 500-iter oracle baseline  (2026-04-07T19:43Z)
Hypothesis: The latest pi2 checkpoint (2026-03-12_17-16-01) trained with 53D obs (41D + 12D last_action). action_term.py only builds 41D → shape mismatch crash. Adding last_action to the action term and re-running 500 iters with the best pi2 checkpoint will give a stronger baseline.
Change:     Added `_last_pi2_actions` (12D) buffer to `TorsoCommandAction` in action_term.py, appended to pi2_obs vector (41D→53D). Reinstalled go1_ball_balance editable package from worktree. Used latest pi2 (2026-03-12_17-16-01/model_best.pt, 53D, early-stopped).
Command:    `$C3R_BIN/gpu_lock.sh uv run --active python scripts/rsl_rl/train_juggle_hier.py --task Isaac-BallJuggleHier-Go1-v0 --pi2-checkpoint .../go1_torso_tracking/2026-03-12_17-16-01/model_best.pt --num_envs 12288 --max_iterations 500 --headless`
Result:
  - Training time: 1679s (~28 min), 12288 envs
  - Curriculum: A→B at iter ~333, B→C at iter ~353, stuck at Stage C (target=0.15m) by iter 500
  - Final metrics (iter 499):
    - mean_episode_length: 995.3 (of 1500 max)
    - mean_reward: 115.0
    - time_out %: 47% (needs 75% for C→D)
    - ball_apex_height: 4.44 (needs 5.0)
    - alive: 0.632, noise_std: 0.675
    - ball_off: 0.7%, ball_below: 52.6%
  - Checkpoint: logs/rsl_rl/go1_ball_juggle_hier/2026-04-07_19-43-53/model_best.pt
  - Compared to iter_001 (41D pi2, 4096 envs): WORSE. iter_001 reached Stage D with 92% timeout; iter_002 stuck at Stage C with 47% timeout. Possible causes: (a) 53D pi2 checkpoint is worse quality, (b) 12288 envs vs 4096 changes batch statistics, (c) pi2 last_action integration has a bug (zeros at first step → obs distribution mismatch).
Decision:   The 53D pi2 with last_action underperforms the 41D pi2 from iter_001. Next iteration should use the 41D pi2 checkpoint that worked in iter_001 (2026-03-12_09-04-32) — but first verify it exists on disk. If it does, re-run 500 iters with that pi2 and 12288 envs to get a clean Stage D+ baseline. The action_term fix (53D) is still correct and needed for the newest pi2, but the pi2 itself may have regressed.

## iter_003 — oracle baseline with 41D pi2 + 12288 envs  (2026-04-07T20:18Z)
Hypothesis: iter_002 regression was due to 53D pi2 quality, not env count. Using the proven 41D pi2 checkpoint with 12288 envs should give a stronger oracle baseline (higher timeout, better survival).
Change:     Same 41D pi2 as iter_001 (2026-03-12_09-04-32/model_best.pt) but 12288 envs (3× more than iter_001). action_term.py now auto-detects pi2 input dim → 41D path taken automatically.
Command:    uv run --active python scripts/rsl_rl/train_juggle_hier.py \
              --task Isaac-BallJuggleHier-Go1-v0 \
              --pi2-checkpoint .../go1_torso_tracking/2026-03-12_09-04-32/model_best.pt \
              --num_envs 12288 --max_iterations 500 --headless
            # checkpoint: logs/rsl_rl/go1_ball_juggle_hier/2026-04-07_20-18-34/model_best.pt
Result:     Training time 1683s (~28 min).  Stage transitions: A→B (~9 iters), B→C (+35 iters),
            C→D (+35 iters). Spent remaining ~280 iters in Stage D.
            At iter 499 (tbdump):
              Train/mean_episode_length  : 1500.0  ← MAXED OUT (1500 steps)
              Episode_Termination/time_out: 98.9%
              Episode_Termination/ball_below: 1.1%  (ball_off: 0%)
              Episode_Reward/ball_apex_height: 2.92  (needs 5.0 for Stage D→E advance)
              Episode_Reward/alive: 0.994
              Train/mean_reward: 78.5
              Policy/mean_noise_std: 0.337  (very converged)
              Perf/total_fps: 191 K env-steps/s (12288 envs much better GPU utilization)
            CONFIRMED: 53D pi2 was the culprit in iter_002, not env count.
            41D pi2 + 12288 envs gives BEST oracle baseline:
              iter_001 (41D, 4096): timeout=92.4%, mean_len=1470, Stage D
              iter_002 (53D, 12288): timeout=47%, mean_len=995, Stage C  ← worst
              iter_003 (41D, 12288): timeout=98.9%, mean_len=1500, Stage D ← BEST
            Ball is still not being actively juggled (apex_rew=2.92 vs 5.0 needed);
            the robot is maintaining the ball on the paddle at Stage D target height
            but not launching it to the apex. Policy is near local optimum.
Decision:   Oracle baseline is now solid. Proceed to fix_plan Task 3: write compare_pi1.py
            eval infrastructure. Also flag that longer training (>500 iters) may be needed
            to advance past Stage D — the apex reward is plateauing at ~3/5. Consider
            running 1000-iter warm-start from iter_003 checkpoint in a future iteration.

## iter_004 — integrate perception ball_obs_spec.py + smoke test d435i  (2026-04-08T03:50Z)
Hypothesis: Swapping the oracle obs terms for perception's `ball_pos_perceived` / `ball_vel_perceived` (in oracle mode) will produce identical training behavior, and d435i mode will run without errors.
Change:     (1) Copied perception's ball_obs_spec.py, ball_ekf.py, __init__.py to local branch.
            (2) Replaced ball_pos/ball_vel ObsTerms in ball_juggle_hier_env_cfg.py with
            `ball_pos_perceived` / `ball_vel_perceived` (default: oracle mode).
            (3) Added `--noise-mode oracle|d435i` CLI flag to train_juggle_hier.py.
            (4) Reinstalled go1_ball_balance package.
Command:    # Smoke test oracle:
            $C3R_BIN/gpu_lock.sh uv run --active python scripts/rsl_rl/train_juggle_hier.py \
              --task Isaac-BallJuggleHier-Go1-v0 \
              --pi2-checkpoint .../2026-03-12_09-04-32/model_best.pt \
              --num_envs 4 --max_iterations 2 --headless --noise-mode oracle
            # Smoke test d435i:
            same with --noise-mode d435i
Result:     Both smoke tests PASS (2 iterations each, no errors).
            Oracle: mean_reward iter 0→0.04, iter 1→1.59
            D435i:  mean_reward iter 0→0.98, iter 1→0.85
            Perception interface integrated successfully. Ready for full d435i training comparison.
Decision:   Next: run full 500-iter d435i training (same pi2, 12288 envs) to quantify degradation
            vs oracle baseline (iter_003: timeout=98.9%, apex=2.92). This is the core of
            the policy agent's mandate: measure noise impact and begin noise scheduling.
