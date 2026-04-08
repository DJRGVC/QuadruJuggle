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
