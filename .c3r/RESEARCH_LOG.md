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
