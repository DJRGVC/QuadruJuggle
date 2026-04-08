# fix_plan.md — experiment queue for policy
#
# Active tasks and forward-looking work only. Completed tasks archived.

## ACTIVE — Break apex plateau via curriculum restructuring

- [x] Diagnose apex≈10.7 plateau: ROOT CAUSE = Stage P [0.30,1.00] range too wide.
      Policy already at Stage P (final). ES patience already 1500. ball_release_vel already 8.0.
- [ ] Narrow Stage P: [0.30, 1.00] → [0.30, 0.60]. Add stages Q-T for wider ranges:
      Q=[0.30,0.70], R=[0.30,0.80], S=[0.30,0.90], T=[0.30,1.00].
      This gives policy time to learn multi-bounce energy injection gradually.
- [ ] Resume training from best checkpoint with restructured curriculum
- [ ] Monitor: does narrower final stage improve apex accuracy above 43%?
- [ ] If plateau persists after restructuring: investigate multi-bounce timing with
      frank-dev mirror law insights (h_dot impulse pattern)

## NEXT — Noise robustness validation

- [ ] Once juggling advances past Stage B: oracle vs d435i comparison on active juggling
- [ ] Full noise-curriculum run to Stage G with working juggling reward
- [ ] Cross-eval matrix: noise-trained vs oracle-trained, tested with both obs types

## USER COMMAND INTERFACE (Daniel request 2026-04-08)

- [ ] Implement WASD velocity + P/L height controls in play.py (or play_teleop.py)
      - WASD: user velocity commands (vx, vy) passed to pi1 or pi2
      - P/L: target apex height up/down
      - Terminal readout showing current command values
      - Integrate with perception's ResidualMixer for velocity (git show agent/perception:source/.../vel_cmd/residual_mixer.py)
      - Pi1 already has target_apex_height obs[39] — just expose as user input
- [ ] Research: how to compose user height commands with curriculum target — override vs. additive?

## INFRASTRUCTURE

- [ ] Run play.py with model_best.pt or model_4249.pt — capture video to confirm juggling visually

## COMPLETED (archived in RESEARCH_LOG_ARCHIVE.md)
# iters 001-003: oracle baseline (41D pi2, 12288 envs, Stage D, timeout=98.9%)
# iters 004-005: perception noise integrated, d435i comparison (~8% degradation)
# iters 006-008: noise curriculum, Stage D plateau broken, noise outperforms oracle
# iters 009-010: root cause chain (sigma_ratio + alive dominance)
# iter_011: ball_low_penalty breaks balance (apex 13.7 peak) but collapses
# iter_012: compaction
# iter_013: ball_low=-2.0 death spiral; sustain-during-blend bugfix
# iter_014: ball_release_velocity_reward SUSTAINS JUGGLING (apex 9.7 stable)
# iter_015: curriculum threshold 0.75→0.30 — CURRICULUM ADVANCES, juggling sustained
# iter_016: continued training → plateau at apex≈10.7, ES triggered
# iter_017: compaction (summarized iters 009-014)
