# Report Text Additions (2026-03-30)

## 1) Core Architecture (CRITICAL wording fix)

The system uses a hierarchical control architecture where a high-level controller (π1) outputs a 6D torso command, and a low-level controller (π2) tracks this command.

The interface between π1 and π2 is:

\[h, \dot h, \mathrm{roll}, \mathrm{pitch}, \omega_{\mathrm{roll}}, \omega_{\mathrm{pitch}}\]

π2 is a torso pose tracking policy, not a velocity controller. This distinction is critical: the high-level controller directly specifies paddle orientation and vertical motion, enabling physically grounded control via reflection geometry.

In addition to the analytic controller, we trained a learned π1 using PPO on top of the frozen π2, enabling a direct comparison between model-based and learned high-level control.

---

## 2) Key Contributions

1. **Hybrid analytic + learned control architecture**  
   Combines a physics-based mirror law controller (π1) with a learned torso tracking policy (π2).  
   Enables interpretable control while retaining learned locomotion robustness.

2. **Direct comparison: analytic vs. learned high-level control**  
   Trained a PPO-based π1 on top of frozen π2.  
   Demonstrates trade-off: RL π1 gives higher peak performance; mirror-law π1 gives stronger robustness and generalization.

3. **Noise-aware control pipeline**  
   Explicitly studies velocity noise impact on control (Fig. 1).  
   Demonstrates failure mode of learned policies outside training distribution.

4. **End-to-end perception → control pipeline**  
   Uses position-only observation + Kalman filtering.  
   Bridges toward real-world stereo-based deployment.

---

## 3) Figure Text (copy-ready)

### Fig. 1 — Noise Sensitivity

Observation: The learned π1 exhibits sharp degradation once velocity noise exceeds the training distribution (σ ≈ 0.1 m/s), while the mirror law degrades approximately linearly.

Interpretation: The mirror law is analytically grounded in reflection physics and does not rely on data-driven interpolation. In contrast, the learned policy implicitly fits the training distribution and fails to extrapolate beyond it.

Implication: For real-world deployment where velocity estimation is noisy, analytic control provides significantly stronger robustness.

### Fig. 2 — Sample Efficiency

Training π1 via reinforcement learning requires an additional 568M environment steps (74% increase) compared to the mirror-law approach.

Since π2 is shared, this cost is entirely attributable to learning the high-level controller.

This highlights a key advantage of analytic control: zero training cost for the high-level policy.

### Fig. 3 — Mirror Law Geometry

The mirror law computes the required paddle normal by enforcing reflection consistency:
- Incoming and outgoing velocities define the desired surface normal.
- This directly maps to roll/pitch commands.

Unlike learned policies, this mapping is deterministic, interpretable, and invariant to data distribution.

### Fig. 5 — Command Stability

Raw velocity noise produces high-frequency oscillations in commanded pitch (RMSE = 11°), which would destabilize the robot.

Applying exponential smoothing reduces RMSE to 4.7° (43% of raw), demonstrating that:
- Control instability is primarily driven by velocity noise.
- Simple filtering is sufficient to stabilize the analytic controller.

This further supports the separation of estimation and control.

### Fig. 6 — Kalman Filter (corrected interpretation)

While the Kalman filter improves position estimation, velocity estimation remains challenging due to contact discontinuities.

In practice:
- Finite differences produce lower RMSE in this setting.
- The Kalman filter provides smoother trajectories and robustness to missing data.

The filter is therefore retained for stability and real-world compatibility, rather than pure accuracy.

---

## 4) Results Narrative Tightening

### High-Level Comparison

| Method | Strength | Weakness |
|---|---|---|
| Mirror law π1 | Robust, interpretable, zero training | Slightly lower peak performance |
| Learned π1 | Higher performance | Poor generalization, expensive |

Key takeaway: The learned π1 improves performance within distribution, while the mirror law provides out-of-distribution robustness and eliminates training cost.

---

## 5) One Thesis-Level Insight

The key insight of this work is that the control structure should match the physics of the task.

Ball bouncing is governed by reflection geometry, which is low-dimensional and analytically solvable.

Learning this mapping via RL introduces unnecessary sample complexity and reduces robustness.

Instead, learning is reserved for π2, where the dynamics are high-dimensional and difficult to model analytically.

---

## 6) Observation Description Fix

π2 observes the 6D torso command produced by π1, not a desired base position.
