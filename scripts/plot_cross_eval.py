#!/usr/bin/env python3
"""Plot cross-eval comparison: oracle vs d435i across target heights."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

targets = [0.10, 0.20, 0.30, 0.42, 0.50]

# Oracle-trained, oracle obs
oracle_timeout = [100.0, 100.0, 100.0, 100.0, 100.0]
oracle_apex    = [7.79, 4.81, 2.66, 1.74, 1.44]
oracle_len     = [1500, 1500, 1500, 1500, 1500]

# D435i-trained, d435i obs
d435i_timeout  = [0.0, 0.0, 40.0, 73.3, 70.0]
d435i_apex     = [10.33, 6.49, 4.28, 2.92, 2.43]
d435i_len      = [309.7, 439.3, 886.0, 1186.6, 1219.0]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

x = np.arange(len(targets))
w = 0.35

# Timeout %
ax = axes[0]
ax.bar(x - w/2, oracle_timeout, w, label="Oracle", color="#2196F3", alpha=0.85)
ax.bar(x + w/2, d435i_timeout, w, label="D435i", color="#FF9800", alpha=0.85)
ax.set_ylabel("Timeout %")
ax.set_xlabel("Target height (m)")
ax.set_xticks(x)
ax.set_xticklabels([f"{t:.2f}" for t in targets])
ax.set_ylim(0, 110)
ax.legend()
ax.set_title("Episode Survival")

# Apex reward
ax = axes[1]
ax.bar(x - w/2, oracle_apex, w, label="Oracle", color="#2196F3", alpha=0.85)
ax.bar(x + w/2, d435i_apex, w, label="D435i", color="#FF9800", alpha=0.85)
ax.set_ylabel("Apex Reward (per step)")
ax.set_xlabel("Target height (m)")
ax.set_xticks(x)
ax.set_xticklabels([f"{t:.2f}" for t in targets])
ax.legend()
ax.set_title("Apex Height Accuracy")

# Mean episode length
ax = axes[2]
ax.bar(x - w/2, oracle_len, w, label="Oracle", color="#2196F3", alpha=0.85)
ax.bar(x + w/2, d435i_len, w, label="D435i", color="#FF9800", alpha=0.85)
ax.set_ylabel("Mean Episode Length (steps)")
ax.set_xlabel("Target height (m)")
ax.set_xticks(x)
ax.set_xticklabels([f"{t:.2f}" for t in targets])
ax.legend()
ax.set_title("Episode Duration")

fig.suptitle("Cross-Eval: Oracle vs D435i Policy (corrected iter 29)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("images/policy/cross_eval_iter029.png", dpi=150, bbox_inches="tight")
print("Saved images/policy/cross_eval_iter029.png")
