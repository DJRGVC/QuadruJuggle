"""Plot D435i noise model characterization across ball heights.

Produces a 4-panel figure showing how position noise, velocity noise,
dropout rate, and detection probability vary with ball-camera distance.
Highlights the difference between old (fixed nominal) and new
(height-dependent) velocity noise models.
"""

import os
import sys
import importlib.util

import numpy as np
import matplotlib.pyplot as plt

# Load perception module without Isaac Lab
class _StubModule:
    def __getattr__(self, name): return _StubModule()
    def __call__(self, *a, **kw): return _StubModule()

for mod_name in [
    "isaaclab", "isaaclab.utils", "isaaclab.utils.math",
    "isaaclab.assets", "isaaclab.managers", "isaaclab.envs",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _StubModule()

_PERC_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance", "go1_ball_balance", "perception",
)

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_load_module("perception.ball_ekf", os.path.join(_PERC_DIR, "ball_ekf.py"))
_load_module("perception.noise_model", os.path.join(_PERC_DIR, "noise_model.py"))
obs_mod = _load_module("perception.ball_obs_spec", os.path.join(_PERC_DIR, "ball_obs_spec.py"))

D435iNoiseParams = obs_mod.D435iNoiseParams


def main():
    params = D435iNoiseParams()
    z = np.linspace(0.01, 1.5, 200)
    dt_camera = 1.0 / 30.0

    # Position noise
    sigma_xy = np.maximum(params.sigma_xy_per_metre * z, params.sigma_xy_floor)
    sigma_z = params.sigma_z_base + params.sigma_z_quadratic * z ** 2

    # Velocity noise (height-dependent)
    sigma_vel_xy = np.sqrt(2) * sigma_xy / dt_camera
    sigma_vel_z = np.sqrt(2) * sigma_z / dt_camera

    # Old velocity noise (fixed nominal z=0.5)
    old_sigma_xy = max(params.sigma_xy_floor, params.sigma_xy_per_metre * 0.5)
    old_sigma_z = params.sigma_z_base + params.sigma_z_quadratic * 0.25
    old_vel_xy = np.sqrt(2) * old_sigma_xy / dt_camera * np.ones_like(z)
    old_vel_z = np.sqrt(2) * old_sigma_z / dt_camera * np.ones_like(z)

    # Dropout
    z_excess = np.maximum(z - 0.5, 0.0)
    p_dropout = params.dropout_base + params.dropout_range * (
        1.0 - np.exp(-z_excess / params.dropout_scale)
    )

    # Stage markers
    stage_heights = {
        "A (0.10m)": 0.10,
        "C (0.30m)": 0.30,
        "E (0.60m)": 0.60,
        "G (1.00m)": 1.00,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("D435i Noise Model — Height Characterization", fontsize=14, fontweight="bold")

    # (a) Position noise
    ax = axes[0, 0]
    ax.plot(z * 1000, sigma_xy * 1000, label="σ_xy (linear)", color="C0")
    ax.plot(z * 1000, sigma_z * 1000, label="σ_z (quadratic)", color="C1")
    for name, h in stage_heights.items():
        ax.axvline(h * 1000, color="gray", ls="--", alpha=0.5, lw=0.8)
        ax.text(h * 1000 + 10, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5,
                name, fontsize=7, rotation=90, va="top")
    ax.set_xlabel("Ball height above paddle (mm)")
    ax.set_ylabel("Position noise σ (mm)")
    ax.set_title("(a) Position noise vs height")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Velocity noise — old vs new
    ax = axes[0, 1]
    ax.plot(z * 1000, sigma_vel_xy, label="σ_vel_xy (new, height-dep)", color="C0")
    ax.plot(z * 1000, sigma_vel_z, label="σ_vel_z (new, height-dep)", color="C1")
    ax.plot(z * 1000, old_vel_xy, label="σ_vel_xy (old, z=0.5m fixed)", color="C0", ls="--", alpha=0.5)
    ax.plot(z * 1000, old_vel_z, label="σ_vel_z (old, z=0.5m fixed)", color="C1", ls="--", alpha=0.5)
    for name, h in stage_heights.items():
        ax.axvline(h * 1000, color="gray", ls="--", alpha=0.5, lw=0.8)
    ax.set_xlabel("Ball height above paddle (mm)")
    ax.set_ylabel("Velocity noise σ (m/s)")
    ax.set_title("(b) Velocity noise — old (dashed) vs new (solid)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Dropout rate
    ax = axes[1, 0]
    ax.plot(z * 1000, p_dropout * 100, color="C3")
    ax.fill_between(z * 1000, 0, p_dropout * 100, alpha=0.2, color="C3")
    for name, h in stage_heights.items():
        ax.axvline(h * 1000, color="gray", ls="--", alpha=0.5, lw=0.8)
    ax.set_xlabel("Ball height above paddle (mm)")
    ax.set_ylabel("Dropout probability (%)")
    ax.set_title("(c) Detection dropout vs height")
    ax.grid(True, alpha=0.3)

    # (d) Vel noise ratio (new/old)
    ax = axes[1, 1]
    ratio_xy = sigma_vel_xy / old_vel_xy
    ratio_z = sigma_vel_z / old_vel_z
    ax.plot(z * 1000, ratio_xy, label="XY ratio", color="C0")
    ax.plot(z * 1000, ratio_z, label="Z ratio", color="C1")
    ax.axhline(1.0, color="gray", ls="-", alpha=0.5)
    for name, h in stage_heights.items():
        ax.axvline(h * 1000, color="gray", ls="--", alpha=0.5, lw=0.8)
    ax.set_xlabel("Ball height above paddle (mm)")
    ax.set_ylabel("Noise ratio (new / old)")
    ax.set_title("(d) Velocity noise correction factor")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Annotate key heights
    for name, h in stage_heights.items():
        rz = np.interp(h * 1000, z * 1000, ratio_z)
        ax.annotate(f"{rz:.1f}×", xy=(h * 1000, rz), fontsize=8,
                    textcoords="offset points", xytext=(8, 5))

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "images", "perception")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "noise_vs_height_iter111.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
