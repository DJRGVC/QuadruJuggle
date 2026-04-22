"""Export π2 (torso-tracking policy) to ONNX and verify it matches PyTorch output.

Usage:
    python scripts/tests/export_pi2_onnx.py
    python scripts/tests/export_pi2_onnx.py --checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt
    python scripts/tests/export_pi2_onnx.py --checkpoint <path> --out sim_to_real/unitree_bringup/config/go1/pi2.onnx

What this does:
    1. Loads the π2 checkpoint (RSL-RL format)
    2. Rebuilds the actor MLP (same as play_mujoco.py)
    3. Exports to ONNX with input name "obs" (shape [1, 39]) and output "actions" (shape [1, 12])
    4. Runs 5 random obs through both PyTorch and ONNX and checks they match
    5. Prints a summary of input/output shapes and max error
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__),
    "../../logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt"
)
DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__),
    "../../sim_to_real/unitree_bringup/config/go1/pi2.onnx"
)

# π2 observation dimension (must match build_pi2_obs in play_mujoco.py)
PI2_OBS_DIM = 39
# π2 action dimension (12 joints)
PI2_ACT_DIM = 12


def build_actor(checkpoint_path: str) -> nn.Sequential:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # RSL-RL 5.x stores actor separately; 4.x puts it in model_state_dict
    if "actor_state_dict" in ckpt:
        sd, prefix = ckpt["actor_state_dict"], "mlp."
    else:
        sd, prefix = ckpt.get("model_state_dict", ckpt), "actor."

    weight_keys = sorted(k for k in sd if k.startswith(prefix) and k.endswith(".weight"))
    if not weight_keys:
        sys.exit(f"[export] No keys with prefix '{prefix}' found in checkpoint. "
                 f"Available prefixes: {set(k.split('.')[0] for k in sd)}")

    layers = [(sd[k].shape[1], sd[k].shape[0]) for k in weight_keys]
    print(f"[export] Actor layers (in→out): {layers}")

    modules = []
    for i, (in_dim, out_dim) in enumerate(layers):
        modules.append(nn.Linear(in_dim, out_dim))
        if i < len(layers) - 1:
            modules.append(nn.ELU())
    net = nn.Sequential(*modules)

    actor_sd = {}
    for k in weight_keys:
        idx = int(k[len(prefix):].split(".")[0])
        actor_sd[f"{idx}.weight"] = sd[k]
        bkey = k.replace(".weight", ".bias")
        if bkey in sd:
            actor_sd[f"{idx}.bias"] = sd[bkey]
    net.load_state_dict(actor_sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def export_onnx(net: nn.Sequential, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dummy = torch.zeros(1, PI2_OBS_DIM)
    torch.onnx.export(
        net,
        dummy,
        out_path,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
    )
    print(f"[export] Saved ONNX → {out_path}")


def verify(net: nn.Sequential, onnx_path: str, n_samples: int = 5) -> None:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print(f"\n[verify] ONNX inputs : {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"[verify] ONNX outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")
    print()

    max_err = 0.0
    for i in range(n_samples):
        obs_np = np.random.randn(1, PI2_OBS_DIM).astype(np.float32)
        obs_t  = torch.from_numpy(obs_np)

        with torch.no_grad():
            pt_out = net(obs_t).numpy()

        ort_out = sess.run(["actions"], {"obs": obs_np})[0]

        err = np.abs(pt_out - ort_out).max()
        max_err = max(max_err, err)
        print(f"  sample {i+1}: pytorch={pt_out[0, :3].round(4)}...  "
              f"onnx={ort_out[0, :3].round(4)}...  max_err={err:.2e}")

    print()
    if max_err < 1e-4:
        print(f"[verify] PASS — max absolute error {max_err:.2e} (< 1e-4, normal float32 drift)")
    else:
        print(f"[verify] WARN — max absolute error {max_err:.2e} is larger than expected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="Path to RSL-RL .pt checkpoint")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="Output .onnx path")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"[export] Checkpoint not found: {args.checkpoint}")

    print(f"[export] Loading checkpoint: {args.checkpoint}")
    net = build_actor(args.checkpoint)

    in_dim = net[0].in_features
    if in_dim != PI2_OBS_DIM:
        print(f"[export] WARNING: actor input dim is {in_dim}, expected {PI2_OBS_DIM}. "
              f"Update PI2_OBS_DIM at top of this script if your obs space changed.")

    export_onnx(net, args.out)
    verify(net, args.out)


if __name__ == "__main__":
    main()
