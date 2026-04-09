"""Parse cross-eval log files and produce a comparison table + plot."""
import re
import sys
import os

def parse_eval_log(path: str) -> list[dict]:
    """Extract per-target results from an eval log file."""
    results = []
    with open(path) as f:
        for line in f:
            # Match lines like:   0.30  0.120    54.3%      1.23     342.1        30
            m = re.match(
                r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([-\d.]+)\s+([\d.]+)\s+(\d+)', line
            )
            if m:
                results.append({
                    'target': float(m.group(1)),
                    'sigma': float(m.group(2)),
                    'timeout_pct': float(m.group(3)),
                    'apex_rew': float(m.group(4)),
                    'mean_len': float(m.group(5)),
                    'episodes': int(m.group(6)),
                })
    return results


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "experiments/iter_028_cross_eval"

    combos = [
        ("d435i_trained__oracle_obs", "D435i → Oracle"),
        ("d435i_trained__d435i_obs", "D435i → D435i"),
        ("oracle_trained__oracle_obs", "Oracle → Oracle"),
        ("oracle_trained__d435i_obs", "Oracle → D435i"),
    ]

    print(f"\n{'='*90}")
    print(f"  CROSS-EVAL COMPARISON TABLE")
    print(f"{'='*90}")

    all_data = {}
    for fname, label in combos:
        path = os.path.join(outdir, f"{fname}.log")
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        data = parse_eval_log(path)
        all_data[label] = data

        if data:
            avg_timeout = sum(d['timeout_pct'] for d in data) / len(data)
            avg_apex = sum(d['apex_rew'] for d in data) / len(data)
            avg_len = sum(d['mean_len'] for d in data) / len(data)
            print(f"\n  {label}:")
            print(f"    Avg timeout: {avg_timeout:.1f}%  |  Avg apex reward: {avg_apex:.2f}  |  Avg ep length: {avg_len:.1f}")
            for d in data:
                print(f"    target={d['target']:.2f}  timeout={d['timeout_pct']:.1f}%  apex={d['apex_rew']:.2f}  len={d['mean_len']:.1f}")

    # Key comparison: noise robustness
    if "Oracle → Oracle" in all_data and "Oracle → D435i" in all_data:
        oo = all_data["Oracle → Oracle"]
        od = all_data["Oracle → D435i"]
        if oo and od:
            oo_apex = sum(d['apex_rew'] for d in oo) / len(oo)
            od_apex = sum(d['apex_rew'] for d in od) / len(od)
            drop = (oo_apex - od_apex) / max(abs(oo_apex), 1e-6) * 100
            print(f"\n  Oracle degradation under noise: {drop:+.1f}% apex reward")

    if "D435i → Oracle" in all_data and "D435i → D435i" in all_data:
        do = all_data["D435i → Oracle"]
        dd = all_data["D435i → D435i"]
        if do and dd:
            do_apex = sum(d['apex_rew'] for d in do) / len(do)
            dd_apex = sum(d['apex_rew'] for d in dd) / len(dd)
            diff = (do_apex - dd_apex) / max(abs(dd_apex), 1e-6) * 100
            print(f"  D435i transfer to oracle obs:   {diff:+.1f}% apex reward")

    print()


if __name__ == "__main__":
    main()
