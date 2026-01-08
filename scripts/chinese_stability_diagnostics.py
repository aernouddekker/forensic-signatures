#!/usr/bin/env python3
"""
Chinese Stability Diagnostics (Evidence Strength)
=================================================

Reads chinese_robustness_sweep outputs and summarizes evidence strength by
stability class, mirroring the Google narrative:

- Stable IOF should concentrate at higher |ΔAICc|
- Flips should cluster at low |ΔAICc| if ambiguity is real

Outputs:
  scripts/output/chinese_robustness_sweep/evidence_strength.json
  scripts/figures/chinese/chinese_evidence_strength.png

Author: Aernoud Dekker
Date: Dec 2025
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
SWEEP_DIR = SCRIPT_DIR / "output" / "chinese_robustness_sweep"
FIG_DIR = SCRIPT_DIR.parent / "figures" / "chinese"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STABILITY_PATH = SWEEP_DIR / "stability_table.json"
DETAILED_PATH = SWEEP_DIR / "detailed_results.json"


def main():
    print("=" * 70)
    print("Chinese Stability Diagnostics (Evidence Strength)")
    print("=" * 70)

    if not STABILITY_PATH.exists() or not DETAILED_PATH.exists():
        raise SystemExit("Missing sweep outputs. Run: python chinese_robustness_sweep.py")

    with open(STABILITY_PATH) as f:
        stab = json.load(f)
    with open(DETAILED_PATH) as f:
        detailed = json.load(f)

    event_details = stab["event_details"]

    # Collect per-event median |ΔAICc| across configs
    per_event_abs = defaultdict(list)

    for cfg, rows in detailed.items():
        for r in rows:
            per_event_abs[r["event_id"]].append(float(r["abs_delta_aicc"]))

    # Summarize by stability class
    by_class = {"stable_iof": [], "stable_std": [], "flip": [], "fluctuate_uncertain": []}
    for eid, info in event_details.items():
        klass = info["stability"]
        vals = per_event_abs.get(eid, [])
        if len(vals) == 0:
            continue
        by_class[klass].append(float(np.median(vals)))

    def stats(arr):
        if len(arr) == 0:
            return {"n": 0}
        a = np.array(arr, dtype=float)
        q1, q3 = np.quantile(a, [0.25, 0.75])
        return {
            "n": int(len(a)),
            "median": float(np.median(a)),
            "q1": float(q1),
            "q3": float(q3),
            "above_4": float(np.mean(a >= 4.0)),
            "above_10": float(np.mean(a >= 10.0)),
        }

    summary = {k: stats(v) for k, v in by_class.items()}
    out_json = SWEEP_DIR / "evidence_strength.json"
    with open(out_json, "w") as f:
        json.dump({"per_class": summary}, f, indent=2)

    print("\n=== Evidence strength by stability class ===")
    for k, s in summary.items():
        if s['n'] == 0:
            print(f"{k:20s} n=   0")
        else:
            print(f"{k:20s} n={s['n']:4d}  median={s.get('median', float('nan')):6.2f}  "
                  f"above4={100*s.get('above_4',0):5.1f}%  above10={100*s.get('above_10',0):5.1f}%")

    # Plot: simple jittered scatter by class
    labels = ["stable_iof", "stable_std", "flip"]
    display_labels = ["Stable IOF", "Stable STD", "Flip"]
    data = [by_class[l] for l in labels]

    plt.figure(figsize=(10, 6))

    # Jittered points
    colors = ['coral', 'lightgreen', 'lightgray']
    for i, (arr, color) in enumerate(zip(data, colors), start=1):
        if not arr:
            continue
        x = np.random.normal(i, 0.08, size=len(arr))
        plt.scatter(x, arr, s=25, alpha=0.6, c=color, edgecolors='gray', linewidths=0.5)

    # Evidence thresholds
    plt.axhline(4.0, color='blue', linestyle="--", alpha=0.6, label=r'$|\Delta\mathrm{AICc}| = 4$ (moderate)')
    plt.axhline(10.0, color='red', linestyle="--", alpha=0.6, label=r'$|\Delta\mathrm{AICc}| = 10$ (strong)')

    plt.xticks([1, 2, 3], display_labels)
    plt.ylabel(r"Per-event median $|\Delta \mathrm{AICc}|$")
    plt.xlabel("Stability Class")
    plt.title("Chinese 63-Qubit: Evidence Strength by Stability Class")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.25)

    # Add counts as text
    for i, (arr, label) in enumerate(zip(data, display_labels), start=1):
        n = len(arr)
        if n > 0:
            med = np.median(arr)
            plt.text(i, plt.ylim()[1] * 0.95, f'n={n}', ha='center', fontsize=10)

    out_fig = FIG_DIR / "chinese_evidence_strength.png"
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    plt.close()

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_fig}")

    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
