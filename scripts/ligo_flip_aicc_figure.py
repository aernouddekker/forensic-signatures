#!/usr/bin/env python3
"""
LIGO Flip AICc Distribution Figure
===================================

Generates figure showing AICc gap distribution by stability class.
Demonstrates that flip events cluster at small |ΔAICc| (genuine ambiguity),
while stable events show larger gaps (confident classification).

This addresses the referee critique: "flip = windowing artifact"

The figure shows:
- Left: Boxplots of |ΔAICc| by stability class
- Right: Density distributions showing flip population clusters at low evidence

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "ligo_envelope"
FIGURES_DIR = SCRIPT_DIR.parent / "figures" / "ligo"
APPENDIX_FIGURES_DIR = SCRIPT_DIR.parent / "figures" / "appendix"

DETAILED_FILE = OUTPUT_DIR / "stability_events_detailed.jsonl"
STABILITY_FILE = OUTPUT_DIR / "stability_events.jsonl"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
APPENDIX_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_events():
    """Load stability classifications and detailed window results."""
    # Load stability classifications
    stability = {}
    with open(STABILITY_FILE) as f:
        for line in f:
            e = json.loads(line)
            stability[e['gps_time']] = e['stability']

    # Load detailed window results
    events = []
    with open(DETAILED_FILE) as f:
        for line in f:
            e = json.loads(line)
            e['stability'] = stability.get(e['gps_time'], 'unknown')
            events.append(e)

    return events


def compute_min_aicc_gap(event):
    """
    Compute minimum |ΔAICc| across windows.

    This represents the "weakest link" - the window where the model
    selection was least confident. Flip events should have at least
    one window with low |ΔAICc|.
    """
    gaps = []
    for w, result in event['window_results'].items():
        delta = result.get('delta_aicc')
        if delta is not None:
            gaps.append(abs(delta))
    return min(gaps) if gaps else None


def compute_mean_aicc_gap(event):
    """Compute mean |ΔAICc| across windows."""
    gaps = []
    for w, result in event['window_results'].items():
        delta = result.get('delta_aicc')
        if delta is not None:
            gaps.append(abs(delta))
    return np.mean(gaps) if gaps else None


def main():
    print("=" * 70)
    print("LIGO Flip AICc Distribution Figure")
    print("=" * 70)

    # Load events
    events = load_events()
    print(f"Loaded {len(events)} events")

    # Separate by stability
    stable_iof = [e for e in events if e['stability'] == 'stable_iof']
    stable_std = [e for e in events if e['stability'] == 'stable_std']
    flip = [e for e in events if e['stability'] == 'flip']

    print(f"  Stable Delayed (IOF): {len(stable_iof)}")
    print(f"  Stable Fast (STD): {len(stable_std)}")
    print(f"  Flip: {len(flip)}")

    # Compute min |ΔAICc| for each event
    aicc_iof = [compute_min_aicc_gap(e) for e in stable_iof if compute_min_aicc_gap(e) is not None]
    aicc_std = [compute_min_aicc_gap(e) for e in stable_std if compute_min_aicc_gap(e) is not None]
    aicc_flip = [compute_min_aicc_gap(e) for e in flip if compute_min_aicc_gap(e) is not None]

    print(f"\nMin |ΔAICc| statistics:")
    print(f"  Stable Delayed: median = {np.median(aicc_iof):.1f}, IQR = [{np.percentile(aicc_iof, 25):.1f}, {np.percentile(aicc_iof, 75):.1f}]")
    print(f"  Stable Fast: median = {np.median(aicc_std):.1f}, IQR = [{np.percentile(aicc_std, 25):.1f}, {np.percentile(aicc_std, 75):.1f}]")
    print(f"  Flip: median = {np.median(aicc_flip):.1f}, IQR = [{np.percentile(aicc_flip, 25):.1f}, {np.percentile(aicc_flip, 75):.1f}]")

    # Count events below thresholds
    threshold_4 = 4
    threshold_10 = 10
    pct_iof_below_4 = 100 * sum(1 for x in aicc_iof if x < threshold_4) / len(aicc_iof)
    pct_std_below_4 = 100 * sum(1 for x in aicc_std if x < threshold_4) / len(aicc_std)
    pct_flip_below_4 = 100 * sum(1 for x in aicc_flip if x < threshold_4) / len(aicc_flip)

    print(f"\nFraction with min |ΔAICc| < {threshold_4}:")
    print(f"  Stable Delayed: {pct_iof_below_4:.1f}%")
    print(f"  Stable Fast: {pct_std_below_4:.1f}%")
    print(f"  Flip: {pct_flip_below_4:.1f}%")

    # Generate figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Color scheme
    colors = {
        'std': '#2ecc71',  # Green for fast
        'flip': '#f39c12',  # Orange for flip
        'iof': '#e74c3c',  # Red for delayed
    }

    # --- Left: Boxplot ---
    ax1 = axes[0]

    # Clip for visualization (long tails compress the plot)
    clip_val = 100
    aicc_iof_clip = [min(x, clip_val) for x in aicc_iof]
    aicc_std_clip = [min(x, clip_val) for x in aicc_std]
    aicc_flip_clip = [min(x, clip_val) for x in aicc_flip]

    data = [aicc_std_clip, aicc_flip_clip, aicc_iof_clip]
    labels = [f'Stable Fast\n(n={len(aicc_std)})',
              f'Flip\n(n={len(aicc_flip)})',
              f'Stable Delayed\n(n={len(aicc_iof)})']

    bp = ax1.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], [colors['std'], colors['flip'], colors['iof']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add threshold lines
    ax1.axhline(y=4, color='gray', linestyle='--', alpha=0.7, label='|ΔAICc| = 4 (strong)')
    ax1.axhline(y=10, color='gray', linestyle=':', alpha=0.7, label='|ΔAICc| = 10 (very strong)')

    ax1.set_ylabel('Minimum |ΔAICc| across windows', fontsize=12)
    ax1.set_title('Evidence Strength by Stability Class', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, clip_val + 10)

    # --- Right: Density plot ---
    ax2 = axes[1]

    # Use log scale for x-axis to handle long tails
    # Focus on the low-evidence region
    x_max = 50

    for arr, color, label in [(aicc_std, colors['std'], 'Stable Fast'),
                               (aicc_flip, colors['flip'], 'Flip'),
                               (aicc_iof, colors['iof'], 'Stable Delayed')]:
        arr_clip = [x for x in arr if x < x_max]
        if len(arr_clip) > 5:
            try:
                kde = gaussian_kde(arr_clip, bw_method=0.3)
                x = np.linspace(0, x_max, 200)
                ax2.fill_between(x, kde(x), alpha=0.3, color=color, label=label)
                ax2.plot(x, kde(x), color=color, linewidth=2)
            except:
                pass

    # Add threshold lines
    ax2.axvline(x=4, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(x=10, color='gray', linestyle=':', alpha=0.7)

    # Annotate thresholds
    ax2.text(4.5, ax2.get_ylim()[1] * 0.9, 'Strong\nevidence', fontsize=9, va='top')
    ax2.text(10.5, ax2.get_ylim()[1] * 0.9, 'Very strong', fontsize=9, va='top')

    ax2.set_xlabel('Minimum |ΔAICc| across windows', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Flip Events Cluster in Low-Evidence Regime', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, x_max)

    plt.tight_layout()

    # Save to both locations
    for path in [FIGURES_DIR / "flip_aicc_gap_distribution.png",
                 APPENDIX_FIGURES_DIR / "flip_aicc_gap_distribution.png",
                 OUTPUT_DIR / "flip_aicc_gap_distribution.png"]:
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    plt.close()

    # Save summary statistics
    summary = {
        'n_stable_iof': len(aicc_iof),
        'n_stable_std': len(aicc_std),
        'n_flip': len(aicc_flip),
        'stable_iof_median': float(np.median(aicc_iof)),
        'stable_std_median': float(np.median(aicc_std)),
        'flip_median': float(np.median(aicc_flip)),
        'stable_iof_iqr': [float(np.percentile(aicc_iof, 25)), float(np.percentile(aicc_iof, 75))],
        'stable_std_iqr': [float(np.percentile(aicc_std, 25)), float(np.percentile(aicc_std, 75))],
        'flip_iqr': [float(np.percentile(aicc_flip, 25)), float(np.percentile(aicc_flip, 75))],
        'pct_iof_below_4': pct_iof_below_4,
        'pct_std_below_4': pct_std_below_4,
        'pct_flip_below_4': pct_flip_below_4,
    }

    with open(OUTPUT_DIR / "flip_aicc_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'flip_aicc_summary.json'}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
