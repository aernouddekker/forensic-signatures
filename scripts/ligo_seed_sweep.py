#!/usr/bin/env python3
"""
LIGO Seed Sweep: Bootstrap robustness check for stability metrics.

Demonstrates that the main findings (stable_delayed fraction, AUC, GMM agreement)
are stable under bootstrap resampling with different random seeds.

Approach:
1. Load all analyzed events from stability_events.jsonl
2. For each seed, bootstrap resample and compute metrics
3. Report median and range across seeds

This addresses the referee critique: "maybe your specific sample is unusual"
by showing the results are stable under resampling.

Outputs:
    - output/ligo_envelope/seed_sweep_results.json
    - figures/appendix/seed_sweep_stability.png

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, roc_auc_score

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "ligo_envelope"
FIGURES_DIR = SCRIPT_DIR.parent / "figures" / "appendix"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Seeds to test
SEEDS = [1, 7, 23, 42, 59, 97, 127, 256, 500, 999]


def load_stability_events():
    """Load analyzed events from stability_events.jsonl."""
    events_file = OUTPUT_DIR / "stability_events.jsonl"
    if not events_file.exists():
        raise FileNotFoundError(
            f"stability_events.jsonl not found. Run ligo_stability_figures.py first."
        )

    events = []
    with open(events_file) as f:
        for line in f:
            events.append(json.loads(line))

    # Filter to OK events (exclude failed)
    ok_events = [e for e in events if e.get('stability') != 'failed']
    print(f"Loaded {len(ok_events)} OK events from stability_events.jsonl")
    return ok_events


def compute_metrics(events):
    """Compute key metrics from a set of events."""
    n_ok = len(events)
    if n_ok == 0:
        return None

    n_stable_delayed = sum(1 for e in events if e['stability'] == 'stable_iof')
    n_stable_fast = sum(1 for e in events if e['stability'] == 'stable_std')
    n_flip = sum(1 for e in events if e['stability'] == 'flip')

    stable_delayed_frac = n_stable_delayed / n_ok

    # AUC for curvature (stable events only)
    stable_events = [e for e in events if e['stability'] in ('stable_iof', 'stable_std')]
    if len(stable_events) >= 20:
        y_true = np.array([1 if e['stability'] == 'stable_iof' else 0 for e in stable_events])
        curvatures = np.array([e['curvature_b'] for e in stable_events])

        # Check for valid data
        valid = np.isfinite(curvatures)
        y_true = y_true[valid]
        curvatures = curvatures[valid]

        if len(np.unique(y_true)) == 2 and len(y_true) >= 10:
            auc = roc_auc_score(y_true, curvatures)
        else:
            auc = None
    else:
        auc = None

    # GMM validation
    ok_curvatures = np.array([e['curvature_b'] for e in events])
    valid = np.isfinite(ok_curvatures)
    ok_curvatures = ok_curvatures[valid]
    ok_curvatures_scaled = ok_curvatures * 1000

    if len(ok_curvatures_scaled) >= 50:
        gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
        gmm.fit(ok_curvatures_scaled.reshape(-1, 1))
        gmm_labels = gmm.predict(ok_curvatures_scaled.reshape(-1, 1))

        # Map GMM labels to delayed/fast based on means
        means = gmm.means_.flatten()
        if means[0] < means[1]:
            fast_idx, delayed_idx = 0, 1
        else:
            fast_idx, delayed_idx = 1, 0

        gmm_is_delayed = (gmm_labels == delayed_idx)

        # Compare to model labels (stable only)
        events_valid = [e for e, v in zip(events, np.isfinite([e['curvature_b'] for e in events])) if v]
        stable_mask = np.array([e['stability'] in ('stable_iof', 'stable_std') for e in events_valid])
        model_is_delayed = np.array([e['stability'] == 'stable_iof' for e in events_valid])

        if stable_mask.sum() >= 20:
            stable_gmm = gmm_is_delayed[stable_mask]
            stable_model = model_is_delayed[stable_mask]
            gmm_agreement = np.mean(stable_gmm == stable_model)
            gmm_ari = adjusted_rand_score(stable_model, stable_gmm)
        else:
            gmm_agreement = None
            gmm_ari = None

        # GMM boundary
        lo, hi = sorted(means)
        b_range = np.linspace(ok_curvatures_scaled.min() - 1, ok_curvatures_scaled.max() + 1, 500)
        posteriors = gmm.predict_proba(b_range.reshape(-1, 1))
        diff = np.abs(posteriors[:, 0] - posteriors[:, 1])
        between_mask = (b_range >= lo) & (b_range <= hi)
        if between_mask.any():
            diff_between = diff[between_mask]
            if diff_between.min() < 0.1:
                gmm_boundary = b_range[between_mask][diff_between.argmin()]
            else:
                gmm_boundary = b_range[diff.argmin()]
        else:
            gmm_boundary = b_range[diff.argmin()]
    else:
        gmm_agreement = None
        gmm_ari = None
        gmm_boundary = None

    return {
        'n_ok': n_ok,
        'n_stable_delayed': n_stable_delayed,
        'n_stable_fast': n_stable_fast,
        'n_flip': n_flip,
        'stable_delayed_frac': stable_delayed_frac,
        'auc': auc,
        'gmm_agreement': gmm_agreement,
        'gmm_ari': gmm_ari,
        'gmm_boundary': gmm_boundary
    }


def run_seed_sweep():
    """Run the complete seed sweep."""
    print("=" * 60)
    print("LIGO Seed Sweep: Bootstrap Robustness Analysis")
    print("=" * 60)

    # Load events
    events = load_stability_events()
    n_events = len(events)

    # Compute original metrics (no resampling)
    print("\nOriginal sample metrics:")
    orig_metrics = compute_metrics(events)
    print(f"  Stable delayed: {orig_metrics['stable_delayed_frac']*100:.1f}%")
    print(f"  AUC: {orig_metrics['auc']:.3f}" if orig_metrics['auc'] else "  AUC: N/A")
    print(f"  GMM agreement: {orig_metrics['gmm_agreement']*100:.1f}%" if orig_metrics['gmm_agreement'] else "  GMM agreement: N/A")

    # Run bootstrap for each seed
    all_results = []
    for seed in SEEDS:
        np.random.seed(seed)

        # Bootstrap resample (with replacement)
        indices = np.random.choice(n_events, n_events, replace=True)
        resampled = [events[i] for i in indices]

        metrics = compute_metrics(resampled)
        if metrics:
            metrics['seed'] = seed
            all_results.append(metrics)
            auc_str = f"{metrics['auc']:.3f}" if metrics['auc'] else 'N/A'
            print(f"  Seed {seed:3d}: delayed={metrics['stable_delayed_frac']*100:.1f}%, AUC={auc_str}")

    if not all_results:
        print("ERROR: No valid results")
        return

    # Compute summary statistics
    delayed_fracs = [r['stable_delayed_frac'] for r in all_results]
    aucs = [r['auc'] for r in all_results if r['auc'] is not None]
    agreements = [r['gmm_agreement'] for r in all_results if r['gmm_agreement'] is not None]
    boundaries = [r['gmm_boundary'] for r in all_results if r['gmm_boundary'] is not None]

    summary = {
        'method': 'bootstrap_resampling',
        'seeds_tested': SEEDS,
        'n_seeds': len(all_results),
        'n_events': n_events,
        'original_metrics': orig_metrics,
        'stable_delayed_frac': {
            'original': orig_metrics['stable_delayed_frac'],
            'median': float(np.median(delayed_fracs)),
            'min': float(np.min(delayed_fracs)),
            'max': float(np.max(delayed_fracs)),
            'std': float(np.std(delayed_fracs)),
            'values': delayed_fracs
        },
        'auc': {
            'original': orig_metrics['auc'],
            'median': float(np.median(aucs)) if aucs else None,
            'min': float(np.min(aucs)) if aucs else None,
            'max': float(np.max(aucs)) if aucs else None,
            'std': float(np.std(aucs)) if aucs else None,
            'values': aucs
        },
        'gmm_agreement': {
            'original': orig_metrics['gmm_agreement'],
            'median': float(np.median(agreements)) if agreements else None,
            'min': float(np.min(agreements)) if agreements else None,
            'max': float(np.max(agreements)) if agreements else None,
            'std': float(np.std(agreements)) if agreements else None,
            'values': agreements
        },
        'gmm_boundary': {
            'original': orig_metrics['gmm_boundary'],
            'median': float(np.median(boundaries)) if boundaries else None,
            'min': float(np.min(boundaries)) if boundaries else None,
            'max': float(np.max(boundaries)) if boundaries else None,
            'values': boundaries
        },
        'per_seed_results': all_results,
        'note': 'Bootstrap resampling demonstrates metric stability under resampling'
    }

    # Save results
    json_path = OUTPUT_DIR / "seed_sweep_results.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Generate figure
    print("\nGenerating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    seed_labels = [str(s) for s in SEEDS[:len(all_results)]]

    # Panel 1: Stable delayed fraction
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(delayed_fracs)), [f * 100 for f in delayed_fracs],
                    color='C0', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axhline(orig_metrics['stable_delayed_frac'] * 100, color='red', linestyle='--',
                linewidth=2, label=f"Original: {orig_metrics['stable_delayed_frac']*100:.1f}%")
    ax1.set_xticks(range(len(seed_labels)))
    ax1.set_xticklabels(seed_labels, rotation=45, ha='right')
    ax1.set_xlabel('Bootstrap seed', fontsize=11)
    ax1.set_ylabel('Stable delayed (%)', fontsize=11)
    ax1.set_title('Delayed fraction stability', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)

    # Panel 2: AUC
    ax2 = axes[1]
    if aucs:
        ax2.bar(range(len(aucs)), aucs, color='C1', alpha=0.7,
                edgecolor='black', linewidth=0.5)
        if orig_metrics['auc']:
            ax2.axhline(orig_metrics['auc'], color='red', linestyle='--',
                        linewidth=2, label=f"Original: {orig_metrics['auc']:.3f}")
        ax2.set_xticks(range(len(aucs)))
        ax2.set_xticklabels(seed_labels[:len(aucs)], rotation=45, ha='right')
        ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlabel('Bootstrap seed', fontsize=11)
    ax2.set_ylabel('AUC (curvature)', fontsize=11)
    ax2.set_title('Curvature AUC stability', fontsize=12)
    ax2.set_ylim(0.9, 1.0)  # Focus on high-AUC region

    # Panel 3: GMM Agreement
    ax3 = axes[2]
    if agreements:
        ax3.bar(range(len(agreements)), [a * 100 for a in agreements],
                color='C2', alpha=0.7, edgecolor='black', linewidth=0.5)
        if orig_metrics['gmm_agreement']:
            ax3.axhline(orig_metrics['gmm_agreement'] * 100, color='red', linestyle='--',
                        linewidth=2, label=f"Original: {orig_metrics['gmm_agreement']*100:.1f}%")
        ax3.set_xticks(range(len(agreements)))
        ax3.set_xticklabels(seed_labels[:len(agreements)], rotation=45, ha='right')
        ax3.legend(loc='lower right', fontsize=9)
    ax3.set_xlabel('Bootstrap seed', fontsize=11)
    ax3.set_ylabel('GMM agreement (%)', fontsize=11)
    ax3.set_title('GMM-label agreement stability', fontsize=12)
    ax3.set_ylim(80, 95)  # Focus on relevant range

    plt.tight_layout()

    fig_path = FIGURES_DIR / "seed_sweep_stability.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BOOTSTRAP ROBUSTNESS SUMMARY")
    print("=" * 60)
    print(f"Seeds tested: {SEEDS}")
    print(f"Events per resample: {n_events}")
    print(f"\nStable delayed fraction:")
    print(f"  Original: {orig_metrics['stable_delayed_frac']*100:.1f}%")
    print(f"  Bootstrap median: {np.median(delayed_fracs)*100:.1f}%")
    print(f"  Bootstrap range:  [{np.min(delayed_fracs)*100:.1f}%, {np.max(delayed_fracs)*100:.1f}%]")
    print(f"  Bootstrap std:    {np.std(delayed_fracs)*100:.2f}%")
    if aucs:
        print(f"\nCurvature AUC:")
        print(f"  Original: {orig_metrics['auc']:.3f}" if orig_metrics['auc'] else "  Original: N/A")
        print(f"  Bootstrap median: {np.median(aucs):.3f}")
        print(f"  Bootstrap range:  [{np.min(aucs):.3f}, {np.max(aucs):.3f}]")
    if agreements:
        print(f"\nGMM agreement:")
        print(f"  Original: {orig_metrics['gmm_agreement']*100:.1f}%" if orig_metrics['gmm_agreement'] else "  Original: N/A")
        print(f"  Bootstrap median: {np.median(agreements)*100:.1f}%")
        print(f"  Bootstrap range:  [{np.min(agreements)*100:.1f}%, {np.max(agreements)*100:.1f}%]")

    print("\n" + "=" * 60)
    print("Seed sweep complete")
    print("=" * 60)


if __name__ == "__main__":
    run_seed_sweep()
