#!/usr/bin/env python3
"""
LIGO GMM Validation: Unsupervised corroboration of model-derived labels.

Fits a 2-component Gaussian Mixture Model to curvature index b (ignoring labels),
then compares GMM cluster assignments to model-derived stable-core labels.

This provides classifier-external validation: if GMM clusters align with
model-derived delayed/fast labels, the population structure is not an artifact
of the classification procedure.

Outputs:
    - output/ligo_envelope/unsupervised_validation.json (metrics)
    - figures/appendix/unsupervised_gmm_validation.png (figure)

Dependencies:
    - scikit-learn (for GaussianMixture)
    - stability_events.jsonl must exist (run ligo_glitch_analysis.py first)

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Paths - anchored to script directory (consistent with other pipeline scripts)
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "ligo_envelope"
FIGURES_DIR = SCRIPT_DIR.parent / "figures" / "appendix"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Random state for reproducibility
RNG_SEED = 42


def load_stability_events():
    """Load stability events from JSONL file."""
    events_file = OUTPUT_DIR / "stability_events.jsonl"
    if not events_file.exists():
        raise FileNotFoundError(
            f"stability_events.jsonl not found. Run ligo_glitch_analysis.py first."
        )

    events = []
    with open(events_file) as f:
        for line in f:
            events.append(json.loads(line))
    return events


def run_gmm_validation():
    """Run GMM validation and generate outputs."""
    print("=" * 60)
    print("LIGO GMM Validation")
    print("=" * 60)

    # Load events
    events = load_stability_events()
    print(f"Loaded {len(events)} events from stability_events.jsonl")

    # Filter to OK events (exclude failed)
    ok_events = [e for e in events if e['stability'] != 'failed']
    print(f"OK events (excluding failed): {len(ok_events)}")

    # Extract curvature and labels
    curvatures_raw = np.array([e['curvature_b'] for e in ok_events], dtype=float)
    labels_raw = [e['stability'] for e in ok_events]

    # Filter NaNs/Infs for robustness
    valid_mask = np.isfinite(curvatures_raw)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  Warning: filtering {n_invalid} non-finite curvature values")
    curvatures = curvatures_raw[valid_mask]
    labels = [labels_raw[i] for i in np.where(valid_mask)[0]]

    # Scale curvature to 10^-3 for better numerical behavior
    curvatures_scaled = curvatures * 1000

    # Count by label
    n_stable_delayed = sum(1 for l in labels if l == 'stable_iof')
    n_stable_fast = sum(1 for l in labels if l == 'stable_std')
    n_flip = sum(1 for l in labels if l == 'flip')

    print(f"  Stable delayed (IOF): {n_stable_delayed}")
    print(f"  Stable fast (STD): {n_stable_fast}")
    print(f"  Flip: {n_flip}")

    # Fit 2-component GMM with n_init for stability
    print("\nFitting 2-component GMM (random_state=42, n_init=10)...")
    gmm = GaussianMixture(
        n_components=2,
        random_state=RNG_SEED,
        covariance_type='full',
        n_init=10
    )
    gmm.fit(curvatures_scaled.reshape(-1, 1))

    # Get GMM parameters
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    # Identify which component is "fast-like" (lower mean) vs "delayed-like" (higher mean)
    if means[0] < means[1]:
        fast_idx, delayed_idx = 0, 1
    else:
        fast_idx, delayed_idx = 1, 0

    print(f"\nGMM components (scaled by 10^3):")
    print(f"  Fast-like:    mean={means[fast_idx]:.3f}, std={stds[fast_idx]:.3f}, weight={weights[fast_idx]:.3f}")
    print(f"  Delayed-like: mean={means[delayed_idx]:.3f}, std={stds[delayed_idx]:.3f}, weight={weights[delayed_idx]:.3f}")

    # Compute decision boundary (where posteriors are equal)
    # Restrict search to interval between the two means (the relevant crossing)
    gmm_labels = gmm.predict(curvatures_scaled.reshape(-1, 1))

    # Find boundary by searching between means
    lo, hi = sorted(means)
    pad = 1.0  # Padding to handle edge cases
    b_range = np.linspace(curvatures_scaled.min() - pad, curvatures_scaled.max() + pad, 1000)
    posteriors = gmm.predict_proba(b_range.reshape(-1, 1))

    # Restrict to region between means to get the correct crossing
    # But verify the posterior difference actually gets near zero there
    diff = np.abs(posteriors[:, 0] - posteriors[:, 1])
    between_means_mask = (b_range >= lo) & (b_range <= hi)

    if between_means_mask.any():
        diff_between = diff[between_means_mask]
        if diff_between.min() < 1e-3:
            # Good crossing found between means
            decision_boundary = b_range[between_means_mask][diff_between.argmin()]
        else:
            # True crossing is outside means interval; use global search
            decision_boundary = b_range[diff.argmin()]
    else:
        # Fallback to global search (shouldn't happen with real data)
        decision_boundary = b_range[diff.argmin()]

    print(f"\nDecision boundary: b = {decision_boundary:.2f} (×10^-3)")

    # Map GMM labels to delayed/fast
    # Component with higher mean is "delayed-like"
    gmm_is_delayed = (gmm_labels == delayed_idx)

    # Compare to model-derived labels (stable-core only)
    # Create binary arrays for stable events
    stable_mask = np.array([l in ('stable_iof', 'stable_std') for l in labels])
    model_is_delayed = np.array([l == 'stable_iof' for l in labels])

    # Agreement metrics on stable-core only
    stable_gmm_delayed = gmm_is_delayed[stable_mask]
    stable_model_delayed = model_is_delayed[stable_mask]

    agreement = np.mean(stable_gmm_delayed == stable_model_delayed)
    ari = adjusted_rand_score(stable_model_delayed, stable_gmm_delayed)
    nmi = normalized_mutual_info_score(stable_model_delayed, stable_gmm_delayed)

    print(f"\nAgreement metrics (stable-core only, N={stable_mask.sum()}):")
    print(f"  Agreement: {agreement*100:.1f}%")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"  Normalized Mutual Info: {nmi:.3f}")

    # Save JSON results
    results = {
        "n_all_ok": len(ok_events),  # Before NaN filtering (raw OK count)
        "n_finite": len(curvatures),  # After NaN filtering
        "n_stable_delayed": n_stable_delayed,
        "n_stable_fast": n_stable_fast,
        "n_flip": n_flip,
        "gmm_components": {
            "fast_like": {
                "mean": float(means[fast_idx]),
                "std": float(stds[fast_idx]),
                "weight": float(weights[fast_idx])
            },
            "delayed_like": {
                "mean": float(means[delayed_idx]),
                "std": float(stds[delayed_idx]),
                "weight": float(weights[delayed_idx])
            }
        },
        "decision_boundary": float(decision_boundary),
        "stable_only_comparison": {
            "n_stable": int(stable_mask.sum()),
            "agreement": float(agreement),
            "agreement_pct": float(agreement * 100),
            "adjusted_rand_index": float(ari),
            "normalized_mutual_info": float(nmi)
        },
        "note": "GMM fit to all OK events (ignoring labels), then compared to model-derived stable-only labels. Curvature scaled by 10^3."
    }

    json_path = OUTPUT_DIR / "unsupervised_validation.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Generate figure
    print("\nGenerating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: GMM fit with component densities
    ax1 = axes[0]

    # Histogram of all OK curvatures (after NaN filtering)
    ax1.hist(curvatures_scaled, bins=50, density=True, alpha=0.6, color='gray',
             edgecolor='black', linewidth=0.5, label='OK events (finite $b$)')

    # GMM component densities - FIXED: use correct component indices
    x_plot = np.linspace(curvatures_scaled.min() - 0.5, curvatures_scaled.max() + 0.5, 500)

    for comp_idx, label_name, color_val in [
        (fast_idx, "Fast-like", "C0"),
        (delayed_idx, "Delayed-like", "C1"),
    ]:
        weight = weights[comp_idx]
        mean = means[comp_idx]
        std = stds[comp_idx]
        component_pdf = weight * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_plot - mean) / std) ** 2)
        ax1.plot(x_plot, component_pdf, color=color_val, linewidth=2,
                 label=f'{label_name} (μ={mean:.2f})')

    # Decision boundary
    ax1.axvline(decision_boundary, color='red', linestyle='--', linewidth=1.5,
               label=f'Boundary: {decision_boundary:.2f}')

    ax1.set_xlabel('Curvature index $b$ (×10$^{-3}$)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('2-component GMM fit (labels ignored)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(curvatures_scaled.min() - 0.5, curvatures_scaled.max() + 0.5)

    # Right panel: Same data colored by model-derived labels
    ax2 = axes[1]

    # Separate by model-derived label
    delayed_curv = curvatures_scaled[np.array([l == 'stable_iof' for l in labels])]
    fast_curv = curvatures_scaled[np.array([l == 'stable_std' for l in labels])]
    flip_curv = curvatures_scaled[np.array([l == 'flip' for l in labels])]

    bins = np.linspace(curvatures_scaled.min() - 0.5, curvatures_scaled.max() + 0.5, 50)

    ax2.hist(fast_curv, bins=bins, alpha=0.6, color='C0', edgecolor='black',
             linewidth=0.5, label=f'Stable Fast (N={len(fast_curv)})')
    ax2.hist(delayed_curv, bins=bins, alpha=0.6, color='C1', edgecolor='black',
             linewidth=0.5, label=f'Stable Delayed (N={len(delayed_curv)})')
    ax2.hist(flip_curv, bins=bins, alpha=0.3, color='gray', edgecolor='black',
             linewidth=0.5, label=f'Flip (N={len(flip_curv)})')

    # Decision boundary
    ax2.axvline(decision_boundary, color='red', linestyle='--', linewidth=1.5)

    ax2.set_xlabel('Curvature index $b$ (×10$^{-3}$)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Colored by model-derived labels', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)

    # Add agreement annotation
    ax2.text(0.02, 0.98, f'Agreement: {agreement*100:.1f}%\nARI: {ari:.2f}\nNMI: {nmi:.2f}',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    fig_path = FIGURES_DIR / "unsupervised_gmm_validation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")

    print("\n" + "=" * 60)
    print("GMM validation complete")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_gmm_validation()
