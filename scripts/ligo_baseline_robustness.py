#!/usr/bin/env python3
"""
LIGO Curvature Baseline Robustness Analysis
============================================

Tests whether curvature-based separation of delayed vs fast recovery events
is sensitive to the baseline estimation method.

IMPORTANT: This is a controlled robustness check. All methods use the SAME
baseline REGION (tail of post-peak analysis window), but vary the ESTIMATOR:
- median: Median of samples in tail region (default in main pipeline)
- theil_sen: Robust linear fit on tail region, extrapolated to window end

This ensures we're testing estimator sensitivity, not region sensitivity.

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import theilslopes

# Import ALL signal processing from canonical source
from ligo_pipeline_common import (
    compute_hilbert_envelope,
    compute_times_ms,
    find_constrained_peak,
    baseline_from_postpeak_window,
    load_cached_strain,
    CURVATURE_WINDOW_MS,
    DEFAULT_BASELINE_WINDOW_MS,
)

# Fixed seed for reproducible bootstrap confidence intervals
np.random.seed(42)

# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "ligo_envelope"
CACHE_DIR = SCRIPT_DIR / "strain_cache"
STABILITY_FILE = OUTPUT_DIR / "stability_events.jsonl"


# -----------------------------------------------------------------------------
# Baseline estimation methods (same REGION, different ESTIMATOR)
# -----------------------------------------------------------------------------

def baseline_median(envelope, times_ms, peak_idx, window_ms, tail_fraction=0.2):
    """
    Baseline using median of tail region (matches main pipeline exactly).
    This is the DEFAULT method - included for comparison.
    """
    return baseline_from_postpeak_window(envelope, times_ms, peak_idx, window_ms, tail_fraction)


def baseline_theil_sen(envelope, times_ms, peak_idx, window_ms, tail_fraction=0.2):
    """
    Baseline using Theil-Sen robust linear fit on tail region.

    Uses the SAME region as main pipeline (last tail_fraction of analysis window),
    but fits a robust line and extrapolates to window end.
    """
    peak_time = times_ms[peak_idx]
    end_time = peak_time + window_ms
    tail_start = peak_time + window_ms * (1 - tail_fraction)

    # Get samples in tail region
    tail_mask = (times_ms >= tail_start) & (times_ms <= end_time)

    if np.sum(tail_mask) < 10:
        # Fallback to median if too few samples
        return baseline_median(envelope, times_ms, peak_idx, window_ms, tail_fraction)

    t_tail = times_ms[tail_mask]
    env_tail = envelope[tail_mask]

    try:
        slope, intercept, _, _ = theilslopes(env_tail, t_tail)
        # Extrapolate to end of window
        return intercept + slope * end_time
    except:
        return baseline_median(envelope, times_ms, peak_idx, window_ms, tail_fraction)


BASELINE_METHODS = {
    'median': baseline_median,
    'theil_sen': baseline_theil_sen,
}


# -----------------------------------------------------------------------------
# Curvature computation with specified baseline method
# -----------------------------------------------------------------------------

def compute_curvature_with_baseline(envelope, times_ms, peak_idx, baseline_method='median',
                                    curvature_window_ms=CURVATURE_WINDOW_MS,
                                    baseline_window_ms=DEFAULT_BASELINE_WINDOW_MS):
    """
    Compute curvature index b using specified baseline method.

    Args:
        envelope: Hilbert envelope
        times_ms: Time array in ms (GPS-relative)
        peak_idx: Index of peak (from find_constrained_peak)
        baseline_method: 'median' or 'theil_sen'
        curvature_window_ms: Window for quadratic fit (default 20ms)
        baseline_window_ms: Window for baseline estimation (default 150ms)

    Returns:
        Curvature index b, or None if computation fails
    """
    baseline_func = BASELINE_METHODS.get(baseline_method, baseline_median)

    peak_time = times_ms[peak_idx]
    peak_val = envelope[peak_idx]
    baseline = baseline_func(envelope, times_ms, peak_idx, baseline_window_ms)

    if peak_val <= baseline:
        return None

    mask = (times_ms >= peak_time) & (times_ms <= peak_time + curvature_window_ms)
    if np.sum(mask) < 10:
        return None

    t_fit = times_ms[mask] - peak_time
    env_fit = envelope[mask]

    # Normalize to recovery coordinate z ∈ [0, 1]
    z_fit = 1 - (env_fit - baseline) / (peak_val - baseline)
    z_fit = np.clip(z_fit, 0, 1.5)

    try:
        coeffs = np.polyfit(t_fit, z_fit, 2)
        return coeffs[0]
    except:
        return None


# -----------------------------------------------------------------------------
# Statistical functions
# -----------------------------------------------------------------------------

def compute_auc(x, y):
    """Compute AUC for separating two distributions."""
    x, y = np.array(x), np.array(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.5
    count = sum(1 for xi in x for yi in y if xi > yi)
    count += 0.5 * sum(1 for xi in x for yi in y if xi == yi)
    return count / (n_x * n_y)


def bootstrap_auc(x, y, n_boot=1000):
    """Bootstrap confidence interval for AUC."""
    x, y = np.array(x), np.array(y)
    aucs = []
    for _ in range(n_boot):
        x_sample = np.random.choice(x, len(x), replace=True)
        y_sample = np.random.choice(y, len(y), replace=True)
        aucs.append(compute_auc(x_sample, y_sample))
    return np.percentile(aucs, [2.5, 97.5])


def cliffs_delta(x, y):
    """Cliff's delta effect size."""
    x, y = np.array(x), np.array(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def main():
    """Run baseline robustness analysis."""
    print("=" * 70)
    print("LIGO Curvature Baseline Robustness Analysis")
    print("=" * 70)
    print(f"\nFixed RNG seed: 42")
    print(f"Baseline region: tail {int(100 * 0.2)}% of {DEFAULT_BASELINE_WINDOW_MS}ms window")
    print(f"Curvature window: {CURVATURE_WINDOW_MS}ms")

    # Load stable-core events
    events = []
    with open(STABILITY_FILE) as f:
        for line in f:
            events.append(json.loads(line))

    stable_iof = [e for e in events if e['stability'] == 'stable_iof']
    stable_std = [e for e in events if e['stability'] == 'stable_std']

    print(f"\nStable core events:")
    print(f"  Stable Delayed: {len(stable_iof)}")
    print(f"  Stable Fast: {len(stable_std)}")

    methods = list(BASELINE_METHODS.keys())
    results = {}

    for method in methods:
        print(f"\nProcessing: {method}")

        b_iof = []
        b_std = []

        for e in stable_iof:
            gps_time = e['gps_time']
            data = load_cached_strain(gps_time, CACHE_DIR)
            if data is None:
                continue

            strain = data['values']
            fs = data['sample_rate']
            times = data['times']
            times_ms = compute_times_ms(times, gps_time)

            envelope = compute_hilbert_envelope(strain, fs)
            peak_idx = find_constrained_peak(envelope, times_ms)

            b = compute_curvature_with_baseline(envelope, times_ms, peak_idx, method)
            if b is not None:
                b_iof.append(b)

        for e in stable_std:
            gps_time = e['gps_time']
            data = load_cached_strain(gps_time, CACHE_DIR)
            if data is None:
                continue

            strain = data['values']
            fs = data['sample_rate']
            times = data['times']
            times_ms = compute_times_ms(times, gps_time)

            envelope = compute_hilbert_envelope(strain, fs)
            peak_idx = find_constrained_peak(envelope, times_ms)

            b = compute_curvature_with_baseline(envelope, times_ms, peak_idx, method)
            if b is not None:
                b_std.append(b)

        if len(b_iof) > 0 and len(b_std) > 0:
            auc = compute_auc(b_iof, b_std)
            auc_ci = bootstrap_auc(b_iof, b_std)
            cliff_d = cliffs_delta(b_iof, b_std)

            results[method] = {
                'n_delayed': len(b_iof),
                'n_fast': len(b_std),
                'median_b_delayed': float(np.median(b_iof) * 1000),
                'median_b_fast': float(np.median(b_std) * 1000),
                'auc': auc,
                'auc_ci_lo': auc_ci[0],
                'auc_ci_hi': auc_ci[1],
                'cliffs_delta': cliff_d,
            }

            print(f"  n_Delayed={len(b_iof)}, n_Fast={len(b_std)}")
            print(f"  Median b: Delayed={np.median(b_iof)*1000:.3f}, Fast={np.median(b_std)*1000:.3f} (×10⁻³)")
            print(f"  AUC = {auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
            print(f"  Cliff's δ = {cliff_d:.3f}")
        else:
            results[method] = {'error': 'insufficient data'}
            print(f"  Insufficient data")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\n{'Method':<15} {'n_Del':<8} {'n_Fast':<8} {'AUC':<24} {'Cliff δ':<10}")
    print("-" * 65)
    for method in methods:
        r = results.get(method, {})
        if 'auc' in r:
            print(f"{method:<15} {r['n_delayed']:<8} {r['n_fast']:<8} "
                  f"{r['auc']:.3f} [{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}]   {r['cliffs_delta']:.3f}")
        else:
            print(f"{method:<15} -- (insufficient data)")

    # Interpretation
    if len(results) >= 2 and all('auc' in r for r in results.values()):
        aucs = [r['auc'] for r in results.values()]
        auc_range = max(aucs) - min(aucs)
        print(f"\nAUC range across methods: {auc_range:.3f}")
        if auc_range < 0.05:
            print("  → Excellent robustness: separation is stable across baseline estimators")
        elif auc_range < 0.10:
            print("  → Good robustness: minor sensitivity to baseline estimator")
        else:
            print("  → Moderate sensitivity: consider reporting range in manuscript")

    # Save results
    output_file = OUTPUT_DIR / "baseline_robustness_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'created': '2025-12-17',
                'rng_seed': 42,
                'curvature_window_ms': CURVATURE_WINDOW_MS,
                'baseline_window_ms': DEFAULT_BASELINE_WINDOW_MS,
                'baseline_region': 'tail 20% of analysis window (matches main pipeline)',
                'methods': methods,
                'note': 'All methods use SAME region, different estimators'
            },
            'results': results
        }, f, indent=2)
    print(f"\nSaved: {output_file}")

    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
