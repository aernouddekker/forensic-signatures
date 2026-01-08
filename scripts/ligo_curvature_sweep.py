#!/usr/bin/env python3
"""
LIGO Curvature Window Sweep
============================

Computes curvature index b across multiple fit windows (10, 20, 30 ms)
to demonstrate that the curvature separation is robust to window choice.

This addresses the referee critique: "why 0-20ms? why that normalization?"

The script:
1. Loads stability classifications from stability_events.jsonl (output of ligo_stability_figures.py)
2. Recomputes curvature b at 10ms, 20ms, and 30ms windows using canonical peak alignment
3. Computes AUC(b) for each window (stable_iof vs stable_std)
4. Outputs results to JSON for LaTeX macro generation

Pipeline discipline:
- Uses SAME peak-finding logic as ligo_stability_figures.py (find_constrained_peak)
- Uses SAME baseline computation (baseline_tail_median from n//5 tail of 150ms window)
- Single centering authority: center_times_on_peak called ONCE inside curvature function
- No double-centering

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr

# Import canonical pipeline invariants from shared module
from ligo_pipeline_common import (
    compute_hilbert_envelope,
    center_times_on_peak,
    extract_fit_window_indices,
    find_constrained_peak,
    load_cached_strain,
    PEAK_SEARCH_WINDOW_MS,
)

from iof_metrics import baseline_tail_median

import warnings
# Suppress only specific known noisy warnings, not all
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "ligo_envelope"
CACHE_DIR = SCRIPT_DIR / "strain_cache"
STABILITY_FILE = OUTPUT_DIR / "stability_events.jsonl"

# Curvature windows to sweep
CURVATURE_WINDOWS_MS = [10, 20, 30]
BASELINE_WINDOW_MS = 150.0  # Fixed baseline window (same as main pipeline)


def load_stability_events():
    """Load stability-classified events from JSONL (output of ligo_stability_figures.py)."""
    events = []
    with open(STABILITY_FILE) as f:
        for line in f:
            events.append(json.loads(line))
    return events


def compute_curvature_at_window(envelope, times_ms, peak_idx, curvature_window_ms,
                                 baseline_window_ms=BASELINE_WINDOW_MS):
    """
    Compute curvature index b at a specific window size.

    Uses canonical pipeline functions:
    - center_times_on_peak: single centering authority (called here, not in caller)
    - extract_fit_window_indices: searchsorted-based window extraction
    - baseline_tail_median: n//5 tail rule

    Args:
        envelope: Hilbert envelope array
        times_ms: Time array in ms (GPS-relative, NOT pre-centered)
        peak_idx: Index of peak in envelope
        curvature_window_ms: Window for quadratic fit (10, 20, or 30 ms)
        baseline_window_ms: Window for baseline estimation (150 ms)

    Returns:
        Curvature index b (quadratic coefficient), or None if computation fails
    """
    peak_val = envelope[peak_idx]

    # Single centering authority: center times on peak here
    times_centered = center_times_on_peak(times_ms, peak_idx)

    # Extract baseline window by indices (same as main pipeline)
    env_fit_150_idx = extract_fit_window_indices(times_centered, peak_idx, baseline_window_ms)

    if len(env_fit_150_idx) < 50:
        return None

    env_fit_150 = envelope[env_fit_150_idx]

    # Canonical baseline from n//5 tail (same as main pipeline)
    baseline_150 = baseline_tail_median(env_fit_150)

    if peak_val <= baseline_150:
        return None

    # Extract curvature window by indices
    env_fit_curv_idx = extract_fit_window_indices(times_centered, peak_idx, curvature_window_ms)

    if len(env_fit_curv_idx) < 5:  # Minimum samples for quadratic fit
        return None

    t_curv = times_centered[env_fit_curv_idx]
    env_curv = envelope[env_fit_curv_idx]

    # Canonical z-transform with epsilon and clip (same as main pipeline)
    z_curv = 1 - (env_curv - baseline_150) / (peak_val - baseline_150 + 1e-30)
    z_curv = np.clip(z_curv, 0, 1.5)

    # Polyfit and return quadratic coefficient
    # b has units ms^-2 (since t is in ms)
    try:
        coeffs = np.polyfit(t_curv, z_curv, 2)
        return float(coeffs[0])  # Quadratic coefficient b
    except:
        return None


def compute_auc(x, y):
    """
    AUC from Mann-Whitney U.

    Returns P(random x > random y).
    """
    if len(x) == 0 or len(y) == 0:
        return 0.5
    u_stat, _ = mannwhitneyu(x, y, alternative='two-sided')
    return u_stat / (len(x) * len(y))


def bootstrap_auc(x, y, n_boot=1000, seed=42):
    """Bootstrap CI for AUC."""
    np.random.seed(seed)
    x, y = np.array(x), np.array(y)
    aucs = []
    for _ in range(n_boot):
        x_boot = np.random.choice(x, len(x), replace=True)
        y_boot = np.random.choice(y, len(y), replace=True)
        aucs.append(compute_auc(x_boot, y_boot))
    return np.percentile(aucs, [2.5, 97.5])


def main():
    print("=" * 70)
    print("LIGO Curvature Window Sweep")
    print("=" * 70)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load stability events
    events = load_stability_events()
    print(f"Loaded {len(events)} events from {STABILITY_FILE.name}")

    # Filter to stable events only (for AUC computation)
    stable_iof = [e for e in events if e['stability'] == 'stable_iof']
    stable_std = [e for e in events if e['stability'] == 'stable_std']

    print(f"  Stable Delayed (IOF): {len(stable_iof)}")
    print(f"  Stable Fast (STD): {len(stable_std)}")

    # Compute curvature at each window for all events
    print(f"\nComputing curvature at windows: {CURVATURE_WINDOWS_MS} ms...")
    print(f"Using canonical peak finder with ±{PEAK_SEARCH_WINDOW_MS} ms constraint")

    # Store curvature values: {window_ms: {gps_time: b}}
    curvatures = {w: {} for w in CURVATURE_WINDOWS_MS}

    processed = 0
    skipped = 0
    peak_failures = 0

    for i, event in enumerate(events):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(events)}...")

        gps = event['gps_time']
        strain_data = load_cached_strain(gps, CACHE_DIR)

        if strain_data is None:
            skipped += 1
            continue

        # Compute envelope
        strain = strain_data['values']
        fs = strain_data['sample_rate']
        times = strain_data['times']

        # Convert times relative to GPS time (GPS-relative, not centered yet)
        times_ms = (times - gps) * 1000

        envelope = compute_hilbert_envelope(strain, fs)

        # Use CANONICAL peak finder (same as ligo_stability_figures.py)
        # This ensures we align to the same peak that was used for classification
        peak_idx = find_constrained_peak(envelope, times_ms, PEAK_SEARCH_WINDOW_MS)

        if peak_idx is None or peak_idx < 0 or peak_idx >= len(envelope):
            peak_failures += 1
            continue

        # Compute curvature at each window
        # Note: times_ms is NOT pre-centered; centering happens inside compute_curvature_at_window
        for w in CURVATURE_WINDOWS_MS:
            b = compute_curvature_at_window(envelope, times_ms, peak_idx, w)
            if b is not None:
                curvatures[w][gps] = b

        processed += 1

    print(f"\nProcessed: {processed}, Skipped (no cache): {skipped}, Peak failures: {peak_failures}")

    # Compute AUC for each window
    results = {'windows': {}}

    print("\nCurvature separation by window:")
    print("-" * 70)

    # Collect b values for cross-window correlation
    b_by_window = {}

    for w in CURVATURE_WINDOWS_MS:
        # Get curvatures for stable populations
        b_iof = [curvatures[w][e['gps_time']] for e in stable_iof
                 if e['gps_time'] in curvatures[w]]
        b_std = [curvatures[w][e['gps_time']] for e in stable_std
                 if e['gps_time'] in curvatures[w]]

        if len(b_iof) < 5 or len(b_std) < 5:
            print(f"  {w} ms: Insufficient data (IOF={len(b_iof)}, STD={len(b_std)})")
            continue

        # Store for correlation
        b_by_window[w] = curvatures[w]

        # Compute AUC (P(random IOF > random STD))
        auc_raw = compute_auc(b_iof, b_std)

        # Ensure AUC represents discriminability (flip if needed to get AUC >= 0.5)
        # Convention: we expect IOF (delayed) to have HIGHER b (accelerating)
        median_iof = np.median(b_iof)
        median_std = np.median(b_std)

        if median_iof < median_std:
            # Sign flipped - report 1-AUC to maintain "higher = better separation"
            auc = 1 - auc_raw
            auc_ci_raw = bootstrap_auc(b_iof, b_std)
            auc_ci = [1 - auc_ci_raw[1], 1 - auc_ci_raw[0]]  # Flip CI too
            sign_note = "flipped"
        else:
            auc = auc_raw
            auc_ci = list(bootstrap_auc(b_iof, b_std))
            sign_note = "normal"

        # Mann-Whitney p-value
        _, p_value = mannwhitneyu(b_iof, b_std, alternative='two-sided')

        # Check sign stability: IOF should have positive median, STD negative
        sign_stable = (median_iof > 0 and median_std < 0)

        results['windows'][str(w)] = {
            'window_ms': w,
            'n_iof': len(b_iof),
            'n_std': len(b_std),
            'auc': float(auc),
            'auc_ci_lower': float(auc_ci[0]),
            'auc_ci_upper': float(auc_ci[1]),
            'auc_orientation': sign_note,
            'p_value': float(p_value),
            # Raw medians in ms^-2
            'median_iof_raw': float(median_iof),
            'median_std_raw': float(median_std),
            # Scaled by 10^3 for readability: column header should be "b (×10⁻³ ms⁻²)"
            # i.e., displayed_value × 10⁻³ ms⁻² = actual_value
            # e.g., 0.0006 ms⁻² → displayed as 0.6 (meaning 0.6 × 10⁻³ ms⁻²)
            'median_iof_scaled': float(median_iof * 1000),
            'median_std_scaled': float(median_std * 1000),
            'sign_stable': bool(sign_stable),
        }

        sign_str = "Yes" if sign_stable else "No"
        print(f"  0-{w:2d} ms: AUC = {auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}], "
              f"p = {p_value:.2e}, Sign stable: {sign_str}")

    # Compute Spearman correlations between windows (all events)
    print("\nCross-window Spearman correlations (all events):")
    correlations = {}
    for i, w1 in enumerate(CURVATURE_WINDOWS_MS):
        for w2 in CURVATURE_WINDOWS_MS[i+1:]:
            if w1 not in b_by_window or w2 not in b_by_window:
                continue
            # Find events present in both windows
            common_gps = set(b_by_window[w1].keys()) & set(b_by_window[w2].keys())
            if len(common_gps) < 10:
                continue
            b1 = [b_by_window[w1][g] for g in common_gps]
            b2 = [b_by_window[w2][g] for g in common_gps]
            rho, p = spearmanr(b1, b2)
            key = f"{w1}_vs_{w2}"
            correlations[key] = {
                'rho': float(rho),
                'p_value': float(p),
                'n_common': len(common_gps),
            }
            print(f"  {w1}ms vs {w2}ms: ρ = {rho:.3f}, p = {p:.2e}, n = {len(common_gps)}")

    results['cross_window_correlations'] = correlations

    # Compute Spearman correlations for stable-core events only
    print("\nCross-window Spearman correlations (stable-core only):")
    stable_gps = set(e['gps_time'] for e in stable_iof + stable_std)
    correlations_stable = {}
    for i, w1 in enumerate(CURVATURE_WINDOWS_MS):
        for w2 in CURVATURE_WINDOWS_MS[i+1:]:
            if w1 not in b_by_window or w2 not in b_by_window:
                continue
            # Find stable events present in both windows
            common_gps = (set(b_by_window[w1].keys()) & set(b_by_window[w2].keys())) & stable_gps
            if len(common_gps) < 10:
                continue
            b1 = [b_by_window[w1][g] for g in common_gps]
            b2 = [b_by_window[w2][g] for g in common_gps]
            rho, p = spearmanr(b1, b2)
            key = f"{w1}_vs_{w2}"
            correlations_stable[key] = {
                'rho': float(rho),
                'p_value': float(p),
                'n_common': len(common_gps),
            }
            print(f"  {w1}ms vs {w2}ms: ρ = {rho:.3f}, p = {p:.2e}, n = {len(common_gps)}")

    results['cross_window_correlations_stable'] = correlations_stable

    # Add summary
    results['summary'] = {
        'n_events': len(events),
        'n_stable_iof': len(stable_iof),
        'n_stable_std': len(stable_std),
        'n_processed': processed,
        'n_skipped': skipped,
        'n_peak_failures': peak_failures,
        'baseline_window_ms': BASELINE_WINDOW_MS,
        'peak_search_window_ms': PEAK_SEARCH_WINDOW_MS,
        'windows_tested': CURVATURE_WINDOWS_MS,
    }

    # Add notes about methodology
    results['notes'] = {
        'auc_orientation': (
            'AUC is oriented so that AUC >= 0.5 indicates separability. '
            'If median(IOF) < median(STD), we report 1-AUC and flip the CI.'
        ),
        'curvature_units': (
            'Raw curvature b has units ms^-2. '
            'Scaled values are b * 1000 for readability; column header should show '
            '"b (×10^-3 ms^-2)" meaning displayed_value × 10^-3 ms^-2 = actual_value.'
        ),
        'peak_alignment': (
            'Uses canonical find_constrained_peak with ±500ms constraint, '
            'same as ligo_stability_figures.py classification.'
        ),
    }

    # Save results
    output_file = OUTPUT_DIR / "curvature_sweep_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")

    # Print table for LaTeX
    print("\n" + "=" * 70)
    print("LaTeX table data:")
    print("=" * 70)
    print("Fit Interval | AUC(b) | 95% CI | Sign Stable?")
    print("-" * 70)
    for w in CURVATURE_WINDOWS_MS:
        if str(w) in results['windows']:
            r = results['windows'][str(w)]
            default = " (default)" if w == 20 else ""
            sign = "Yes" if r['sign_stable'] else "No"
            print(f"0--{w}~ms{default} | {r['auc']:.3f} | [{r['auc_ci_lower']:.3f}, {r['auc_ci_upper']:.3f}] | {sign}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
