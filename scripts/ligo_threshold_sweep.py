#!/usr/bin/env python3
"""
LIGO AICc Threshold Sensitivity Analysis
=========================================

Post-processing script that reclassifies events at different AICc thresholds
using stored per-window delta_aicc values from the main pipeline.

This is a deterministic sweep on frozen pipeline output - no model re-fitting
occurs. The analysis tests whether stability classification and curvature
separation are sensitive to the confidence threshold choice.

Thresholds tested: |ΔAICc| ≥ 1, 2 (default), 4

Output:
- Stability fractions at each threshold
- Flip breakdown (determinate vs uncertain)
- Curvature AUC on stable-core subset

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
from pathlib import Path

# Fixed seed for reproducible bootstrap confidence intervals
np.random.seed(42)

# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "ligo_envelope"
DETAILED_FILE = OUTPUT_DIR / "stability_events_detailed.jsonl"

WINDOWS = ['60', '100', '150']


# -----------------------------------------------------------------------------
# Classification functions
# -----------------------------------------------------------------------------

def classify_window(delta_aicc, geometry, threshold):
    """
    Classify a single window based on AICc evidence.

    Parameters
    ----------
    delta_aicc : float or None
        AICc difference (runner-up - winner), always positive
    geometry : str
        Winner's geometry ('delayed' or 'fast')
    threshold : float
        Minimum delta_aicc for confident classification

    Returns
    -------
    str
        Classification: 'iof', 'standard', 'uncertain', or 'failed'
    """
    if delta_aicc is None:
        return 'failed'

    if delta_aicc >= threshold:
        return 'iof' if geometry == 'delayed' else 'standard'
    else:
        return 'uncertain'


def classify_event(window_results, threshold):
    """
    Determine event stability from 3-window classification.

    Parameters
    ----------
    window_results : dict
        Per-window results with delta_aicc and geometry
    threshold : float
        AICc threshold for confident classification

    Returns
    -------
    tuple
        (stability, flip_type) where stability is one of 'stable_iof',
        'stable_std', 'flip', 'failed' and flip_type is 'determinate',
        'uncertain', or None
    """
    classifications = []
    has_uncertain = False

    for w in WINDOWS:
        wr = window_results.get(w, {})
        cls = classify_window(
            wr.get('delta_aicc'),
            wr.get('geometry'),
            threshold
        )
        classifications.append(cls)
        if cls == 'uncertain':
            has_uncertain = True

    if 'failed' in classifications:
        return 'failed', None

    if all(c == 'iof' for c in classifications):
        return 'stable_iof', None
    elif all(c == 'standard' for c in classifications):
        return 'stable_std', None
    else:
        flip_type = 'uncertain' if has_uncertain else 'determinate'
        return 'flip', flip_type


# -----------------------------------------------------------------------------
# Statistical functions
# -----------------------------------------------------------------------------

def compute_auc(x, y):
    """
    Compute area under ROC curve for separating two distributions.

    Parameters
    ----------
    x, y : array-like
        Values from two groups

    Returns
    -------
    float
        AUC value in [0, 1]
    """
    x, y = np.array(x), np.array(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.5
    count = sum(1 for xi in x for yi in y if xi > yi)
    count += 0.5 * sum(1 for xi in x for yi in y if xi == yi)
    return count / (n_x * n_y)


def bootstrap_auc(x, y, n_boot=1000):
    """
    Compute bootstrap confidence interval for AUC.

    Parameters
    ----------
    x, y : array-like
        Values from two groups
    n_boot : int
        Number of bootstrap iterations

    Returns
    -------
    list
        [lower, upper] 95% confidence interval
    """
    x, y = np.array(x), np.array(y)
    aucs = []
    for _ in range(n_boot):
        x_sample = np.random.choice(x, len(x), replace=True)
        y_sample = np.random.choice(y, len(y), replace=True)
        aucs.append(compute_auc(x_sample, y_sample))
    return np.percentile(aucs, [2.5, 97.5])


def cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size.

    Parameters
    ----------
    x, y : array-like
        Values from two groups

    Returns
    -------
    float
        Effect size in [-1, 1]
    """
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

def run_threshold_sweep(events, thresholds):
    """
    Run classification sweep across multiple thresholds.

    Parameters
    ----------
    events : list of dict
        Events with per-window results and curvature
    thresholds : list of float
        AICc thresholds to test

    Returns
    -------
    dict
        Results keyed by threshold
    """
    results = {}

    for T in thresholds:
        counts = {
            'stable_iof': 0,
            'stable_std': 0,
            'flip': 0,
            'flip_determinate': 0,
            'flip_uncertain': 0,
            'failed': 0,
            'flip_short_delayed': 0,  # IOF at 60ms window
            'flip_long_delayed': 0,   # IOF at 150ms window
        }

        b_iof = []
        b_std = []

        for event in events:
            stability, flip_type = classify_event(event['window_results'], T)
            counts[stability] += 1

            if stability == 'flip' and flip_type:
                counts[f'flip_{flip_type}'] += 1

                # Track flip direction for determinate flips
                if flip_type == 'determinate':
                    wr = event['window_results']
                    cls_60 = classify_window(wr.get('60', {}).get('delta_aicc'), wr.get('60', {}).get('geometry'), T)
                    cls_150 = classify_window(wr.get('150', {}).get('delta_aicc'), wr.get('150', {}).get('geometry'), T)
                    if cls_60 == 'iof':
                        counts['flip_short_delayed'] += 1
                    if cls_150 == 'iof':
                        counts['flip_long_delayed'] += 1

            if event['curvature_b'] is not None:
                if stability == 'stable_iof':
                    b_iof.append(event['curvature_b'])
                elif stability == 'stable_std':
                    b_std.append(event['curvature_b'])

        n_ok = counts['stable_iof'] + counts['stable_std'] + counts['flip']

        if len(b_iof) > 0 and len(b_std) > 0:
            auc = compute_auc(b_iof, b_std)
            auc_ci = bootstrap_auc(b_iof, b_std)
            cliff_d = cliffs_delta(b_iof, b_std)
        else:
            auc, auc_ci, cliff_d = None, [None, None], None

        results[T] = {
            'threshold': T,
            'n_ok': n_ok,
            'n_stable_iof': counts['stable_iof'],
            'pct_stable_iof': 100 * counts['stable_iof'] / n_ok if n_ok > 0 else 0,
            'n_stable_std': counts['stable_std'],
            'pct_stable_std': 100 * counts['stable_std'] / n_ok if n_ok > 0 else 0,
            'n_flip': counts['flip'],
            'pct_flip': 100 * counts['flip'] / n_ok if n_ok > 0 else 0,
            'n_flip_determinate': counts['flip_determinate'],
            'n_flip_uncertain': counts['flip_uncertain'],
            'n_flip_short_delayed': counts['flip_short_delayed'],  # IOF at 60ms
            'n_flip_long_delayed': counts['flip_long_delayed'],    # IOF at 150ms
            'pct_flip_short_delayed': 100 * counts['flip_short_delayed'] / counts['flip_determinate'] if counts['flip_determinate'] > 0 else 0,
            'pct_flip_long_delayed': 100 * counts['flip_long_delayed'] / counts['flip_determinate'] if counts['flip_determinate'] > 0 else 0,
            'n_failed': counts['failed'],
            'auc': auc,
            'auc_ci_lo': auc_ci[0],
            'auc_ci_hi': auc_ci[1],
            'cliffs_delta': cliff_d,
        }

    return results


def main():
    """Run AICc threshold sensitivity analysis."""
    print("=" * 70)
    print("LIGO AICc Threshold Sensitivity Analysis")
    print("=" * 70)
    print(f"\nFixed RNG seed: 42")
    print(f"Reading: {DETAILED_FILE.name}")

    events = []
    with open(DETAILED_FILE) as f:
        for line in f:
            events.append(json.loads(line))
    print(f"Loaded: {len(events)} events")

    thresholds = [1, 2, 4]
    results = run_threshold_sweep(events, thresholds)

    # Results table
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\n{'Threshold':<12} {'N_ok':<8} {'Stable Delayed':<18} {'Stable Fast':<16} {'Flip':<12}")
    print("-" * 66)
    for T in thresholds:
        r = results[T]
        print(f"|ΔAICc| ≥ {T:<3} {r['n_ok']:<8} "
              f"{r['n_stable_iof']:>3} ({r['pct_stable_iof']:>5.1f}%)      "
              f"{r['n_stable_std']:>3} ({r['pct_stable_std']:>5.1f}%)    "
              f"{r['n_flip']:>3} ({r['pct_flip']:>5.1f}%)")

    print("\n" + "-" * 66)
    print("Flip Breakdown:")
    print(f"{'Threshold':<12} {'Determinate':<15} {'Uncertain':<15}")
    print("-" * 42)
    for T in thresholds:
        r = results[T]
        if r['n_flip'] > 0:
            det_pct = 100 * r['n_flip_determinate'] / r['n_flip']
            unc_pct = 100 * r['n_flip_uncertain'] / r['n_flip']
        else:
            det_pct, unc_pct = 0, 0
        print(f"|ΔAICc| ≥ {T:<3} {r['n_flip_determinate']:>3} ({det_pct:>5.1f}%)      "
              f"{r['n_flip_uncertain']:>3} ({unc_pct:>5.1f}%)")

    print("\n" + "-" * 66)
    print("Curvature Separation (Stable Core):")
    print(f"{'Threshold':<12} {'AUC':<24} {'Cliff δ':<10}")
    print("-" * 46)
    for T in thresholds:
        r = results[T]
        if r['auc'] is not None:
            print(f"|ΔAICc| ≥ {T:<3} {r['auc']:.3f} [{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}]   {r['cliffs_delta']:.3f}")
        else:
            print(f"|ΔAICc| ≥ {T:<3} --")

    # Save results
    output_file = OUTPUT_DIR / "threshold_sweep_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'created': '2025-12-16',
                'rng_seed': 42,
                'n_events': len(events),
                'thresholds': thresholds,
            },
            'results': {str(k): v for k, v in results.items()}
        }, f, indent=2)
    print(f"\nSaved: {output_file}")

    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
