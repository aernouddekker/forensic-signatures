#!/usr/bin/env python3
"""
LIGO Rejected-Morphology Stress Test
====================================

Computes curvature index for REJECTED (complex) events AND runs the same
model tournament to test whether the curvature-geometry association persists
outside the single-pulse morphology filter.

This addresses a key referee concern: "Your result is an artifact of your
morphology filter."

This is an out-of-distribution robustness check:
- If association persists: filter is not required to induce the association
- If association disappears: association is specific to filtered morphology
- Result informs interpretation but is qualitative due to distribution shift

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
import pickle
from pathlib import Path
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import mannwhitneyu, ks_2samp
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

# Import canonical pipeline invariants from shared module
from ligo_pipeline_common import (
    bandpass_filter,
    compute_hilbert_envelope,
    baseline_from_postpeak_window,
    find_constrained_peak,
    compute_times_ms,
    compute_curvature_index as common_compute_curvature_index,
    CURVATURE_WINDOW_MS as COMMON_CURVATURE_WINDOW_MS
)

# Fixed seed for reproducibility
np.random.seed(42)

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "ligo_envelope"
RESULTS_FILE = OUTPUT_DIR / "ligo_envelope_Extremely_Loud_results.jsonl"
STABILITY_FILE = OUTPUT_DIR / "stability_events.jsonl"
CACHE_DIR = SCRIPT_DIR / "strain_cache"

# Analysis parameters
CURVATURE_WINDOW_MS = 20.0
ANALYSIS_WINDOW_MS = 100.0  # For model tournament
AICC_THRESHOLD = 2.0


# =============================================================================
# Signal processing - IMPORTED FROM ligo_pipeline_common.py
# =============================================================================
# bandpass_filter, compute_hilbert_envelope, baseline_from_postpeak_window
# are now imported from the canonical source. No local copies.


def compute_curvature_index(envelope, times_ms, window_ms=CURVATURE_WINDOW_MS, peak_idx=None,
                            baseline_window_ms=150.0):
    """
    Compute early-time curvature index b.
    Fits z(t) = z0 + a*t + b*t^2 over first window_ms after peak.
    """
    if peak_idx is None:
        peak_idx = np.argmax(envelope)
    peak_time = times_ms[peak_idx]
    peak_val = envelope[peak_idx]
    # Baseline from tail of analysis window (matches main pipeline)
    baseline = baseline_from_postpeak_window(envelope, times_ms, peak_idx, baseline_window_ms)

    if peak_val <= baseline:
        return None

    mask = (times_ms >= peak_time) & (times_ms <= peak_time + window_ms)
    if np.sum(mask) < 10:
        return None

    t_fit = times_ms[mask] - peak_time
    env_fit = envelope[mask]
    z_fit = 1 - (env_fit - baseline) / (peak_val - baseline)

    try:
        coeffs = np.polyfit(t_fit, z_fit, 2)
        return coeffs[0]  # Quadratic coefficient
    except:
        return None


# =============================================================================
# Model definitions (for tournament on complex events)
# =============================================================================

def exponential_recovery(t, A, tau, baseline):
    """z = baseline - A * exp(-t/tau)"""
    return baseline - A * np.exp(-t / tau)


def sigmoid_recovery(t, A, k, t0, baseline):
    """z = baseline - A / (1 + exp(k*(t-t0)))"""
    return baseline - A / (1 + np.exp(k * (t - t0)))


def delayed_exponential_recovery(t, A, tau, t_delay, baseline):
    """z = baseline - A * exp(-(t-t_delay)/tau) for t > t_delay"""
    result = np.full_like(t, baseline - A, dtype=float)
    mask = t > t_delay
    result[mask] = baseline - A * np.exp(-(t[mask] - t_delay) / tau)
    return result


def compute_aicc(n, k, sse):
    """Compute corrected AIC from SSE."""
    if sse <= 0 or n <= k + 1:
        return np.inf
    aic = n * np.log(sse / n) + 2 * k
    correction = 2 * k * (k + 1) / (n - k - 1)
    return aic + correction


def fit_and_classify(envelope, times_ms, window_ms=ANALYSIS_WINDOW_MS, peak_idx=None):
    """
    Fit models and classify as delayed/fast/uncertain.
    Returns (classification, curvature_b).
    """
    if peak_idx is None:
        peak_idx = np.argmax(envelope)
    peak_time = times_ms[peak_idx]
    peak_val = envelope[peak_idx]
    # Baseline from tail of analysis window (matches main pipeline)
    baseline = baseline_from_postpeak_window(envelope, times_ms, peak_idx, window_ms)

    if peak_val <= baseline * 1.1:
        return None, None

    # Extract window
    mask = (times_ms >= peak_time) & (times_ms <= peak_time + window_ms)
    t_window = times_ms[mask] - peak_time
    env_window = envelope[mask]

    if len(t_window) < 20:
        return None, None

    # Convert to recovery variable z
    z = 1 - (env_window - baseline) / (peak_val - baseline)

    # Compute curvature
    curv_mask = t_window <= CURVATURE_WINDOW_MS
    if np.sum(curv_mask) >= 10:
        try:
            coeffs = np.polyfit(t_window[curv_mask], z[curv_mask], 2)
            curvature_b = coeffs[0]
        except:
            curvature_b = None
    else:
        curvature_b = None

    # Fit models
    n = len(z)
    results = {}

    # Fast: exponential
    try:
        bounds = ([0.1, 1, 0.5], [2, 200, 1.5])
        p0 = [0.8, 20, 1.0]
        popt, _ = curve_fit(exponential_recovery, t_window, z, p0=p0, bounds=bounds, maxfev=5000)
        pred = exponential_recovery(t_window, *popt)
        sse = np.sum((z - pred)**2)
        results['exp'] = {'aicc': compute_aicc(n, 3, sse), 'geometry': 'fast'}
    except:
        pass

    # Delayed: sigmoid
    try:
        bounds = ([0.1, 0.01, 1, 0.5], [2, 1, 50, 1.5])
        p0 = [0.8, 0.1, 10, 1.0]
        popt, _ = curve_fit(sigmoid_recovery, t_window, z, p0=p0, bounds=bounds, maxfev=5000)
        pred = sigmoid_recovery(t_window, *popt)
        sse = np.sum((z - pred)**2)
        t_inf = popt[2]
        if 0 < t_inf < window_ms:
            results['sig'] = {'aicc': compute_aicc(n, 4, sse), 'geometry': 'delayed'}
    except:
        pass

    # Delayed: delayed exponential
    try:
        bounds = ([0.1, 1, 1, 0.5], [2, 200, 40, 1.5])
        p0 = [0.8, 20, 5, 1.0]
        popt, _ = curve_fit(delayed_exponential_recovery, t_window, z, p0=p0, bounds=bounds, maxfev=5000)
        pred = delayed_exponential_recovery(t_window, *popt)
        sse = np.sum((z - pred)**2)
        t_inf = popt[2]
        if 0 < t_inf < window_ms:
            results['del_exp'] = {'aicc': compute_aicc(n, 4, sse), 'geometry': 'delayed'}
    except:
        pass

    if not results:
        return None, curvature_b

    # Classify
    fast_aiccs = [r['aicc'] for r in results.values() if r['geometry'] == 'fast']
    del_aiccs = [r['aicc'] for r in results.values() if r['geometry'] == 'delayed']

    best_fast = min(fast_aiccs) if fast_aiccs else np.inf
    best_del = min(del_aiccs) if del_aiccs else np.inf

    delta = best_fast - best_del

    if delta >= AICC_THRESHOLD:
        return 'delayed', curvature_b
    elif delta <= -AICC_THRESHOLD:
        return 'fast', curvature_b
    else:
        return 'uncertain', curvature_b


def load_cached_strain(gps_time):
    """Load cached strain data."""
    # Use full GPS time to match ligo_glitch_analysis.py caching
    cache_file = CACHE_DIR / f"strain_{gps_time:.6f}.pkl"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except:
        return None


def compute_auc(delayed_values, fast_values):
    """Compute AUC for separation (how well values discriminate delayed from fast)."""
    if len(delayed_values) == 0 or len(fast_values) == 0:
        return None

    # Mann-Whitney U statistic normalized to [0,1] is the AUC
    n1, n2 = len(delayed_values), len(fast_values)
    try:
        stat, p = mannwhitneyu(delayed_values, fast_values, alternative='two-sided')
        auc = stat / (n1 * n2)
        return max(auc, 1 - auc)  # Ensure AUC >= 0.5
    except:
        return None


def main():
    print("=" * 70)
    print("LIGO Rejected-Morphology Stress Test")
    print("=" * 70)

    # Load all events (deduplicated by GPS time)
    events_by_gps = {}
    with open(RESULTS_FILE) as f:
        for line in f:
            e = json.loads(line)
            gps = e.get('gps_time')
            if gps not in events_by_gps:
                events_by_gps[gps] = e
    events = list(events_by_gps.values())

    ok_events = [e for e in events if e.get('status') == 'ok']
    complex_events = [e for e in events if e.get('status') == 'complex']

    print(f"\nTotal unique events: {len(events)}")
    print(f"  OK events: {len(ok_events)}")
    print(f"  Complex (rejected) events: {len(complex_events)}")

    # Load OK curvatures from stability analysis (deduplicated)
    ok_curvatures = {'stable_delayed': [], 'stable_fast': [], 'flip': []}
    seen_gps = set()
    with open(STABILITY_FILE) as f:
        for line in f:
            e = json.loads(line)
            gps = e.get('gps_time')
            if gps in seen_gps:
                continue
            seen_gps.add(gps)
            b = e.get('curvature_b')
            stability = e.get('stability', '')
            if b is not None:
                if 'iof' in stability.lower() or 'delayed' in stability.lower():
                    ok_curvatures['stable_delayed'].append(b)
                elif 'std' in stability.lower() or 'fast' in stability.lower():
                    ok_curvatures['stable_fast'].append(b)
                else:
                    ok_curvatures['flip'].append(b)

    print(f"\nOK events with curvature:")
    print(f"  Stable Delayed: {len(ok_curvatures['stable_delayed'])}")
    print(f"  Stable Fast: {len(ok_curvatures['stable_fast'])}")
    print(f"  Flip: {len(ok_curvatures['flip'])}")

    # Compute curvature AND classification for complex events
    print(f"\nAnalyzing complex events (curvature + model tournament)...")
    complex_curvatures = []
    complex_classified = {'delayed': [], 'fast': [], 'uncertain': []}
    complex_reasons = {'peak_dominance': 0, 'envelope_decay': 0}

    for i, event in enumerate(complex_events):
        gps_time = event['gps_time']
        reason = event.get('reason', '').split(':')[0]

        # Load strain data
        strain_data = load_cached_strain(gps_time)
        if strain_data is None:
            continue

        strain = strain_data['values']
        fs = strain_data['sample_rate']
        # CRITICAL: Convert times relative to GPS time (not segment start)
        # to match ligo_glitch_analysis.py peak localization
        times_ms = (strain_data['times'] - gps_time) * 1000

        # Compute envelope
        envelope = compute_hilbert_envelope(strain, fs)

        # Constrain peak search to ±500ms around GPS time to ensure
        # same peak as main pipeline (ligo_glitch_analysis.py)
        search_mask = np.abs(times_ms) < 500
        search_envelope = envelope.copy()
        search_envelope[~search_mask] = 0
        peak_idx = np.argmax(search_envelope)

        # Run tournament and get classification + curvature
        classification, b = fit_and_classify(envelope, times_ms, peak_idx=peak_idx)

        if b is not None and np.isfinite(b):
            complex_curvatures.append(b)

            if classification == 'delayed':
                complex_classified['delayed'].append(b)
            elif classification == 'fast':
                complex_classified['fast'].append(b)
            elif classification == 'uncertain':
                complex_classified['uncertain'].append(b)

            if 'peak_dominance' in reason:
                complex_reasons['peak_dominance'] += 1
            elif 'envelope_decay' in reason:
                complex_reasons['envelope_decay'] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(complex_events)} complex events")

    print(f"\nComplex events with valid curvature: {len(complex_curvatures)}")
    print(f"  Classified Delayed: {len(complex_classified['delayed'])}")
    print(f"  Classified Fast: {len(complex_classified['fast'])}")
    print(f"  Classified Uncertain: {len(complex_classified['uncertain'])}")
    print(f"  Peak dominance failures: {complex_reasons['peak_dominance']}")
    print(f"  Envelope decay failures: {complex_reasons['envelope_decay']}")

    # Analysis 1: Distribution comparison
    print("\n" + "=" * 70)
    print("STRESS TEST RESULTS")
    print("=" * 70)

    all_ok = ok_curvatures['stable_delayed'] + ok_curvatures['stable_fast'] + ok_curvatures['flip']

    print("\nDistribution statistics (curvature b × 10³):")
    print(f"  OK events (N={len(all_ok)}): median={np.median(all_ok)*1000:.2f}, IQR=[{np.percentile(all_ok, 25)*1000:.2f}, {np.percentile(all_ok, 75)*1000:.2f}]")
    if len(complex_curvatures) > 0:
        print(f"  Complex events (N={len(complex_curvatures)}): median={np.median(complex_curvatures)*1000:.2f}, IQR=[{np.percentile(complex_curvatures, 25)*1000:.2f}, {np.percentile(complex_curvatures, 75)*1000:.2f}]")
    else:
        print(f"  Complex events (N=0): No events passed sanity checks")

    # KS test for distribution difference
    if len(complex_curvatures) > 0:
        ks_stat, ks_p = ks_2samp(all_ok, complex_curvatures)
    else:
        ks_stat, ks_p = float('nan'), float('nan')
    print(f"\nKS test (OK vs Complex): D={ks_stat:.3f}, p={ks_p:.4f}")

    # Analysis 2: AUC comparison
    # For OK events: AUC between stable_delayed and stable_fast
    auc_ok = compute_auc(ok_curvatures['stable_delayed'], ok_curvatures['stable_fast'])
    print(f"\nOK stable-core AUC(b): {auc_ok:.3f}")

    # For complex events: split by median and compute "pseudo-AUC"
    # This shows whether complex events have ANY separation structure
    if len(complex_curvatures) > 20:
        median_b = np.median(complex_curvatures)
        high_b = [b for b in complex_curvatures if b > median_b]
        low_b = [b for b in complex_curvatures if b <= median_b]

        # This "AUC" should be ~0.5 by construction if using median split
        # but variance tells us about separation structure
        print(f"\nComplex events median split:")
        print(f"  High b (>{median_b*1000:.2f}×10⁻³): N={len(high_b)}, mean={np.mean(high_b)*1000:.2f}×10⁻³")
        print(f"  Low b (≤{median_b*1000:.2f}×10⁻³): N={len(low_b)}, mean={np.mean(low_b)*1000:.2f}×10⁻³")

    # Analysis 3: AUC(b) for classified complex events (key stress test result)
    # This runs the SAME tournament on complex events and checks if b discriminates
    auc_complex = None
    if len(complex_classified['delayed']) >= 5 and len(complex_classified['fast']) >= 5:
        auc_complex = compute_auc(complex_classified['delayed'], complex_classified['fast'])
        print(f"\n*** KEY STRESS TEST RESULT ***")
        print(f"Complex events classified by same tournament:")
        print(f"  Delayed (N={len(complex_classified['delayed'])}): median b = {np.median(complex_classified['delayed'])*1000:.2f}×10⁻³")
        print(f"  Fast (N={len(complex_classified['fast'])}): median b = {np.median(complex_classified['fast'])*1000:.2f}×10⁻³")
        print(f"  AUC(b) for Delayed vs Fast: {auc_complex:.3f}")

        if auc_complex < 0.6:
            print(f"  Result: Association absent in rejected morphology (AUC near chance)")
        else:
            print(f"  Result: Association persists in rejected morphology (AUC above chance)")
    else:
        print(f"\n*** KEY STRESS TEST RESULT ***")
        print(f"Insufficient classified complex events for AUC:")
        print(f"  Delayed: {len(complex_classified['delayed'])}, Fast: {len(complex_classified['fast'])}")

    # Analysis 4: Compare variance/spread
    ok_delayed = ok_curvatures['stable_delayed']
    ok_fast = ok_curvatures['stable_fast']

    print(f"\nComparison with OK events:")
    print(f"  OK Delayed mean: {np.mean(ok_delayed)*1000:.3f}×10⁻³")
    print(f"  OK Fast mean: {np.mean(ok_fast)*1000:.3f}×10⁻³")
    print(f"  OK Separation: {(np.mean(ok_delayed) - np.mean(ok_fast))*1000:.3f}×10⁻³")
    if len(complex_curvatures) > 0:
        print(f"  Complex mean: {np.mean(complex_curvatures)*1000:.3f}×10⁻³")
    else:
        print(f"  Complex mean: N/A (no events with valid curvature)")

    # Save results - handle empty complex_curvatures array
    if len(complex_curvatures) > 0:
        complex_median = float(np.median(complex_curvatures))
        complex_iqr = [float(np.percentile(complex_curvatures, 25)), float(np.percentile(complex_curvatures, 75))]
        complex_mean = float(np.mean(complex_curvatures))
    else:
        complex_median = None
        complex_iqr = [None, None]
        complex_mean = None

    results = {
        'n_ok': len(all_ok),
        'n_complex': len(complex_curvatures),
        'ok_stable_core_auc': auc_ok,
        'ok_median': float(np.median(all_ok)),
        'ok_iqr': [float(np.percentile(all_ok, 25)), float(np.percentile(all_ok, 75))],
        'complex_median': complex_median,
        'complex_iqr': complex_iqr,
        'ks_statistic': float(ks_stat) if not np.isnan(ks_stat) else None,
        'ks_pvalue': float(ks_p) if not np.isnan(ks_p) else None,
        'ok_delayed_mean': float(np.mean(ok_delayed)),
        'ok_fast_mean': float(np.mean(ok_fast)),
        'complex_mean': complex_mean,
        # Stress test: tournament classification on complex events
        'complex_classified_delayed': len(complex_classified['delayed']),
        'complex_classified_fast': len(complex_classified['fast']),
        'complex_classified_uncertain': len(complex_classified['uncertain']),
        'complex_auc': auc_complex,
        'note': 'Rejected-morphology stress test: same tournament on complex events'
    }

    output_file = OUTPUT_DIR / "negative_control.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")

    # Summary interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Check if complex events cluster differently
    if len(complex_curvatures) > 0:
        complex_in_ok_range = sum(1 for b in complex_curvatures
                                  if np.percentile(all_ok, 10) <= b <= np.percentile(all_ok, 90))
        pct_in_range = 100 * complex_in_ok_range / len(complex_curvatures)
        print(f"\nComplex events within OK 10-90th percentile range: {complex_in_ok_range}/{len(complex_curvatures)} ({pct_in_range:.1f}%)")
    else:
        print(f"\nComplex events within OK 10-90th percentile range: 0/0 (N/A - no complex events with valid curvature)")

    # Key interpretation: compare AUCs
    print("\n" + "-" * 50)
    print("KEY RESULT: AUC Comparison")
    print("-" * 50)
    print(f"  OK stable-core AUC(b):      {auc_ok:.3f}")
    if auc_complex is not None:
        print(f"  Complex tournament AUC(b):  {auc_complex:.3f}")

        if auc_ok > 0.7 and auc_complex < 0.6:
            print("\nSTRESS TEST RESULT: Association absent in rejected morphology")
            print("  - OK events: Strong curvature separation (AUC = {:.3f})".format(auc_ok))
            print("  - Complex events: Chance-level discrimination (AUC = {:.3f})".format(auc_complex))
            print("  Interpretation: Association is specific to single-peak morphology")
        elif auc_ok > 0.7 and auc_complex >= 0.6:
            print("\nSTRESS TEST RESULT: Association persists in rejected morphology")
            print("  - OK events: AUC = {:.3f}".format(auc_ok))
            print("  - Complex events: AUC = {:.3f}".format(auc_complex))
            print("  Interpretation: Association is not uniquely explained by morphology filter")
            print("  Note: AUC values are qualitative due to distribution shift (KS significant)")
        else:
            print("\nSTRESS TEST RESULT: Insufficient separation in OK events for comparison")
    else:
        print("\nSTRESS TEST RESULT: Could not compute complex AUC (insufficient classified events)")


if __name__ == "__main__":
    main()
