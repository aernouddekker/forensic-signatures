#!/usr/bin/env python3
"""
LIGO Alternative Likelihood Sensitivity Check
==============================================

Re-runs model classification on log(envelope) instead of raw envelope.
This transforms multiplicative noise to additive, making Gaussian SSE
more appropriate for positive-valued LIGO envelopes.

The key question: does stable-core membership change under log-domain fits?

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
import pickle
from pathlib import Path
from scipy.signal import hilbert, butter, filtfilt
from scipy.optimize import curve_fit, minimize
import warnings

warnings.filterwarnings('ignore')

# Import canonical pipeline invariants from shared module
from ligo_pipeline_common import (
    bandpass_filter,
    compute_hilbert_envelope,
    baseline_from_postpeak_window,
    find_constrained_peak,
    compute_times_ms,
    load_cached_strain as common_load_cached_strain,
    ANALYSIS_WINDOWS_MS
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
WINDOWS_MS = [60, 100, 150]
AICC_THRESHOLD = 2.0


# =============================================================================
# Signal processing - IMPORTED FROM ligo_pipeline_common.py
# =============================================================================
# bandpass_filter, compute_hilbert_envelope, baseline_from_postpeak_window
# are now imported from the canonical source. No local copies.


def load_cached_strain(gps_time):
    """Load cached strain data using canonical loader with legacy fallback."""
    return common_load_cached_strain(gps_time, CACHE_DIR)


# =============================================================================
# Model definitions (for recovery variable z in [0,1])
# =============================================================================

def exponential_recovery(t, A, tau, baseline):
    """z = baseline - A * exp(-t/tau), rises from baseline-A to baseline"""
    return baseline - A * np.exp(-t / tau)


def sigmoid_recovery(t, A, k, t0, baseline):
    """z = baseline - A / (1 + exp(k*(t-t0))), sigmoid with inflection at t0"""
    return baseline - A / (1 + np.exp(k * (t - t0)))


def delayed_exponential_recovery(t, A, tau, t_delay, baseline):
    """z = baseline - A * exp(-(t-t_delay)/tau) for t > t_delay, else baseline-A"""
    result = np.full_like(t, baseline - A, dtype=float)
    mask = t > t_delay
    result[mask] = baseline - A * np.exp(-(t[mask] - t_delay) / tau)
    return result


def compute_aicc(n, k, sse):
    """Compute corrected AIC from SSE assuming Gaussian residuals."""
    if sse <= 0 or n <= k + 1:
        return np.inf
    aic = n * np.log(sse / n) + 2 * k
    correction = 2 * k * (k + 1) / (n - k - 1)
    return aic + correction


def fit_models_to_recovery(t_ms, z, window_ms):
    """
    Fit competing models to recovery data z(t).
    Returns dict of model results with AICc, parameters, geometry.
    """
    results = {}
    n = len(z)

    # Fast models (t_inf = 0)
    # 1. Exponential
    try:
        bounds = ([0.1, 1, 0.5], [2, 200, 1.5])
        p0 = [0.8, 20, 1.0]
        popt, _ = curve_fit(exponential_recovery, t_ms, z, p0=p0, bounds=bounds, maxfev=5000)
        pred = exponential_recovery(t_ms, *popt)
        sse = np.sum((z - pred)**2)
        aicc = compute_aicc(n, 3, sse)
        results['exponential'] = {
            'aicc': aicc, 'params': popt.tolist(), 'geometry': 'fast', 't_inf': 0,
            'tau': popt[1]
        }
    except:
        pass

    # Delayed models (t_inf > 0)
    # 2. Sigmoid
    try:
        bounds = ([0.1, 0.01, 1, 0.5], [2, 1, 50, 1.5])
        p0 = [0.8, 0.1, 10, 1.0]
        popt, _ = curve_fit(sigmoid_recovery, t_ms, z, p0=p0, bounds=bounds, maxfev=5000)
        pred = sigmoid_recovery(t_ms, *popt)
        sse = np.sum((z - pred)**2)
        aicc = compute_aicc(n, 4, sse)
        t_inf = popt[2]  # t0 is inflection
        tau = 1 / popt[1]  # tau = 1/k
        if t_inf > 0 and t_inf < window_ms:
            results['sigmoid'] = {
                'aicc': aicc, 'params': popt.tolist(), 'geometry': 'delayed',
                't_inf': t_inf, 'tau': tau
            }
    except:
        pass

    # 3. Delayed exponential
    try:
        bounds = ([0.1, 1, 1, 0.5], [2, 200, 40, 1.5])
        p0 = [0.8, 20, 5, 1.0]
        popt, _ = curve_fit(delayed_exponential_recovery, t_ms, z, p0=p0, bounds=bounds, maxfev=5000)
        pred = delayed_exponential_recovery(t_ms, *popt)
        sse = np.sum((z - pred)**2)
        aicc = compute_aicc(n, 4, sse)
        t_inf = popt[2]  # t_delay
        tau = popt[1]
        if t_inf > 0 and t_inf < window_ms:
            results['delayed_exp'] = {
                'aicc': aicc, 'params': popt.tolist(), 'geometry': 'delayed',
                't_inf': t_inf, 'tau': tau
            }
    except:
        pass

    return results


def classify_event(model_results, threshold=AICC_THRESHOLD):
    """
    Classify event as 'iof' (delayed), 'std' (fast), or 'uncertain'.
    """
    if not model_results:
        return 'failed', None

    fast_models = [r for r in model_results.values() if r['geometry'] == 'fast']
    delayed_models = [r for r in model_results.values() if r['geometry'] == 'delayed']

    if not fast_models and not delayed_models:
        return 'failed', None

    best_fast = min([m['aicc'] for m in fast_models]) if fast_models else np.inf
    best_delayed = min([m['aicc'] for m in delayed_models]) if delayed_models else np.inf

    delta = best_fast - best_delayed

    if delta >= threshold:
        return 'iof', delta
    elif delta <= -threshold:
        return 'std', delta
    else:
        return 'uncertain', delta


def analyze_event_log_domain(strain_data, window_ms, gps_time):
    """
    Analyze event in log-domain (log-envelope).
    Returns classification and delta_aicc.
    """
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
    peak_time = times_ms[peak_idx]
    peak_val = envelope[peak_idx]
    # Baseline from tail of analysis window (matches main pipeline)
    baseline = baseline_from_postpeak_window(envelope, times_ms, peak_idx, window_ms)

    if peak_val <= baseline * 1.1:
        return 'failed', None

    # Extract window after peak
    mask = (times_ms >= peak_time) & (times_ms <= peak_time + window_ms)
    t_window = times_ms[mask] - peak_time
    env_window = envelope[mask]

    if len(t_window) < 20:
        return 'failed', None

    # Convert to log-domain recovery variable
    # log(E) ranges from log(peak) to log(baseline)
    # Normalize to z in [0, 1]: z = 1 - (log(E) - log(baseline)) / (log(peak) - log(baseline))
    # This is equivalent to z = 1 - log(E/baseline) / log(peak/baseline)

    eps = 1e-10  # Numerical stability
    log_peak = np.log(peak_val + eps)
    log_baseline = np.log(baseline + eps)
    log_env = np.log(env_window + eps)

    if log_peak <= log_baseline:
        return 'failed', None

    z = 1 - (log_env - log_baseline) / (log_peak - log_baseline)

    # Fit models
    model_results = fit_models_to_recovery(t_window, z, window_ms)

    # Classify
    return classify_event(model_results)


def main():
    print("=" * 70)
    print("LIGO Alternative Likelihood Sensitivity Check")
    print("(Log-domain envelope fitting)")
    print("=" * 70)

    # Load original classifications
    original = {}
    with open(STABILITY_FILE) as f:
        for line in f:
            e = json.loads(line)
            original[e['gps_time']] = e['stability']

    print(f"\nLoaded {len(original)} original classifications")

    # Load OK events
    ok_events = []
    with open(RESULTS_FILE) as f:
        for line in f:
            e = json.loads(line)
            if e.get('status') == 'ok':
                ok_events.append(e)

    print(f"Processing {len(ok_events)} OK events in log-domain...")

    # Analyze each event at all windows
    log_results = {}

    for i, event in enumerate(ok_events):
        gps_time = event['gps_time']

        strain_data = load_cached_strain(gps_time)
        if strain_data is None:
            continue

        window_classifications = []

        for window_ms in WINDOWS_MS:
            classification, delta = analyze_event_log_domain(strain_data, window_ms, gps_time)
            window_classifications.append(classification)

        # Determine stability
        if 'failed' in window_classifications:
            stability = 'failed'
        elif all(c == 'iof' for c in window_classifications):
            stability = 'stable_iof'
        elif all(c == 'std' for c in window_classifications):
            stability = 'stable_std'
        else:
            stability = 'flip'

        log_results[gps_time] = stability

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(ok_events)} events")

    print(f"\nLog-domain classification complete: {len(log_results)} events")

    # Compare to original
    print("\n" + "=" * 70)
    print("COMPARISON: Original vs Log-Domain Classification")
    print("=" * 70)

    # Count by stability
    orig_counts = {'stable_iof': 0, 'stable_std': 0, 'flip': 0, 'failed': 0}
    log_counts = {'stable_iof': 0, 'stable_std': 0, 'flip': 0, 'failed': 0}

    for gps, orig_stab in original.items():
        if 'iof' in orig_stab.lower() or 'delayed' in orig_stab.lower():
            orig_counts['stable_iof'] += 1
        elif 'std' in orig_stab.lower() or 'fast' in orig_stab.lower():
            orig_counts['stable_std'] += 1
        elif 'flip' in orig_stab.lower():
            orig_counts['flip'] += 1
        else:
            orig_counts['failed'] += 1

    for gps, log_stab in log_results.items():
        log_counts[log_stab] += 1

    print("\nStability counts:")
    print(f"  Original: Delayed={orig_counts['stable_iof']}, Fast={orig_counts['stable_std']}, Flip={orig_counts['flip']}")
    print(f"  Log-domain: Delayed={log_counts['stable_iof']}, Fast={log_counts['stable_std']}, Flip={log_counts['flip']}, Failed={log_counts['failed']}")

    # Agreement analysis
    agreements = {'same': 0, 'different': 0, 'missing': 0}
    transitions = {'delayed_to_fast': 0, 'fast_to_delayed': 0, 'stable_to_flip': 0, 'flip_to_stable': 0}

    for gps, orig_stab in original.items():
        if gps not in log_results:
            agreements['missing'] += 1
            continue

        log_stab = log_results[gps]

        # Normalize original
        if 'iof' in orig_stab.lower() or 'delayed' in orig_stab.lower():
            orig_norm = 'delayed'
        elif 'std' in orig_stab.lower() or 'fast' in orig_stab.lower():
            orig_norm = 'fast'
        elif 'flip' in orig_stab.lower():
            orig_norm = 'flip'
        else:
            orig_norm = 'other'

        # Normalize log
        if log_stab == 'stable_iof':
            log_norm = 'delayed'
        elif log_stab == 'stable_std':
            log_norm = 'fast'
        elif log_stab == 'flip':
            log_norm = 'flip'
        else:
            log_norm = 'other'

        if orig_norm == log_norm:
            agreements['same'] += 1
        else:
            agreements['different'] += 1

            if orig_norm == 'delayed' and log_norm == 'fast':
                transitions['delayed_to_fast'] += 1
            elif orig_norm == 'fast' and log_norm == 'delayed':
                transitions['fast_to_delayed'] += 1
            elif orig_norm in ['delayed', 'fast'] and log_norm == 'flip':
                transitions['stable_to_flip'] += 1
            elif orig_norm == 'flip' and log_norm in ['delayed', 'fast']:
                transitions['flip_to_stable'] += 1

    total_compared = agreements['same'] + agreements['different']
    agreement_pct = 100 * agreements['same'] / total_compared if total_compared > 0 else 0

    print(f"\nAgreement analysis:")
    print(f"  Same classification: {agreements['same']}/{total_compared} ({agreement_pct:.1f}%)")
    print(f"  Different classification: {agreements['different']}/{total_compared}")
    print(f"  Missing in log-domain: {agreements['missing']}")

    print(f"\nTransitions:")
    print(f"  Delayed → Fast: {transitions['delayed_to_fast']}")
    print(f"  Fast → Delayed: {transitions['fast_to_delayed']}")
    print(f"  Stable → Flip: {transitions['stable_to_flip']}")
    print(f"  Flip → Stable: {transitions['flip_to_stable']}")

    # Save results
    results = {
        'n_original': len(original),
        'n_log_domain': len(log_results),
        'original_counts': orig_counts,
        'log_domain_counts': log_counts,
        'agreement_pct': agreement_pct,
        'agreements': agreements,
        'transitions': transitions,
        'note': 'Alternative likelihood: log-domain envelope fitting'
    }

    output_file = OUTPUT_DIR / "alternative_likelihood.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if agreement_pct >= 80:
        print(f"\nResult: HIGH AGREEMENT ({agreement_pct:.1f}%)")
        print(f"  - Stable-core membership is robust to likelihood specification")
        print(f"  - Population structure persists under log-domain transformation")
    elif agreement_pct >= 60:
        print(f"\nResult: MODERATE AGREEMENT ({agreement_pct:.1f}%)")
        print(f"  - Some sensitivity to likelihood specification")
        print(f"  - Core populations persist but boundary events shift")
    else:
        print(f"\nResult: LOW AGREEMENT ({agreement_pct:.1f}%)")
        print(f"  - Results are sensitive to likelihood specification")
        print(f"  - Interpretation should account for noise model uncertainty")


if __name__ == "__main__":
    main()
