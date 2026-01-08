#!/usr/bin/env python3
"""
LIGO Envelope-Based Classification
===================================

Improved LIGO glitch analysis using:
1. Hilbert envelope (not |strain|) to avoid zero-crossing artifacts
2. Relaxed sanity checks: peak dominance + envelope decay trend
3. Model-based classification (AICc tournament)

Key improvements over ligo_model_based.py:
- Hilbert envelope respects oscillatory transients without folding artifacts
- Peak dominance check allows ringing but rejects multi-component events
- Theil-Sen slope ensures overall decay without requiring strict monotonicity

Usage:
    python ligo_glitch_analysis.py --n_events 500 --classes Blip
    python ligo_glitch_analysis.py --n_events 500 --classes Tomte Blip
    python ligo_glitch_analysis.py --n_events 0 --classes Extremely_Loud  # All events

Author: Aernoud Dekker
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import json
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.signal import hilbert, find_peaks, butter, filtfilt
from scipy.stats import theilslopes
import warnings
import hashlib

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import from iof_metrics
from iof_metrics import (
    FitParams, FitResult, NumpyEncoder,
    exponential_recovery, exponential_recovery_fixed,
    power_law_recovery, logistic_sigmoid, delayed_exponential,
    compute_aic, compute_aicc, fit_model,
    compute_model_t_peak, get_model_geometry,
    baseline_tail_median,  # Canonical baseline function
)

# Import canonical pipeline invariants from shared module
from ligo_pipeline_common import (
    bandpass_filter,
    compute_hilbert_envelope,
    distance_samples_from_times,
    compute_times_ms,
    find_constrained_peak,
    baseline_from_postpeak_window,
    load_cached_strain as common_load_cached_strain,
    load_strain_from_bulk,
    PEAK_SEARCH_WINDOW_MS,
)

# Check for gwpy
try:
    from gwpy.timeseries import TimeSeries
    GWPY_AVAILABLE = True
except ImportError:
    GWPY_AVAILABLE = False
    print("Warning: gwpy not installed. Install with: pip install gwpy")


# =============================================================================
# Signal Processing - IMPORTED FROM ligo_pipeline_common.py
# =============================================================================
# bandpass_filter, compute_hilbert_envelope, distance_samples_from_times
# are now imported from the canonical source. No local copies.


# =============================================================================
# Improved Sanity Checks
# =============================================================================


def check_peak_dominance(envelope: np.ndarray,
                         times_ms: np.ndarray,
                         peak_ratio_threshold: float = 0.7,
                         max_secondary_peaks: int = 3,
                         secondary_threshold: float = 0.5,
                         min_peak_sep_ms: float = 2.5) -> Tuple[bool, str, dict]:
    """
    Check if event has a single dominant peak (allows damped oscillations).

    Args:
        envelope: Hilbert envelope of signal
        times_ms: Time array in milliseconds (for sample-rate-independent distance)
        peak_ratio_threshold: Reject if p2/p1 > this value
        max_secondary_peaks: Max allowed peaks above secondary_threshold * p1
        secondary_threshold: Fraction of max for secondary peak counting
        min_peak_sep_ms: Minimum separation between peaks in ms (default 2.5)

    Returns:
        (is_dominant, reason, metrics)
    """
    if len(envelope) < 50:
        return False, "insufficient_data", {}

    # Find all peaks with sample-rate-independent distance
    prominence = 0.1 * np.max(envelope)
    distance = distance_samples_from_times(times_ms, min_peak_sep_ms)
    peaks, properties = find_peaks(envelope, prominence=prominence, distance=distance)

    if len(peaks) == 0:
        return False, "no_peaks_found", {}

    # Get peak heights
    peak_heights = envelope[peaks]
    sorted_heights = np.sort(peak_heights)[::-1]

    p1 = sorted_heights[0]  # Max peak

    metrics = {
        'n_peaks': len(peaks),
        'p1': float(p1),
        'peak_heights': [float(h) for h in sorted_heights[:5]],
        'peak_distance_samples': int(distance),
        'min_peak_sep_ms': float(min_peak_sep_ms)
    }

    # Check 1: Is second peak too large?
    if len(sorted_heights) >= 2:
        p2 = sorted_heights[1]
        ratio = p2 / p1
        metrics['p2'] = float(p2)
        metrics['p2_p1_ratio'] = float(ratio)

        if ratio > peak_ratio_threshold:
            return False, f"secondary_peak_too_large (p2/p1={ratio:.2f}>{peak_ratio_threshold})", metrics

    # Check 2: Too many significant secondary peaks?
    n_significant = np.sum(peak_heights > secondary_threshold * p1)
    metrics['n_significant_peaks'] = int(n_significant)

    if n_significant > max_secondary_peaks:
        return False, f"too_many_peaks ({n_significant}>{max_secondary_peaks})", metrics

    return True, "dominant_peak_ok", metrics


def check_envelope_decay(envelope: np.ndarray,
                         times_ms: np.ndarray,
                         peak_idx: int,
                         window_ms: float = 100.0,
                         min_decay_slope: float = -0.001) -> Tuple[bool, str, dict]:
    """
    Check if envelope shows overall decay after peak using Theil-Sen robust regression.

    Args:
        envelope: Hilbert envelope
        times_ms: Time array in milliseconds
        peak_idx: Index of main peak
        window_ms: Window for decay check (ms)
        min_decay_slope: Minimum negative slope (log-envelope per ms)

    Returns:
        (is_decaying, reason, metrics)
    """
    # Get post-peak data
    post_peak_mask = (times_ms >= times_ms[peak_idx]) & (times_ms <= times_ms[peak_idx] + window_ms)

    if np.sum(post_peak_mask) < 20:
        return False, "insufficient_post_peak_data", {}

    t_post = times_ms[post_peak_mask] - times_ms[peak_idx]
    env_post = envelope[post_peak_mask]

    # Log-envelope for exponential decay check
    # Add small offset to avoid log(0)
    log_env = np.log(env_post + 1e-30)

    # Theil-Sen robust regression (resistant to outliers)
    try:
        slope, intercept, low_slope, high_slope = theilslopes(log_env, t_post)
    except Exception:
        return False, "theilsen_failed", {}

    metrics = {
        'decay_slope': float(slope),
        'slope_ci_low': float(low_slope),
        'slope_ci_high': float(high_slope),
        'decay_halflife_ms': float(-np.log(2) / slope) if slope < 0 else float('inf')
    }

    # Check if slope is sufficiently negative
    if slope > min_decay_slope:
        return False, f"envelope_not_decaying (slope={slope:.4f}>{min_decay_slope})", metrics

    return True, "envelope_decay_ok", metrics


def run_sanity_checks(envelope: np.ndarray,
                      times_ms: np.ndarray,
                      peak_ratio_threshold: float = 0.7,
                      max_secondary_peaks: int = 3,
                      min_decay_slope: float = -0.001,
                      window_ms: float = 100.0) -> Tuple[str, str, dict]:
    """
    Run all sanity checks on an event.

    Returns:
        (status, reason, all_metrics)
        status: 'ok', 'complex', or 'failed'
    """
    all_metrics = {}

    # Find main peak
    peak_idx = np.argmax(envelope)
    all_metrics['peak_idx'] = int(peak_idx)
    all_metrics['peak_time_ms'] = float(times_ms[peak_idx])
    all_metrics['peak_value'] = float(envelope[peak_idx])

    # Check 1: Peak dominance (sample-rate-independent distance)
    is_dominant, dom_reason, dom_metrics = check_peak_dominance(
        envelope, times_ms, peak_ratio_threshold, max_secondary_peaks
    )
    all_metrics['peak_dominance'] = dom_metrics
    all_metrics['peak_dominance_reason'] = dom_reason

    if not is_dominant:
        return 'complex', f"peak_dominance: {dom_reason}", all_metrics

    # Check 2: Envelope decay
    is_decaying, decay_reason, decay_metrics = check_envelope_decay(
        envelope, times_ms, peak_idx, window_ms, min_decay_slope
    )
    all_metrics['envelope_decay'] = decay_metrics
    all_metrics['envelope_decay_reason'] = decay_reason

    if not is_decaying:
        return 'complex', f"envelope_decay: {decay_reason}", all_metrics

    return 'ok', 'passed_all_checks', all_metrics


# =============================================================================
# Model Fitting (same as before but with envelope input)
# =============================================================================

def fit_competing_models_envelope(
    t_ms: np.ndarray,
    y: np.ndarray,
    baseline_estimate: float,
    params: FitParams,
    model_set: Set[str] = None
) -> Dict[str, FitResult]:
    """
    Fit competing models to envelope recovery data.

    Envelope decays from peak to baseline, so we invert to recovery view.
    """
    if model_set is None:
        model_set = {'exponential', 'exponential_fixed', 'power_law', 'sigmoid', 'delayed'}

    results = {}

    if len(y) < 20:
        return results

    max_value = y[0]  # Should be at peak
    amplitude = max_value - baseline_estimate

    if amplitude < 0.01 * max_value:  # Less than 1% amplitude
        return results

    # Compute fixed baseline from late-time data if requested
    if params.fix_baseline:
        late_start = int(len(y) * (1 - params.baseline_fraction))
        fixed_baseline = np.median(y[late_start:])
    else:
        fixed_baseline = baseline_estimate

    # Invert to recovery view: 0 at baseline, 1 at peak
    # recovery = (y - baseline) / amplitude
    # But models expect: y = baseline - A*exp(-t/tau) starting at baseline-A and going to baseline
    # For envelope decay: envelope goes from max to baseline
    # So we model: y = baseline + A*exp(-t/tau) where A = peak - baseline

    # Actually, let's invert to standard recovery form used in McEwan:
    # recovery_view = 1 - (y - baseline) / (max - baseline)
    # This goes from 0 at peak to 1 at baseline

    recovery = 1 - (y - fixed_baseline) / (max_value - fixed_baseline + 1e-30)
    recovery = np.clip(recovery, 0, 1.5)  # Allow slight overshoot but cap

    # Now fit recovery models
    min_value = recovery[0]  # Should be ~0
    amplitude_r = 1.0 - min_value  # Should be ~1

    # --- Exponential (free baseline) ---
    if 'exponential' in model_set:
        results['exponential'] = fit_model(
            t_ms, recovery,
            exponential_recovery,
            p0=[amplitude_r, 15.0, 1.0],
            bounds=([0, 0.5, 0.5], [2.0, 200, 1.5]),
            model_name='exponential',
            n_params=3,
            maxfev=params.maxfev
        )

    # --- Exponential (fixed baseline = 1) ---
    if 'exponential_fixed' in model_set:
        results['exponential_fixed'] = fit_model(
            t_ms, recovery,
            exponential_recovery_fixed(1.0),
            p0=[amplitude_r, 15.0],
            bounds=([0, 0.5], [2.0, 200]),
            model_name='exponential_fixed',
            n_params=2,
            maxfev=params.maxfev
        )

    # --- Power Law ---
    if 'power_law' in model_set:
        results['power_law'] = fit_model(
            t_ms, recovery,
            power_law_recovery,
            p0=[amplitude_r, 10.0, 1.0],
            bounds=([0, 0.5, 0.5], [2.0, 200, 1.5]),
            model_name='power_law',
            n_params=3,
            maxfev=params.maxfev
        )

    # --- Sigmoid (IOF) ---
    if 'sigmoid' in model_set:
        results['sigmoid'] = fit_model(
            t_ms, recovery,
            logistic_sigmoid,
            p0=[amplitude_r, 0.15, 20.0, min_value],
            bounds=([0, 0.01, 1, -0.5], [2.0, 2, 80, 0.5]),
            model_name='sigmoid',
            n_params=4,
            maxfev=params.maxfev
        )

    # --- Delayed Exponential (IOF) ---
    if 'delayed' in model_set:
        results['delayed'] = fit_model(
            t_ms, recovery,
            delayed_exponential,
            p0=[amplitude_r, 20.0, 5.0, min_value],
            bounds=([0, 0.5, 0, -0.5], [2.0, 200, 50, 0.5]),
            model_name='delayed',
            n_params=4,
            maxfev=params.maxfev
        )

    return results


def check_fit_sanity(fit: FitResult) -> Tuple[bool, str]:
    """Check if a fit result is physically sensible."""
    if not fit.success:
        return False, "fit_failed"

    if fit.r2 < 0.3:
        return False, f"low_r2_{fit.r2:.2f}"

    params = fit.params
    if 'tau' in params:
        tau = params['tau']
        if tau < 0.5 or tau > 180:
            return False, f"tau_out_of_range_{tau:.1f}"

    if 'delay' in params:
        delay = params['delay']
        if delay < 0 or delay > 80:
            return False, f"delay_out_of_range_{delay:.1f}"

    if 't0' in params:
        t0 = params['t0']
        if t0 < 0 or t0 > 80:
            return False, f"t0_out_of_range_{t0:.1f}"

    return True, "ok"


def classify_event_model_based(
    fits: Dict[str, FitResult]
) -> Tuple[str, str, str, str, float, float, float, float]:
    """
    Classify event using model-based approach.

    Returns:
        classification, reason, winning_model, runner_up, model_t_peak, tau, H, delta_aicc
    """
    valid_fits = {}
    for name, fit in fits.items():
        is_sane, reason = check_fit_sanity(fit)
        if is_sane:
            valid_fits[name] = fit

    if not valid_fits:
        return 'failed', 'no valid fits', 'none', 'none', 0.0, 0.0, 0.0, 0.0

    sorted_fits = sorted(valid_fits.items(), key=lambda x: x[1].aicc)
    best_name, best_fit = sorted_fits[0]

    if len(sorted_fits) >= 2:
        runner_up_name, runner_up_fit = sorted_fits[1]
        delta_aicc = runner_up_fit.aicc - best_fit.aicc
    else:
        runner_up_name = 'none'
        delta_aicc = float('inf')

    model_t_peak = compute_model_t_peak(best_name, best_fit.params)

    tau = best_fit.params.get('tau', 0.0)
    if tau == 0 and 'k' in best_fit.params:
        tau = 4.0 / best_fit.params['k']

    H = model_t_peak / tau if tau > 0 else 0.0

    geometry = get_model_geometry(best_name)

    reasons = []
    if delta_aicc < 2.0:
        if len(sorted_fits) >= 2:
            geometries = [get_model_geometry(name) for name, _ in sorted_fits[:2]]
            if geometries[0] != geometries[1]:
                reasons.append(f"ambiguous (dAICc={delta_aicc:.1f}<2)")
                return ('uncertain', "; ".join(reasons), best_name, runner_up_name,
                       model_t_peak, tau, H, delta_aicc)

    if geometry == 'fast':
        classification = 'standard'
        reasons.append(f"fast ({best_name})")
    elif geometry == 'delayed':
        classification = 'iof'
        reasons.append(f"delayed ({best_name})")
    else:
        classification = 'uncertain'
        reasons.append(f"unknown ({best_name})")

    reasons.append(f"dAICc={delta_aicc:.1f}")

    return (classification, "; ".join(reasons), best_name, runner_up_name,
           model_t_peak, tau, H, delta_aicc)


# =============================================================================
# Data Loading
# =============================================================================

def load_gravity_spy_data(csv_path: Path) -> pd.DataFrame:
    """Load Gravity Spy glitch classifications."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} glitches from {csv_path}")
    return df


def get_events_by_class(
    df: pd.DataFrame,
    classes: List[str],
    n_events: int = 0,
    min_snr: float = 50,
    min_confidence: float = 0.8,
    seed: int = 42
) -> pd.DataFrame:
    """Get events from specified Gravity Spy classes."""
    mask = (
        (df['ml_label'].isin(classes)) &
        (df['snr'] >= min_snr) &
        (df['ml_confidence'] >= min_confidence)
    )
    filtered = df[mask].copy()
    raw_n = len(filtered)

    print(f"Filtered to {raw_n} events (classes={classes}, SNR>={min_snr}, conf>={min_confidence})")

    # === Deduplicate by (ifo, event_time) to handle catalog duplicates ===
    # Gravity Spy can have multiple rows for the same glitch (different annotations)
    # Resolution: keep max SNR, then max ml_confidence, then first row (deterministic)
    filtered = filtered.reset_index(drop=False).rename(columns={"index": "_row"})

    key_cols = ["event_time"]
    if "ifo" in filtered.columns:
        key_cols = ["ifo", "event_time"]

    # Count duplicates before deduping
    dup_counts = filtered.groupby(key_cols).size().sort_values(ascending=False)
    max_mult = int(dup_counts.iloc[0]) if len(dup_counts) > 0 else 1
    worst_key = dup_counts.index[0] if len(dup_counts) > 0 else None

    # Sort for deterministic tie-breaking: max snr, max ml_confidence, first row
    filtered = (
        filtered.sort_values(
            ["snr", "ml_confidence", "_row"],
            ascending=[False, False, True],
            kind="mergesort"  # stable sort
        )
        .drop_duplicates(subset=key_cols, keep="first")
        .drop(columns=["_row"])
    )

    unique_n = len(filtered)
    print(f"[dedupe] raw rows: {raw_n}, unique keys: {unique_n} (key={key_cols})")
    if max_mult > 1:
        print(f"[dedupe] max multiplicity: {max_mult} for {worst_key}")

    if n_events > 0 and len(filtered) > n_events:
        filtered = filtered.sample(n=n_events, random_state=seed)
        print(f"Sampled {n_events} events")

    return filtered.sort_values('snr', ascending=False).reset_index(drop=True)


def fetch_strain_cached(
    gps_time: float,
    cache_dir: Path,
    bulk_dir: Optional[Path] = None,
    window_before: float = 2.0,
    window_after: float = 5.0
) -> Optional[dict]:
    """
    Fetch strain data with multi-level fallback:
    1. Pickle cache (fastest)
    2. Bulk HDF5 files (if bulk_dir provided)
    3. GWOSC API fetch (slowest - downloads 67MB per 7-second request!)
    """
    # Level 1: Try pickle cache (fastest)
    cache_file_new = cache_dir / f"strain_{gps_time:.6f}.pkl"
    cache_file_legacy = cache_dir / f"strain_{int(gps_time)}.pkl"

    if cache_file_new.exists():
        try:
            with open(cache_file_new, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass

    if cache_file_legacy.exists():
        try:
            with open(cache_file_legacy, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass

    # Level 2: Try bulk HDF5 files (if available)
    if bulk_dir is not None:
        bulk_data = load_strain_from_bulk(gps_time, bulk_dir, window_before, window_after)
        if bulk_data is not None:
            # Optionally cache to pickle for future runs
            try:
                with open(cache_file_new, 'wb') as f:
                    pickle.dump(bulk_data, f)
            except Exception:
                pass
            return bulk_data

    # Level 3: Fetch from GWOSC API (slowest - downloads 67MB file!)
    if not GWPY_AVAILABLE:
        return None

    cache_file = cache_file_new
    start = gps_time - window_before
    end = gps_time + window_after

    try:
        strain = TimeSeries.fetch_open_data('H1', start, end)

        cache_data = {
            'times': strain.times.value,
            'values': strain.value,
            'gps_time': gps_time,
            'sample_rate': strain.sample_rate.value
        }

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass

        return cache_data

    except Exception as e:
        return None


# =============================================================================
# Main Processing
# =============================================================================

def process_single_event(
    strain_data: dict,
    gps_time: float,
    event_id: int,
    row: pd.Series,
    fit_params: FitParams,
    window_ms: float = 100.0,
    peak_ratio_threshold: float = 0.7,
    bandpass: bool = True
) -> Optional[dict]:
    """Process a single LIGO event with envelope-based classification."""
    if strain_data is None:
        return None

    times = strain_data['times']
    raw_strain = strain_data['values']
    sample_rate = strain_data['sample_rate']

    # Compute Hilbert envelope
    envelope = compute_hilbert_envelope(raw_strain, sample_rate, bandpass=bandpass)

    # Convert to ms relative to GPS time
    times_ms = (times - gps_time) * 1000

    # Find region around glitch (within 500ms of reported time)
    mask = np.abs(times_ms) < 500
    if not np.any(mask):
        return {'status': 'failed', 'reason': 'no_data_near_gps_time', 'event_id': event_id}

    local_times = times_ms[mask]
    local_envelope = envelope[mask]

    # Run sanity checks
    status, reason, sanity_metrics = run_sanity_checks(
        local_envelope, local_times,
        peak_ratio_threshold=peak_ratio_threshold,
        window_ms=window_ms
    )

    base_result = {
        'event_id': event_id,
        'gps_time': float(gps_time),
        'snr': float(row['snr']),
        'ml_label': row['ml_label'],
        'ml_confidence': float(row['ml_confidence']),
        'peak_frequency': float(row.get('peak_frequency', np.nan)),
        'duration': float(row.get('duration', np.nan)),
        'sanity_metrics': sanity_metrics
    }

    if status != 'ok':
        base_result['status'] = status
        base_result['reason'] = reason
        return base_result

    # Extract post-peak window for fitting
    peak_idx = sanity_metrics['peak_idx']
    peak_time_ms = local_times[peak_idx]

    post_peak_mask = (local_times >= peak_time_ms) & (local_times <= peak_time_ms + window_ms)

    if np.sum(post_peak_mask) < 50:
        base_result['status'] = 'failed'
        base_result['reason'] = 'insufficient_post_peak_data'
        return base_result

    t_fit = local_times[post_peak_mask] - peak_time_ms  # Start at 0
    env_fit = local_envelope[post_peak_mask]

    # Canonical baseline from tail of fit window (n//5 rule)
    baseline = baseline_tail_median(env_fit)

    # Fit models
    fits = fit_competing_models_envelope(t_fit, env_fit, baseline, fit_params)

    if not fits:
        base_result['status'] = 'failed'
        base_result['reason'] = 'all_fits_failed'
        return base_result

    # Classify
    (classification, class_reason, winning_model, runner_up,
     model_t_peak, tau, H, delta_aicc) = classify_event_model_based(fits)

    # Get all AICc and R² values
    aicc_values = {name: fit.aicc for name, fit in fits.items() if fit.success}
    r2_values = {name: fit.r2 for name, fit in fits.items() if fit.success}

    base_result.update({
        'status': 'ok',
        'classification': classification,
        'classification_reason': class_reason,
        'winning_model': winning_model,
        'runner_up': runner_up,
        'delta_aicc': float(delta_aicc),
        'model_t_peak_ms': float(model_t_peak),
        'tau_ms': float(tau),
        'H': float(H),
        'aicc_values': aicc_values,
        'r2_values': r2_values,
        'winning_params': fits[winning_model].params if winning_model in fits else {},
        'peak_envelope': float(sanity_metrics['peak_value']),
        'baseline_envelope': float(baseline)
    })

    return base_result


def run_analysis(
    csv_path: Path,
    output_dir: Path,
    cache_dir: Path,
    classes: List[str],
    n_events: int,
    min_snr: float,
    window_ms: float,
    fix_baseline: bool,
    peak_ratio_threshold: float,
    bandpass: bool,
    seed: int,
    bulk_dir: Optional[Path] = None
):
    """Run full analysis pipeline."""

    df = load_gravity_spy_data(csv_path)
    events = get_events_by_class(df, classes, n_events, min_snr, seed=seed)

    if len(events) == 0:
        print("No events to process!")
        return None, None

    fit_params = FitParams(fix_baseline=fix_baseline)
    cache_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    results = []
    status_counts = defaultdict(int)

    print(f"\nProcessing {len(events)} events...")
    print(f"Classes: {classes}")
    print(f"Using Hilbert envelope (bandpass={bandpass})")
    print(f"Peak ratio threshold: {peak_ratio_threshold}")
    if bulk_dir:
        print(f"Bulk data dir: {bulk_dir}")
    print("=" * 60)
    sys.stdout.flush()

    class_running = defaultdict(int)

    for i, (idx, row) in enumerate(events.iterrows()):
        event_num = i + 1
        gps = row['event_time']

        # Fetch strain data (tries: pickle cache -> bulk HDF5 -> GWOSC API)
        strain_data = fetch_strain_cached(gps, cache_dir, bulk_dir)

        if strain_data is None:
            status_counts['fetch_failed'] += 1
            print(f"[{event_num:4d}/{len(events)}] GPS {gps:.1f} - FETCH FAILED")
            sys.stdout.flush()
            continue

        result = process_single_event(
            strain_data, gps, event_num, row,
            fit_params, window_ms, peak_ratio_threshold, bandpass
        )

        if result:
            results.append(result)
            status_counts[result['status']] += 1

            # Update running classification counts
            if result['status'] == 'ok':
                cls = result.get('classification', 'unknown')
                class_running[cls] += 1
                cls_str = f"{cls.upper():8s}"
            else:
                cls_str = f"{result['status']:8s}"

            # Print progress with running totals
            ok_count = len([r for r in results if r['status'] == 'ok'])
            iof_count = class_running['iof']
            std_count = class_running['standard']
            print(f"[{event_num:4d}/{len(events)}] GPS {gps:.1f} - {cls_str} | OK: {ok_count}, IOF: {iof_count}, STD: {std_count}")
            sys.stdout.flush()

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {len(results)}")
    for status, count in sorted(status_counts.items()):
        pct = 100 * count / len(results) if results else 0
        print(f"  {status}: {count} ({pct:.1f}%)")

    ok_results = [r for r in results if r['status'] == 'ok']
    pass_rate = 100 * len(ok_results) / len(results) if results else 0
    print(f"\nPass rate: {pass_rate:.1f}%")

    if ok_results:
        print(f"\n{'='*60}")
        print("CLASSIFICATION SUMMARY (passed events only)")
        print(f"{'='*60}")

        class_counts = defaultdict(int)
        for r in ok_results:
            class_counts[r['classification']] += 1

        total = len(ok_results)
        for cls in ['standard', 'iof', 'uncertain', 'failed']:
            count = class_counts[cls]
            pct = 100 * count / total if total > 0 else 0
            print(f"  {cls.upper()}: {count} ({pct:.1f}%)")

        classified = class_counts['standard'] + class_counts['iof']
        if classified > 0:
            iof_of_classified = 100 * class_counts['iof'] / classified
            print(f"\n  IOF of classified: {iof_of_classified:.1f}%")

        delta_aiccs = [r['delta_aicc'] for r in ok_results if r['delta_aicc'] < 1000]
        if delta_aiccs:
            print(f"\n  Delta AICc distribution:")
            print(f"    Median: {np.median(delta_aiccs):.1f}")
            strong = sum(1 for d in delta_aiccs if d >= 10)
            moderate = sum(1 for d in delta_aiccs if 4 <= d < 10)
            weak = sum(1 for d in delta_aiccs if d < 2)
            print(f"    Strong (>=10): {100*strong/len(delta_aiccs):.1f}%")
            print(f"    Moderate (4-10): {100*moderate/len(delta_aiccs):.1f}%")
            print(f"    Weak (<2): {100*weak/len(delta_aiccs):.1f}%")

        iof_results = [r for r in ok_results if r['classification'] == 'iof']
        if iof_results:
            H_values = [r['H'] for r in iof_results if r['H'] > 0]
            if H_values:
                print(f"\n  Dimensionless hesitation (H = t_peak/tau) for IOF:")
                print(f"    n = {len(H_values)}")
                print(f"    Median: {np.median(H_values):.3f}")
                print(f"    IQR: [{np.percentile(H_values, 25):.3f}, {np.percentile(H_values, 75):.3f}]")

        print(f"\n  Winning models:")
        model_counts = defaultdict(int)
        for r in ok_results:
            model_counts[r['winning_model']] += 1
        for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
            print(f"    {model}: {count} ({100*count/total:.1f}%)")

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")

    # === Writer-side guardrail: assert no duplicate GPS times ===
    gps_times = [r['gps_time'] for r in results]
    unique_gps = set(gps_times)
    if len(gps_times) != len(unique_gps):
        dup_count = len(gps_times) - len(unique_gps)
        from collections import Counter
        gps_counts = Counter(gps_times)
        worst_gps, worst_n = gps_counts.most_common(1)[0]
        raise AssertionError(
            f"Duplicate GPS times in results! {dup_count} duplicates found. "
            f"Worst offender: GPS {worst_gps} appears {worst_n} times. "
            f"Check input deduplication in get_events_by_class()."
        )

    # Create class-specific output name
    class_str = "_".join(classes)
    jsonl_path = output_dir / f'ligo_envelope_{class_str}_results.jsonl'
    with open(jsonl_path, 'w') as f:
        for r in results:
            # Remove sanity_metrics for cleaner output (too verbose)
            r_clean = {k: v for k, v in r.items() if k != 'sanity_metrics'}
            f.write(json.dumps(r_clean, cls=NumpyEncoder) + '\n')
    print(f"  Results: {jsonl_path} ({len(results)} unique events)")

    summary = {
        'n_total': len(results),
        'n_ok': len(ok_results),
        'n_complex': status_counts['complex'],
        'n_failed': status_counts['failed'] + status_counts.get('fetch_failed', 0),
        'pass_rate': pass_rate,
        'classification_counts': dict(class_counts) if ok_results else {},
        'classes_analyzed': classes,
        'min_snr': min_snr,
        'window_ms': window_ms,
        'peak_ratio_threshold': peak_ratio_threshold,
        'bandpass': bandpass
    }

    if ok_results and classified > 0:
        summary['iof_fraction'] = class_counts['iof'] / len(ok_results)
        summary['iof_of_classified'] = class_counts['iof'] / classified

    summary_path = output_dir / f'ligo_envelope_{class_str}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")

    print(f"\n{'='*60}")
    print("Analysis complete!")

    return results, summary


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LIGO envelope-based IOF classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--n_events', type=int, default=500,
                       help='Number of events (0 = all)')
    parser.add_argument('--min_snr', type=float, default=50,
                       help='Minimum SNR threshold')
    parser.add_argument('--classes', nargs='+', default=['Blip'],
                       help='Gravity Spy classes to analyze')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Path to Gravity Spy CSV')

    parser.add_argument('--window_ms', type=float, default=100.0,
                       help='Analysis window in milliseconds')
    parser.add_argument('--fix_baseline', action='store_true', default=True)
    parser.add_argument('--no_fix_baseline', action='store_false', dest='fix_baseline')

    parser.add_argument('--peak_ratio_threshold', type=float, default=0.7,
                       help='Reject if p2/p1 > this value')
    parser.add_argument('--no_bandpass', action='store_false', dest='bandpass',
                       help='Disable bandpass filtering before Hilbert')

    parser.add_argument('--output_dir', type=str, default='output/ligo_envelope',
                       help='Output directory')
    parser.add_argument('--cache_dir', type=str, default='strain_cache',
                       help='Strain data cache directory')
    parser.add_argument('--bulk_data_dir', type=str, default=None,
                       help='Directory with pre-downloaded bulk HDF5 files (from ligo_bulk_download.py)')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output_dir
    cache_dir = script_dir / args.cache_dir
    bulk_dir = Path(args.bulk_data_dir) if args.bulk_data_dir else None

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        csv_path = script_dir / "data" / "H1_O3a_glitches.csv"

    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)

    if not GWPY_AVAILABLE:
        print("Error: gwpy required. Install with: pip install gwpy")
        sys.exit(1)

    run_analysis(
        csv_path=csv_path,
        output_dir=output_dir,
        cache_dir=cache_dir,
        classes=args.classes,
        n_events=args.n_events,
        min_snr=args.min_snr,
        window_ms=args.window_ms,
        fix_baseline=args.fix_baseline,
        peak_ratio_threshold=args.peak_ratio_threshold,
        bandpass=args.bandpass,
        seed=args.seed,
        bulk_dir=bulk_dir
    )


if __name__ == '__main__':
    main()
