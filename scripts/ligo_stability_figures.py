#!/usr/bin/env python3
"""
LIGO 3-Window Stability Figures
================================

Generates LIGO stability analysis figures for the manuscript.

Implements the full manuscript methodology:
1. Load cached strain data for OK events (from ligo_glitch_analysis.py output)
2. Run model tournament at W ∈ {60, 100, 150} ms windows
3. Compute curvature consistently (0-20 ms post-peak)
4. Determine stability: stable_IOF (3/3), stable_STD (3/3), flip (mixed), failed
5. Generate curvature plot for stable-core subset
6. Bootstrap regression on stable-core events

Prerequisites:
    - Run ligo_glitch_analysis.py first to generate cached strain data and results
    - Strain cache in scripts/strain_cache/
    - Results in scripts/output/ligo_envelope/

Usage:
    python ligo_stability_figures.py

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.stats import mannwhitneyu
from scipy.optimize import minimize

# Import canonical pipeline invariants from shared module
# These are the ONLY sources for these functions - no local copies allowed
from ligo_pipeline_common import (
    bandpass_filter,
    compute_hilbert_envelope,
    compute_times_ms,
    center_times_on_peak,
    extract_fit_window_indices,
    find_constrained_peak,
    baseline_from_postpeak_window,
    compute_recovery_z,
    extract_fit_window,
    compute_curvature_index as common_compute_curvature_index,
    load_cached_strain as common_load_cached_strain,
    PEAK_SEARCH_WINDOW_MS,
    CURVATURE_WINDOW_MS as COMMON_CURVATURE_WINDOW_MS,
    ANALYSIS_WINDOWS_MS,
    DEFAULT_BASELINE_WINDOW_MS
)

# Import canonical fitting functions from iof_metrics
# These are the single source of truth for LIGO classification
from iof_metrics import (
    FitResult,
    baseline_tail_median,
    fit_envelope_with_baseline,
    check_fit_sanity as canonical_check_fit_sanity,
    get_model_geometry,
)

# Provenance tracking
try:
    from provenance import add_provenance
except ImportError:
    # Fallback if provenance module not available
    def add_provenance(output, *args, **kwargs):
        return output
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Fixed seed for reproducible bootstrap confidence intervals
np.random.seed(42)

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "ligo_envelope"
FIGURES_DIR = SCRIPT_DIR.parent / "figures" / "ligo"
RESULTS_FILE = OUTPUT_DIR / "ligo_envelope_Extremely_Loud_results.jsonl"
CACHE_DIR = SCRIPT_DIR / "strain_cache"

# Analysis parameters (frozen pipeline)
WINDOWS_MS = [60, 100, 150]
CURVATURE_WINDOW_MS = 20.0
STABILITY_THRESHOLD = 1.0  # 3/3 = 100% agreement required

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Signal processing - IMPORTED FROM ligo_pipeline_common.py
# =============================================================================
# bandpass_filter, compute_hilbert_envelope, baseline_from_postpeak_window
# are now imported from the canonical source. No local copies.


# =============================================================================
# Model fitting - IMPORTED FROM iof_metrics.py
# =============================================================================
# fit_envelope_with_baseline, check_fit_sanity, get_model_geometry
# are now imported from the canonical source. No local copies.


def classify_event(envelope, times_ms, window_ms, peak_idx=None):
    """
    Classify a single event at a given window size using CANONICAL functions.

    Returns (classification, winning_model, delta_aicc, geometry, t_inf, tau).

    t_inf and tau are extracted from the winning model:
    - Fast models (exponential, power_law): t_inf = 0, tau from params
    - Sigmoid: t_inf = t0, tau = 1/k
    - Delayed exponential: t_inf = delay, tau from params

    Args:
        envelope: Hilbert envelope array
        times_ms: Time array in ms (GPS-relative)
        window_ms: Analysis window in ms
        peak_idx: Pre-computed peak index. If None, uses find_constrained_peak.
                  IMPORTANT: Pass this to ensure same peak as main pipeline.
    """
    if peak_idx is None:
        peak_idx = find_constrained_peak(envelope, times_ms)

    peak_val = envelope[peak_idx]

    # Center times on peak (canonical approach)
    times_centered = center_times_on_peak(times_ms, peak_idx)

    # Extract window by INDEX using searchsorted (exact match to mask rule)
    env_fit_idx = extract_fit_window_indices(times_centered, peak_idx, window_ms)

    if len(env_fit_idx) < 50:
        return 'failed', None, None, None, None, None

    t_fit = times_centered[env_fit_idx]
    env_fit = envelope[env_fit_idx]

    # INVARIANT checks
    assert env_fit_idx[0] == peak_idx, f"env_fit does not start at peak"
    assert abs(t_fit[0]) < 1e-9, f"t_fit[0] must be 0 after centering"

    # Canonical baseline (n//5 tail rule)
    baseline = baseline_tail_median(env_fit)

    if peak_val <= baseline:
        return 'failed', None, None, None, None, None

    # Fit models using canonical fitter (explicit baseline, no recompute)
    fits = fit_envelope_with_baseline(t_fit, env_fit, baseline)

    if not fits:
        return 'failed', None, None, None, None, None

    # Filter to sane fits only using canonical sanity checker
    valid_fits = {}
    for name, fit in fits.items():
        is_sane, _ = canonical_check_fit_sanity(fit)
        if is_sane:
            valid_fits[name] = fit

    if not valid_fits:
        return 'failed', None, None, None, None, None

    # Find winner by AICc
    sorted_fits = sorted(valid_fits.items(), key=lambda x: x[1].aicc)
    winner_name, winner_fit = sorted_fits[0]

    # Delta AICc to runner-up
    if len(sorted_fits) > 1:
        delta_aicc = sorted_fits[1][1].aicc - winner_fit.aicc
    else:
        delta_aicc = float('inf')

    # Get geometry using canonical function
    geometry = get_model_geometry(winner_name)

    # Extract t_inf and tau from winner model
    params = winner_fit.params
    if winner_name in ['exponential', 'exponential_fixed', 'power_law']:
        # Fast models: t_inf = 0, tau from params
        t_inf = 0.0
        tau = params.get('tau', 20.0)
    elif winner_name == 'sigmoid':
        # Sigmoid: t_inf = t0, tau = 1/k
        t_inf = params.get('t0', 0.0)
        k = params.get('k', 0.1)
        tau = 1.0 / k if k > 0 else 20.0
    elif winner_name == 'delayed':
        # Delayed exponential: t_inf = delay, tau from params
        t_inf = params.get('delay', 0.0)
        tau = params.get('tau', 20.0)
    else:
        t_inf = 0.0
        tau = 20.0

    # Check if model selection is confident (ΔAICc >= 2)
    if delta_aicc < 2.0 and len(sorted_fits) >= 2:
        # Models are comparable - check if they agree on geometry
        g0 = get_model_geometry(sorted_fits[0][0])
        g1 = get_model_geometry(sorted_fits[1][0])
        if g0 != g1:
            # Models disagree on geometry - uncertain
            return 'uncertain', winner_name, delta_aicc, geometry, t_inf, tau

    # Classify by geometry
    if geometry == 'delayed':
        classification = 'iof'
    else:
        classification = 'standard'

    return classification, winner_name, delta_aicc, geometry, t_inf, tau


def compute_curvature_index(envelope, times_ms, window_ms=CURVATURE_WINDOW_MS, peak_idx=None,
                            baseline_window_ms=150.0):
    """
    Compute early-time curvature index b using canonical function.
    Wrapper that maintains API compatibility while delegating to common module.
    """
    if peak_idx is None:
        peak_idx = np.argmax(envelope)
    return common_compute_curvature_index(
        envelope, times_ms, peak_idx,
        curvature_window_ms=window_ms,
        baseline_window_ms=baseline_window_ms
    )


# =============================================================================
# Data loading
# =============================================================================

def load_results():
    """Load per-event results from JSONL, deduplicated by GPS time.

    Note: Gravity Spy catalog contains multiple rows per physical glitch
    (different annotations). We deduplicate by gps_time to get unique events.
    Cached strain is keyed by GPS time, so duplicates affect counts but not
    strain content.
    """
    events_by_gps = {}
    with open(RESULTS_FILE) as f:
        for line in f:
            event = json.loads(line)
            gps = event.get('gps_time')
            # Keep first occurrence (all duplicates have same strain/analysis)
            if gps not in events_by_gps:
                events_by_gps[gps] = event
    return list(events_by_gps.values())


def load_cached_strain(gps_time):
    """Load cached strain data using canonical loader with legacy fallback."""
    return common_load_cached_strain(gps_time, CACHE_DIR)


# =============================================================================
# Statistics
# =============================================================================

def cliffs_delta(x, y):
    """Cliff's delta effect size."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    dominance = sum(1 if xi > yj else (-1 if xi < yj else 0)
                    for xi in x for yj in y)
    return dominance / (n1 * n2)


def compute_auc(x, y):
    """AUC from Mann-Whitney U."""
    if len(x) == 0 or len(y) == 0:
        return 0.5
    u_stat, _ = mannwhitneyu(x, y, alternative='two-sided')
    return u_stat / (len(x) * len(y))


def bootstrap_auc(x, y, n_boot=1000):
    """Bootstrap CI for AUC."""
    x, y = np.array(x), np.array(y)
    aucs = [compute_auc(
        np.random.choice(x, len(x), replace=True),
        np.random.choice(y, len(y), replace=True)
    ) for _ in range(n_boot)]
    return np.percentile(aucs, [2.5, 97.5])


def logistic_regression_bootstrap(curvatures, snrs, labels, n_boot=1000):
    """Bootstrap logistic regression: IOF ~ b + SNR, and b-only model."""
    from scipy.special import expit

    b = np.array(curvatures)
    snr = np.array(snrs)
    y = np.array(labels)

    b_z = (b - np.mean(b)) / (np.std(b) + 1e-10)
    snr_z = (snr - np.mean(snr)) / (np.std(snr) + 1e-10)

    def neg_ll(params, X, y):
        prob = expit(X @ params)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))

    def fit(X, y):
        n, k = X.shape
        res = minimize(neg_ll, np.zeros(k), args=(X, y), method='BFGS')
        bic = 2 * res.fun + k * np.log(n)
        return res.x, bic

    # Design matrices
    X_null = np.ones((len(y), 1))  # Intercept only
    X_b = np.column_stack([np.ones(len(y)), b_z])  # b-only
    X_snr = np.column_stack([np.ones(len(y)), snr_z])  # SNR-only
    X_full = np.column_stack([np.ones(len(y)), b_z, snr_z])  # Full model

    # Fit models
    _, bic_null = fit(X_null, y)
    params_b, bic_b = fit(X_b, y)
    params_snr, bic_snr = fit(X_snr, y)
    params_full, bic_full = fit(X_full, y)

    beta_b = params_full[1]
    beta_snr = params_full[2]
    beta_b_only = params_b[1]
    beta_snr_only = params_snr[1]  # SNR-only model coefficient
    delta_bic = bic_full - bic_snr  # Full vs SNR-only
    delta_bic_b_only = bic_b - bic_null  # b-only vs null
    delta_bic_snr_only = bic_snr - bic_null  # SNR-only vs null

    # Bootstrap
    beta_b_boot, delta_bic_boot = [], []
    beta_b_only_boot, delta_bic_b_only_boot = [], []
    beta_snr_only_boot = []
    n = len(y)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            _, bic_n = fit(X_null[idx], y[idx])
            p_b, bic_bo = fit(X_b[idx], y[idx])
            p_s, bic_s = fit(X_snr[idx], y[idx])
            p, bic_f = fit(X_full[idx], y[idx])
            beta_b_boot.append(p[1])
            delta_bic_boot.append(bic_f - bic_s)
            beta_b_only_boot.append(p_b[1])
            delta_bic_b_only_boot.append(bic_bo - bic_n)
            beta_snr_only_boot.append(p_s[1])
        except:
            pass

    return {
        'beta_b': beta_b,
        'beta_b_ci': np.percentile(beta_b_boot, [2.5, 97.5]) if beta_b_boot else [0, 0],
        'beta_snr': beta_snr,
        'delta_bic': delta_bic,
        'delta_bic_ci': np.percentile(delta_bic_boot, [2.5, 97.5]) if delta_bic_boot else [0, 0],
        # b-only model results
        'beta_b_only': beta_b_only,
        'beta_b_only_ci': np.percentile(beta_b_only_boot, [2.5, 97.5]) if beta_b_only_boot else [0, 0],
        'delta_bic_b_only': delta_bic_b_only,
        'delta_bic_b_only_ci': np.percentile(delta_bic_b_only_boot, [2.5, 97.5]) if delta_bic_b_only_boot else [0, 0],
        # SNR-only model results
        'beta_snr_only': beta_snr_only,
        'beta_snr_only_ci': np.percentile(beta_snr_only_boot, [2.5, 97.5]) if beta_snr_only_boot else [0, 0],
        'delta_bic_snr_only': delta_bic_snr_only,
    }


# =============================================================================
# Plotting
# =============================================================================

def generate_curvature_plot(stable_iof, stable_std, flip_events, auc, auc_ci, cliff_d):
    """Generate curvature separation plot for stable-core events."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    b_iof = [e['curvature_b'] * 1000 for e in stable_iof]
    b_std = [e['curvature_b'] * 1000 for e in stable_std]
    b_flip = [e['curvature_b'] * 1000 for e in flip_events]

    # Boxplot
    ax1 = axes[0]
    data = [b_std, b_flip, b_iof]
    labels = [f'Stable Fast\n(n={len(b_std)})', f'Flip\n(n={len(b_flip)})', f'Stable Delayed\n(n={len(b_iof)})']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    bp = ax1.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Curvature index $b$ (×10⁻³)', fontsize=12)
    ax1.set_title('Curvature by Stability Category', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Density
    ax2 = axes[1]
    from scipy.stats import gaussian_kde

    for arr, color, label in [(b_std, '#2ecc71', 'Stable Fast'),
                               (b_flip, '#f39c12', 'Flip'),
                               (b_iof, '#e74c3c', 'Stable Delayed')]:
        if len(arr) > 5:
            try:
                kde = gaussian_kde(arr)
                x = np.linspace(min(arr) - 1, max(arr) + 1, 200)
                ax2.fill_between(x, kde(x), alpha=0.3, color=color, label=label)
                ax2.plot(x, kde(x), color=color, linewidth=2)
            except:
                pass

    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Curvature index $b$ (×10⁻³)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'Curvature Distributions (Stable Core)\nAUC = {auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}], Cliff\'s δ = {cliff_d:.3f}', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    for path in [FIGURES_DIR / "curvature_index_plot.png", OUTPUT_DIR / "curvature_index_plot.png"]:
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("LIGO 3-Window Stability Analysis & Plot Regeneration")
    print("=" * 70)

    # Load results to get OK events
    events = load_results()
    ok_events = [e for e in events if e.get('status') == 'ok']
    print(f"Loaded {len(ok_events)} OK events from results file")

    # Process each event with 3-window stability
    print(f"\nRunning 3-window stability analysis (W = {WINDOWS_MS} ms)...")

    processed = []
    cache_hits = 0

    for i, e in enumerate(ok_events):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(ok_events)}...")

        gps = e['gps_time']
        strain_data = load_cached_strain(gps)

        if strain_data is None:
            continue

        cache_hits += 1

        # Compute envelope
        strain = strain_data['values']
        fs = strain_data['sample_rate']
        times = strain_data['times']

        # CRITICAL: Convert times relative to GPS time (not segment start)
        # and constrain to ±500ms to match ligo_glitch_analysis.py peak localization
        times_ms = (times - gps) * 1000

        envelope = compute_hilbert_envelope(strain, fs)

        # Constrain to ±500ms around GPS time to ensure same peak as main pipeline
        search_mask = np.abs(times_ms) < 500
        if not np.any(search_mask):
            continue

        # Find peak within constrained region
        search_envelope = envelope.copy()
        search_envelope[~search_mask] = 0  # Zero out regions outside ±500ms
        peak_idx_global = np.argmax(search_envelope)
        peak_time_ms = times_ms[peak_idx_global]

        # Shift times so peak is at t=0 (consistent with downstream analysis)
        times_ms = times_ms - peak_time_ms

        # Classify at each window (pass peak_idx to ensure same peak as main pipeline)
        window_results = {}
        t_infs = []
        taus = []
        for w in WINDOWS_MS:
            classification, model, delta, geom, t_inf, tau = classify_event(
                envelope, times_ms, w, peak_idx=peak_idx_global
            )
            window_results[w] = {
                'classification': classification,
                'model': model,
                'delta_aicc': delta,
                'geometry': geom,
                't_inf': t_inf,
                'tau': tau,
            }
            if t_inf is not None and tau is not None:
                t_infs.append(t_inf)
                taus.append(tau)

        # Compute curvature (fixed 20ms window, independent of model)
        curvature_b = compute_curvature_index(envelope, times_ms, peak_idx=peak_idx_global)

        # Determine stability
        classifications = [window_results[w]['classification'] for w in WINDOWS_MS]

        if 'failed' in classifications:
            stability = 'failed'
        elif 'uncertain' in classifications:
            # Any uncertain window means flip (mixed/uncertain)
            stability = 'flip'
        elif all(c == 'iof' for c in classifications):
            stability = 'stable_iof'
        elif all(c == 'standard' for c in classifications):
            stability = 'stable_std'
        else:
            stability = 'flip'

        if curvature_b is not None:
            # Compute D = t_inf / tau (average across windows)
            if t_infs and taus:
                avg_t_inf = np.mean(t_infs)
                avg_tau = np.mean(taus)
                D = avg_t_inf / avg_tau if avg_tau > 0 else 0.0
            else:
                D = 0.0
                avg_t_inf = 0.0
                avg_tau = 20.0

            processed.append({
                'gps_time': gps,
                'snr': e.get('snr', 0),
                'stability': stability,
                'curvature_b': curvature_b,
                'D': D,
                't_inf': avg_t_inf,
                'tau': avg_tau,
                'window_results': window_results,
            })

    print(f"\nCache hits: {cache_hits}")
    print(f"Events with valid curvature: {len(processed)}")

    # Separate by stability
    stable_iof = [e for e in processed if e['stability'] == 'stable_iof']
    stable_std = [e for e in processed if e['stability'] == 'stable_std']
    flip = [e for e in processed if e['stability'] == 'flip']
    failed = [e for e in processed if e['stability'] == 'failed']

    print(f"\nStability classification:")
    print(f"  Stable Delayed (IOF): {len(stable_iof)} ({100*len(stable_iof)/len(processed):.1f}%)")
    print(f"  Stable Fast (STD):    {len(stable_std)} ({100*len(stable_std)/len(processed):.1f}%)")
    print(f"  Flip:                 {len(flip)} ({100*len(flip)/len(processed):.1f}%)")
    print(f"  Failed:               {len(failed)} ({100*len(failed)/len(processed):.1f}%)")

    if len(stable_iof) == 0 or len(stable_std) == 0:
        print("ERROR: Not enough stable events for analysis")
        return

    # Curvature analysis on stable core
    b_iof = [e['curvature_b'] for e in stable_iof]
    b_std = [e['curvature_b'] for e in stable_std]
    b_flip = [e['curvature_b'] for e in flip]

    stat, p_value = mannwhitneyu(b_iof, b_std, alternative='two-sided')
    cliff_d = cliffs_delta(b_iof, b_std)
    auc = compute_auc(b_iof, b_std)
    auc_ci = bootstrap_auc(b_iof, b_std)

    # Compute medians and IQRs for each population (scaled to 10^-3)
    b_iof_scaled = [b * 1000 for b in b_iof]
    b_std_scaled = [b * 1000 for b in b_std]
    b_flip_scaled = [b * 1000 for b in b_flip]

    curvature_stats = {
        'delayed': {
            'median': np.median(b_iof_scaled),
            'iqr_lo': np.percentile(b_iof_scaled, 25),
            'iqr_hi': np.percentile(b_iof_scaled, 75),
        },
        'fast': {
            'median': np.median(b_std_scaled),
            'iqr_lo': np.percentile(b_std_scaled, 25),
            'iqr_hi': np.percentile(b_std_scaled, 75),
        },
        'flip': {
            'median': np.median(b_flip_scaled) if b_flip_scaled else 0,
            'iqr_lo': np.percentile(b_flip_scaled, 25) if b_flip_scaled else 0,
            'iqr_hi': np.percentile(b_flip_scaled, 75) if b_flip_scaled else 0,
        },
    }

    print(f"\nCurvature separation (stable core only):")
    print(f"  Stable Delayed median b: {curvature_stats['delayed']['median']:.2f} × 10⁻³ (IQR [{curvature_stats['delayed']['iqr_lo']:.1f}, {curvature_stats['delayed']['iqr_hi']:.1f}])")
    print(f"  Stable Fast median b: {curvature_stats['fast']['median']:.2f} × 10⁻³ (IQR [{curvature_stats['fast']['iqr_lo']:.1f}, {curvature_stats['fast']['iqr_hi']:.1f}])")
    print(f"  Mann-Whitney p = {p_value:.2e}")
    print(f"  Cliff's δ = {cliff_d:.3f}")
    print(f"  AUC = {auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")

    # Generate plot
    generate_curvature_plot(stable_iof, stable_std, flip, auc, auc_ci, cliff_d)

    # Bootstrap regression on stable core
    print("\nBootstrap regression (stable core)...")
    curvatures = [e['curvature_b'] for e in stable_iof + stable_std]
    snrs = [e['snr'] for e in stable_iof + stable_std]
    labels = [1] * len(stable_iof) + [0] * len(stable_std)

    reg = logistic_regression_bootstrap(curvatures, snrs, labels, n_boot=1000)

    print(f"  Full model (IOF ~ b + SNR):")
    print(f"    β_b = {reg['beta_b']:.2f} [{reg['beta_b_ci'][0]:.2f}, {reg['beta_b_ci'][1]:.2f}]")
    print(f"    β_SNR = {reg['beta_snr']:.2f}")
    print(f"    ΔBIC (vs SNR-only) = {reg['delta_bic']:.1f} [{reg['delta_bic_ci'][0]:.1f}, {reg['delta_bic_ci'][1]:.1f}]")
    print(f"  b-only model (IOF ~ b):")
    print(f"    β_b = {reg['beta_b_only']:.2f} [{reg['beta_b_only_ci'][0]:.2f}, {reg['beta_b_only_ci'][1]:.2f}]")
    print(f"    ΔBIC (vs null) = {reg['delta_bic_b_only']:.1f} [{reg['delta_bic_b_only_ci'][0]:.1f}, {reg['delta_bic_b_only_ci'][1]:.1f}]")

    # Save results
    output = {
        'methodology': '3-window stability (60/100/150 ms)',
        'n_ok': len(processed),
        'n_stable_iof': len(stable_iof),
        'n_stable_std': len(stable_std),
        'n_flip': len(flip),
        'n_failed': len(failed),
        'stable_iof_pct': 100 * len(stable_iof) / len(processed),
        'stable_std_pct': 100 * len(stable_std) / len(processed),
        'flip_pct': 100 * len(flip) / len(processed),
        'auc': auc,
        'auc_ci_lower': auc_ci[0],
        'auc_ci_upper': auc_ci[1],
        'cliffs_delta': cliff_d,
        # Curvature stats (scaled to 10^-3)
        'curvature_delayed_median': curvature_stats['delayed']['median'],
        'curvature_delayed_iqr_lo': curvature_stats['delayed']['iqr_lo'],
        'curvature_delayed_iqr_hi': curvature_stats['delayed']['iqr_hi'],
        'curvature_fast_median': curvature_stats['fast']['median'],
        'curvature_fast_iqr_lo': curvature_stats['fast']['iqr_lo'],
        'curvature_fast_iqr_hi': curvature_stats['fast']['iqr_hi'],
        # Mann-Whitney test for curvature separation
        'curvature_mann_whitney_p': float(p_value),
        'curvature_mann_whitney_p_exp': int(-np.floor(np.log10(max(p_value, 1e-300)))),  # exponent for p < 10^-X
        # Full model (IOF ~ b + SNR)
        'beta_b': reg['beta_b'],
        'beta_b_ci_lower': reg['beta_b_ci'][0],
        'beta_b_ci_upper': reg['beta_b_ci'][1],
        'beta_snr': reg['beta_snr'],
        'delta_bic': reg['delta_bic'],
        'delta_bic_ci_lower': reg['delta_bic_ci'][0],
        'delta_bic_ci_upper': reg['delta_bic_ci'][1],
        # b-only model (IOF ~ b, without SNR control)
        'beta_b_only': reg['beta_b_only'],
        'beta_b_only_ci_lower': reg['beta_b_only_ci'][0],
        'beta_b_only_ci_upper': reg['beta_b_only_ci'][1],
        'delta_bic_b_only': reg['delta_bic_b_only'],
        'delta_bic_b_only_ci_lower': reg['delta_bic_b_only_ci'][0],
        'delta_bic_b_only_ci_upper': reg['delta_bic_b_only_ci'][1],
        # SNR-only model (IOF ~ SNR, without curvature)
        'beta_snr_only': reg['beta_snr_only'],
        'beta_snr_only_ci_lower': reg['beta_snr_only_ci'][0],
        'beta_snr_only_ci_upper': reg['beta_snr_only_ci'][1],
        'delta_bic_snr_only': reg['delta_bic_snr_only'],
    }

    # Add provenance metadata
    add_provenance(
        output, __file__,
        params={
            'windows_ms': [60, 100, 150],
            'stability_threshold': 3,  # 3/3 windows must agree
            'aicc_threshold': 2,
            'curvature_window_ms': 20,
            'n_bootstrap': 1000,
        },
        data_source='ligo_gwosc',
    )

    with open(OUTPUT_DIR / "stability_analysis.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'stability_analysis.json'}")

    # Also update bootstrap_beta_b.json for consistency
    with open(OUTPUT_DIR / "bootstrap_beta_b.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'bootstrap_beta_b.json'}")

    # Export per-event data for phase diagram and other analyses
    # (This is the canonical source for 3-window stability classification)
    events_file = OUTPUT_DIR / "stability_events.jsonl"
    with open(events_file, 'w') as f:
        for event in processed:
            # Create a simplified record (exclude window_results to keep file small)
            record = {
                'gps_time': event['gps_time'],
                'snr': event['snr'],
                'stability': event['stability'],
                'curvature_b': event['curvature_b'],
                'D': event['D'],
                't_inf': event['t_inf'],
                'tau': event['tau'],
            }
            f.write(json.dumps(record) + '\n')
    print(f"Saved: {events_file} ({len(processed)} events)")

    # Export detailed per-window data for threshold sensitivity analysis
    # This allows post-processing threshold sweeps without re-fitting
    detailed_file = OUTPUT_DIR / "stability_events_detailed.jsonl"
    with open(detailed_file, 'w') as f:
        for event in processed:
            # Extract per-window delta_aicc and classification
            window_deltas = {}
            for w in WINDOWS_MS:
                wr = event['window_results'].get(w, {})
                window_deltas[str(w)] = {
                    'delta_aicc': wr.get('delta_aicc'),
                    'classification': wr.get('classification'),
                    'geometry': wr.get('geometry'),
                }
            record = {
                'gps_time': event['gps_time'],
                'snr': event['snr'],
                'curvature_b': event['curvature_b'],
                'window_results': window_deltas,
            }
            f.write(json.dumps(record) + '\n')
    print(f"Saved: {detailed_file} ({len(processed)} events with per-window data)")

    print("\n" + "=" * 70)
    print("Done! Plots regenerated from stable-core analysis.")
    print("=" * 70)


if __name__ == "__main__":
    main()
