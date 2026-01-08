#!/usr/bin/env python3
"""
Generate Appendix Figures for Forensic Signatures Manuscript.
==============================================================

Creates:
1. Dip test on curvature b for LIGO stable events (Appendix B)
2. Curated event examples (2 stable STD, 2 stable IOF, 2 flip) (Appendix C)
3. Summary of stability distribution

Prerequisites:
    - Run ligo_glitch_analysis.py first to generate cached strain data and results
    - Strain cache in scripts/strain_cache/
    - Results in scripts/output/ligo_envelope/

Dependencies:
    pip install numpy scipy matplotlib diptest

Usage:
    python ligo_appendix_figures.py

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Import canonical pipeline invariants from shared module
# These are the ONLY sources for these functions - no local copies allowed
from ligo_pipeline_common import (
    bandpass_filter,
    compute_hilbert_envelope,
    baseline_from_postpeak_window,
    find_constrained_peak,
    compute_times_ms,
    center_times_on_peak,
    extract_fit_window_indices,
    compute_curvature_index as common_compute_curvature_index,
    load_cached_strain as common_load_cached_strain,
    ANALYSIS_WINDOWS_MS,
    CURVATURE_WINDOW_MS as COMMON_CURVATURE_WINDOW_MS
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

# Try to import diptest
try:
    from diptest import diptest
    HAS_DIPTEST = True
except ImportError:
    HAS_DIPTEST = False
    print("Warning: 'diptest' not installed. Install with: pip install diptest")

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "ligo_envelope"
FIGURES_DIR = SCRIPT_DIR.parent / "figures" / "ligo"
APPENDIX_DIR = SCRIPT_DIR.parent / "figures" / "appendix"
RESULTS_FILE = OUTPUT_DIR / "ligo_envelope_Extremely_Loud_results.jsonl"
CACHE_DIR = SCRIPT_DIR / "strain_cache"

APPENDIX_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters (must match ligo_stability_figures.py)
WINDOWS_MS = [60, 100, 150]
CURVATURE_WINDOW_MS = 20.0


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


def classify_event_at_window(envelope, times_ms, window_ms, peak_idx=None):
    """
    Classify at a single window using CANONICAL functions.

    Args:
        envelope: Hilbert envelope array
        times_ms: Time array in ms (GPS-relative)
        window_ms: Analysis window in ms
        peak_idx: Pre-computed peak index. If None, uses argmax.
                  IMPORTANT: Pass this to ensure same peak as main pipeline.

    Returns:
        (classification, winner_name, winner_data)
    """
    if peak_idx is None:
        peak_idx = np.argmax(envelope)

    peak_val = envelope[peak_idx]

    # Center times on peak (canonical approach)
    times_centered = center_times_on_peak(times_ms, peak_idx)

    # Extract window by INDEX using searchsorted (exact match to mask rule)
    env_fit_idx = extract_fit_window_indices(times_centered, peak_idx, window_ms)

    if len(env_fit_idx) < 50:
        return 'failed', None, None

    t_fit = times_centered[env_fit_idx]
    env_fit = envelope[env_fit_idx]

    # INVARIANT checks
    assert env_fit_idx[0] == peak_idx, f"env_fit does not start at peak"
    assert abs(t_fit[0]) < 1e-9, f"t_fit[0] must be 0 after centering"

    # Canonical baseline (n//5 tail rule)
    baseline = baseline_tail_median(env_fit)

    if peak_val <= baseline:
        return 'failed', None, None

    # Fit models using canonical fitter (explicit baseline, no recompute)
    fits = fit_envelope_with_baseline(t_fit, env_fit, baseline)

    if not fits:
        return 'failed', None, None

    # Filter to sane fits only using canonical sanity checker
    valid_fits = {}
    for name, fit in fits.items():
        is_sane, _ = canonical_check_fit_sanity(fit)
        if is_sane:
            valid_fits[name] = fit

    if not valid_fits:
        return 'failed', None, None

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

    # Check if model selection is confident (ΔAICc >= 2)
    if delta_aicc < 2.0 and len(sorted_fits) >= 2:
        g0 = get_model_geometry(sorted_fits[0][0])
        g1 = get_model_geometry(sorted_fits[1][0])
        if g0 != g1:
            # Models disagree on geometry - uncertain
            # Return winner_fit converted to dict for compatibility
            return 'uncertain', winner_name, {
                'aicc': winner_fit.aicc,
                'params': winner_fit.params,
                'geometry': geometry,
                'r2': winner_fit.r2
            }

    # Classify by geometry
    classification = 'iof' if geometry == 'delayed' else 'standard'

    # Return winner_fit converted to dict for compatibility with existing code
    return classification, winner_name, {
        'aicc': winner_fit.aicc,
        'params': winner_fit.params,
        'geometry': geometry,
        'r2': winner_fit.r2
    }


def compute_curvature_index(envelope, times_ms, peak_idx=None, baseline_window_ms=150.0):
    """Compute curvature b over first 20ms post-peak using canonical function."""
    if peak_idx is None:
        peak_idx = np.argmax(envelope)
    return common_compute_curvature_index(
        envelope, times_ms, peak_idx,
        curvature_window_ms=CURVATURE_WINDOW_MS,
        baseline_window_ms=baseline_window_ms
    )


def load_cached_strain(gps_time):
    """Load cached strain data using canonical loader with legacy fallback."""
    return common_load_cached_strain(gps_time, CACHE_DIR)


def load_results():
    """Load per-event results, deduplicated by GPS time."""
    events_by_gps = {}
    with open(RESULTS_FILE) as f:
        for line in f:
            event = json.loads(line)
            gps = event.get('gps_time')
            if gps not in events_by_gps:
                events_by_gps[gps] = event
    return list(events_by_gps.values())


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_all_events():
    """Re-run 3-window stability and collect curvature for all events."""
    print("Loading and analyzing events...")
    events = load_results()
    ok_events = [e for e in events if e.get('status') == 'ok']
    print(f"Found {len(ok_events)} OK events")

    processed = []

    for i, e in enumerate(ok_events):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(ok_events)}...")

        gps = e['gps_time']
        strain_data = load_cached_strain(gps)
        if strain_data is None:
            continue

        strain = strain_data['values']
        fs = strain_data['sample_rate']
        times = strain_data['times']
        # CRITICAL: Convert times relative to GPS time (not segment start)
        # to match ligo_glitch_analysis.py peak localization
        times_ms = (times - gps) * 1000

        envelope = compute_hilbert_envelope(strain, fs)

        # Constrain peak search to ±500ms around GPS time to ensure
        # same peak as main pipeline (ligo_glitch_analysis.py)
        search_mask = np.abs(times_ms) < 500
        search_envelope = envelope.copy()
        search_envelope[~search_mask] = 0
        peak_idx = np.argmax(search_envelope)

        # Classify at each window
        window_results = {}
        for w in WINDOWS_MS:
            cls, model, data = classify_event_at_window(envelope, times_ms, w, peak_idx=peak_idx)
            window_results[w] = {'classification': cls, 'model': model, 'data': data}

        # Compute curvature
        curvature_b = compute_curvature_index(envelope, times_ms, peak_idx=peak_idx)

        # Determine stability
        classifications = [window_results[w]['classification'] for w in WINDOWS_MS]
        if 'failed' in classifications:
            stability = 'failed'
        elif 'uncertain' in classifications:
            stability = 'flip'
        elif all(c == 'iof' for c in classifications):
            stability = 'stable_iof'
        elif all(c == 'standard' for c in classifications):
            stability = 'stable_std'
        else:
            stability = 'flip'

        if curvature_b is not None:
            processed.append({
                'gps_time': gps,
                'snr': e.get('snr', 0),
                'stability': stability,
                'curvature_b': curvature_b,
                'window_results': window_results,
                'envelope': envelope,
                'times_ms': times_ms,
            })

    return processed


def generate_dip_test_unlabeled(processed):
    """
    Run dip test on ALL OK events (stable + flips) ignoring stability labels.

    This addresses the referee concern that bimodality might be induced by
    conditioning on stability labels. By running the test on the full pool
    of analyzable events, we show the structure exists in the raw data.

    Outputs structured JSON with winsorized and trimmed variants for robustness.
    """
    if not HAS_DIPTEST:
        print("Skipping unlabeled dip test (diptest not installed)")
        return

    from scipy.stats import mstats

    # All events with valid curvature (including flips, excluding failed)
    all_ok = [e for e in processed if e['stability'] != 'failed' and e['curvature_b'] is not None]

    # Deduplicate by GPS time (take first occurrence)
    seen_gps = set()
    unique_events = []
    for e in all_ok:
        gps = e['gps_time']
        if gps not in seen_gps:
            seen_gps.add(gps)
            unique_events.append(e)

    # Stable-only events (for comparison)
    stable_events = [e for e in unique_events if e['stability'] in ('stable_iof', 'stable_std')]

    b_all = np.array([e['curvature_b'] * 1000 for e in unique_events])
    b_stable = np.array([e['curvature_b'] * 1000 for e in stable_events])

    if len(b_all) < 20:
        print(f"  Too few events for unlabeled dip test: {len(b_all)}")
        return

    # Helper: run dip test with variants
    def run_dip_variants(data, label):
        results = {}

        # Original
        dip, p = diptest(data)
        results['original'] = {'dip': float(dip), 'p': float(p)}

        # Winsorized (1st-99th percentile)
        winsorized = mstats.winsorize(data, limits=[0.01, 0.01])
        dip_w, p_w = diptest(np.array(winsorized))
        results['winsorized_1_99'] = {'dip': float(dip_w), 'p': float(p_w)}

        # Trimmed (k=5 each tail)
        k = 5
        if len(data) > 2 * k:
            sorted_data = np.sort(data)
            trimmed = sorted_data[k:-k]
            dip_t, p_t = diptest(trimmed)
            results['trimmed_k5'] = {'dip': float(dip_t), 'p': float(p_t)}
        else:
            results['trimmed_k5'] = {'dip': None, 'p': None}

        return results

    # Run tests for both cohorts
    all_ok_results = run_dip_variants(b_all, "all_ok")
    stable_only_results = run_dip_variants(b_stable, "stable_only")

    # Count by category (for reporting)
    n_delayed = sum(1 for e in unique_events if e['stability'] == 'stable_iof')
    n_fast = sum(1 for e in unique_events if e['stability'] == 'stable_std')
    n_flip = sum(1 for e in unique_events if e['stability'] == 'flip')

    print(f"\nDip test on ALL OK events (ignoring labels):")
    print(f"  All OK: N={len(b_all)}, dip={all_ok_results['original']['dip']:.4f}, p={all_ok_results['original']['p']:.4f}")
    print(f"    Winsorized p={all_ok_results['winsorized_1_99']['p']:.4f}, Trimmed p={all_ok_results['trimmed_k5']['p']:.4f}")
    print(f"  Stable-only: N={len(b_stable)}, dip={stable_only_results['original']['dip']:.4f}, p={stable_only_results['original']['p']:.4f}")
    print(f"    Trimmed p={stable_only_results['trimmed_k5']['p']:.4f}")
    print(f"  Breakdown: Delayed={n_delayed}, Fast={n_fast}, Flip={n_flip}")

    # Save results in structure expected by generate_macros.py
    results = {
        'all_ok': {
            'n': len(b_all),
            **all_ok_results
        },
        'stable_only': {
            'n': len(b_stable),
            **stable_only_results
        },
        'breakdown': {
            'n_stable_delayed': n_delayed,
            'n_stable_fast': n_fast,
            'n_flip': n_flip
        },
        'note': 'Dip test on all OK events (ignoring stability labels) with robustness variants'
    }

    outpath = OUTPUT_DIR / "dip_test_unlabeled.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outpath}")

    return results


def generate_dip_test_figure(processed):
    """Generate dip test figure for curvature b and save results to JSON."""
    if not HAS_DIPTEST:
        print("Skipping dip test (diptest not installed)")
        return

    stable_iof = [e for e in processed if e['stability'] == 'stable_iof']
    stable_std = [e for e in processed if e['stability'] == 'stable_std']

    b_iof = np.array([e['curvature_b'] * 1000 for e in stable_iof])
    b_std = np.array([e['curvature_b'] * 1000 for e in stable_std])
    b_all = np.concatenate([b_iof, b_std])

    # Run dip tests
    dip_all, p_all = diptest(b_all)
    dip_iof, p_iof = diptest(b_iof) if len(b_iof) > 10 else (0, 1)
    dip_std, p_std = diptest(b_std) if len(b_std) > 10 else (0, 1)

    print(f"\nDip test results:")
    print(f"  All stable: dip={dip_all:.4f}, p={p_all:.4f}")
    print(f"  Stable IOF: dip={dip_iof:.4f}, p={p_iof:.4f}")
    print(f"  Stable STD: dip={dip_std:.4f}, p={p_std:.4f}")

    # Save results to JSON for macro generation
    import json
    dip_results = {
        'dip_all': float(dip_all),
        'p_all': float(p_all),
        'dip_delayed': float(dip_iof),
        'p_delayed': float(p_iof),
        'dip_fast': float(dip_std),
        'p_fast': float(p_std),
        'n_all': len(b_all),
        'n_delayed': len(b_iof),
        'n_fast': len(b_std),
    }
    dip_json_path = OUTPUT_DIR / "dip_test.json"
    with open(dip_json_path, 'w') as f:
        json.dump(dip_results, f, indent=2)
    print(f"  Saved dip test results to {dip_json_path}")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Combined distribution
    ax = axes[0]
    ax.hist(b_all, bins=30, density=True, alpha=0.7, color='gray', edgecolor='black')
    ax.set_xlabel('Curvature index $b$ (×10⁻³)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'All Stable Events (n={len(b_all)})\nDip={dip_all:.3f}, p={p_all:.3f}', fontsize=11)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # IOF distribution
    ax = axes[1]
    ax.hist(b_iof, bins=20, density=True, alpha=0.7, color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Curvature index $b$ (×10⁻³)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Stable Delayed (n={len(b_iof)})\nDip={dip_iof:.3f}, p={p_iof:.3f}', fontsize=11)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # STD distribution
    ax = axes[2]
    ax.hist(b_std, bins=20, density=True, alpha=0.7, color='#2ecc71', edgecolor='black')
    ax.set_xlabel('Curvature index $b$ (×10⁻³)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Stable Fast (n={len(b_std)})\nDip={dip_std:.3f}, p={p_std:.3f}', fontsize=11)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    outpath = APPENDIX_DIR / "dip_test_curvature.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def generate_event_examples(processed):
    """Generate curated event example figures."""
    stable_iof = [e for e in processed if e['stability'] == 'stable_iof']
    stable_std = [e for e in processed if e['stability'] == 'stable_std']
    flips = [e for e in processed if e['stability'] == 'flip']

    # Sort by curvature to get representative examples
    stable_iof.sort(key=lambda x: x['curvature_b'], reverse=True)  # Most positive first
    stable_std.sort(key=lambda x: x['curvature_b'])  # Most negative first
    flips.sort(key=lambda x: abs(x['curvature_b']))  # Near zero first

    # Select examples
    examples = []
    if len(stable_std) >= 2:
        examples.append(('Stable Fast #1', stable_std[0], '#2ecc71'))
        examples.append(('Stable Fast #2', stable_std[1], '#2ecc71'))
    if len(stable_iof) >= 2:
        examples.append(('Stable Delayed #1', stable_iof[0], '#e74c3c'))
        examples.append(('Stable Delayed #2', stable_iof[1], '#e74c3c'))
    if len(flips) >= 2:
        examples.append(('Flip #1', flips[0], '#f39c12'))
        examples.append(('Flip #2', flips[1], '#f39c12'))

    if not examples:
        print("Not enough events for examples")
        return

    # Create figure
    n_examples = len(examples)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (title, event, color) in enumerate(examples):
        ax = axes[i]
        times_ms = event['times_ms']
        envelope = event['envelope']

        # Find peak with same constraint as pipeline (±500ms around GPS time = t=0)
        search_mask = np.abs(times_ms) < 500
        search_envelope = envelope.copy()
        search_envelope[~search_mask] = 0
        peak_idx = np.argmax(search_envelope)
        peak_time = times_ms[peak_idx]

        # Plot 150ms window
        mask = (times_ms >= peak_time - 20) & (times_ms <= peak_time + 150)
        t_plot = times_ms[mask] - peak_time
        env_plot = envelope[mask]

        # Normalize with baseline from analysis window (matches pipeline)
        peak_val = envelope[peak_idx]
        baseline = baseline_from_postpeak_window(envelope, times_ms, peak_idx, 150.0)
        env_norm = (env_plot - baseline) / (peak_val - baseline)

        ax.plot(t_plot, env_norm, color=color, linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

        # Annotate
        b = event['curvature_b'] * 1000
        snr = event['snr']
        ax.set_title(f"{title}\n$b$={b:.2f}×10⁻³, SNR={snr:.0f}", fontsize=10)
        ax.set_xlabel('Time from peak (ms)', fontsize=9)
        ax.set_ylabel('Normalized envelope', fontsize=9)
        ax.set_xlim(-20, 150)
        ax.set_ylim(-0.2, 1.1)
        ax.grid(alpha=0.3)

        # Add window classification text
        w_text = []
        for w in WINDOWS_MS:
            wr = event['window_results'].get(w, {})
            cls = wr.get('classification', '?')
            w_text.append(f"{w}ms:{cls[:3]}")
        ax.text(0.98, 0.02, ' | '.join(w_text), transform=ax.transAxes,
                fontsize=7, ha='right', va='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide unused axes
    for j in range(n_examples, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Representative Event Examples (3-Window Stability Classification)', fontsize=12, y=1.02)
    plt.tight_layout()
    outpath = APPENDIX_DIR / "event_examples.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def generate_null_control_figure():
    """
    Generate bar plot comparing null simulation ceilings with real LIGO.

    Shows:
    - Null baseline (nominal 30-100ms window)
    - Worst-case fixed-window tuning (noise/AICc/delay sweep)
    - Worst-case window tuning (40-120ms)
    - Real LIGO observed

    With 95% Wilson confidence intervals.
    """
    null_sim_path = OUTPUT_DIR / "null_simulation.json"
    if not null_sim_path.exists():
        print(f"  WARNING: {null_sim_path} not found. Run ligo_null_simulation.py first.")
        return

    with open(null_sim_path, 'r') as f:
        data = json.load(f)

    nf = data.get('null_fast', {})
    sweep = data.get('sweep', [])
    real_ligo = data.get('real_ligo', {})

    if not nf or not sweep:
        print("  WARNING: Incomplete null simulation data. Run with --mode both then --mode sweep.")
        return

    # Extract data
    null_baseline = 100 * nf.get('mislabel_rate', 0)
    null_baseline_lo = 100 * nf.get('mislabel_ci_lower', 0)
    null_baseline_hi = 100 * nf.get('mislabel_ci_upper', 0)

    # Find worst-case fixed-window (noise, AICc, delay sweeps)
    fixed_window_sweeps = [s for s in sweep if s.get('param') != 'late_window']
    if fixed_window_sweeps:
        max_fixed = max(fixed_window_sweeps, key=lambda x: x.get('delayed_rate', 0))
        worst_fixed = 100 * max_fixed.get('delayed_rate', 0)
        worst_fixed_hi = 100 * max_fixed.get('ci', [0, 0])[1]
        worst_fixed_lo = 100 * max_fixed.get('ci', [0, 0])[0]
    else:
        worst_fixed = worst_fixed_lo = worst_fixed_hi = 0

    # Find worst-case window
    window_sweeps = [s for s in sweep if s.get('param') == 'late_window']
    if window_sweeps:
        max_window = max(window_sweeps, key=lambda x: x.get('delayed_rate', 0))
        worst_window = 100 * max_window.get('delayed_rate', 0)
        worst_window_hi = 100 * max_window.get('ci', [0, 0])[1]
        worst_window_lo = 100 * max_window.get('ci', [0, 0])[0]
    else:
        worst_window = worst_window_lo = worst_window_hi = 0

    real_delayed = 100 * real_ligo.get('delayed_frac', 0.52)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['Null\nBaseline', 'Worst-Case\n(Fixed Window)', 'Worst-Case\n(Any Window)', 'Real\nLIGO']
    values = [null_baseline, worst_fixed, worst_window, real_delayed]
    errors_lo = [null_baseline - null_baseline_lo, worst_fixed - worst_fixed_lo, worst_window - worst_window_lo, 0]
    errors_hi = [null_baseline_hi - null_baseline, worst_fixed_hi - worst_fixed, worst_window_hi - worst_window, 0]

    colors = ['#4682B4', '#FF6347', '#FF6347', '#228B22']  # Blue, red, red, green
    x = np.arange(len(categories))

    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
    ax.errorbar(x[:3], values[:3], yerr=[errors_lo[:3], errors_hi[:3]],
                fmt='none', capsize=5, capthick=2, color='black', linewidth=2)

    # Add value labels on bars
    for i, (xi, val) in enumerate(zip(x, values)):
        if i < 3:
            label = f'{val:.1f}%'
        else:
            label = f'{val:.0f}%'
        ax.text(xi, val + 2, label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add horizontal line at real LIGO level for comparison
    ax.axhline(y=real_delayed, color='green', linestyle='--', alpha=0.5, linewidth=1.5)

    # Annotations
    ax.annotate('', xy=(2.5, worst_window + 3), xytext=(2.5, real_delayed - 3),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    gap = real_delayed - worst_window
    ax.text(2.7, (worst_window + real_delayed) / 2, f'{gap:.0f}%\ngap',
            ha='left', va='center', fontsize=10)

    ax.set_ylabel('Delayed Fraction (%)', fontsize=12)
    ax.set_title('Null Simulation Control: False-Delayed Ceiling vs Real LIGO', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 65)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add text box explaining the control
    textstr = ('Null simulation: Pure fast-geometry traces\n'
               'classified using window-decoupled pipeline.\n'
               'Max ceiling: ~16% (4× below real LIGO).')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    outpath = APPENDIX_DIR / "null_control_bars.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")

    # Also save PDF for LaTeX inclusion
    outpath_pdf = APPENDIX_DIR / "null_control_bars.pdf"
    plt.savefig(outpath_pdf, bbox_inches='tight')
    print(f"Saved: {outpath_pdf}")
    plt.close()


def main():
    print("=" * 70)
    print("Generating Appendix Figures")
    print("=" * 70)

    # Analyze all events
    processed = analyze_all_events()

    # Summary
    stable_iof = [e for e in processed if e['stability'] == 'stable_iof']
    stable_std = [e for e in processed if e['stability'] == 'stable_std']
    flips = [e for e in processed if e['stability'] == 'flip']
    failed = [e for e in processed if e['stability'] == 'failed']

    print(f"\nStability summary:")
    print(f"  Stable IOF: {len(stable_iof)}")
    print(f"  Stable STD: {len(stable_std)}")
    print(f"  Flip: {len(flips)}")
    print(f"  Failed: {len(failed)}")

    # Generate figures
    print("\n" + "-" * 40)
    print("Generating dip test figure (stable-only)...")
    generate_dip_test_figure(processed)

    print("\n" + "-" * 40)
    print("Running dip test on ALL OK events (unlabeled)...")
    generate_dip_test_unlabeled(processed)

    print("\n" + "-" * 40)
    print("Generating event examples...")
    generate_event_examples(processed)

    print("\n" + "-" * 40)
    print("Generating null control figure...")
    generate_null_control_figure()

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
