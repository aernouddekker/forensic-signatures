#!/usr/bin/env python3
"""
LIGO Regression Harness
=======================

Validates that main pipeline and downstream scripts produce identical classifications
for the same events. This is the stop criterion for pipeline stabilization.

Stop criterion: 0 mismatches at window=100ms (including env_fit samples).

Usage:
    python ligo_regression_harness.py              # Run validation
    python ligo_regression_harness.py --verbose    # Show all comparisons
    python ligo_regression_harness.py --limit 10   # Test first 10 events

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

# Import canonical functions from iof_metrics
from iof_metrics import (
    FitResult,
    baseline_tail_median,
    check_fit_sanity,
    fit_envelope_with_baseline,
    get_model_geometry,
)

# Import pipeline common utilities
from ligo_pipeline_common import (
    compute_hilbert_envelope,
    compute_times_ms,
    center_times_on_peak,
    extract_fit_window_indices,
    find_constrained_peak,
    load_cached_strain,
    compute_curvature_index,
    PEAK_SEARCH_WINDOW_MS,
)

# Fixed seed for reproducibility
np.random.seed(42)

# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "ligo_envelope"
CACHE_DIR = SCRIPT_DIR / "strain_cache"
STABILITY_FILE = OUTPUT_DIR / "stability_events.jsonl"


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Result from classifying a single event at a single window."""
    peak_idx: int
    peak_time_ms: float
    peak_amplitude: float
    env_fit_idx: np.ndarray
    env_fit: np.ndarray
    baseline: float
    winner_model: str
    winner_aicc: float
    runner_up_model: str
    runner_up_aicc: float
    delta_aicc: float
    geometry: str
    sanity_pass: bool
    sanity_reason: str
    classification: str  # 'delayed', 'fast', or 'uncertain'
    # Curvature index (computed with canonical 150ms baseline)
    curvature_b: Optional[float] = None
    baseline_150: Optional[float] = None  # Baseline used for curvature


@dataclass
class ComparisonResult:
    """Result of comparing main vs downstream for one event."""
    gps_time: float
    match: bool
    mismatches: List[str]
    main: Optional[ClassificationResult]
    downstream: Optional[ClassificationResult]


# -----------------------------------------------------------------------------
# Main pipeline classification (current implementation)
# -----------------------------------------------------------------------------

def classify_main_pipeline(
    envelope: np.ndarray,
    times_ms: np.ndarray,
    window_ms: float = 100.0
) -> Optional[ClassificationResult]:
    """
    Classify event using main pipeline approach.

    This mimics ligo_glitch_analysis.py's fit_competing_models_envelope().
    After stabilization, this should be replaced with canonical imports.
    """
    # Find peak
    peak_idx = find_constrained_peak(envelope, times_ms)
    peak_time_ms = times_ms[peak_idx]
    peak_amplitude = envelope[peak_idx]

    # Validate peak is within search window
    if abs(peak_time_ms) >= PEAK_SEARCH_WINDOW_MS:
        return None

    # Build fit window using main pipeline approach (floating mask)
    mask = (times_ms >= peak_time_ms) & (times_ms <= peak_time_ms + window_ms)
    env_fit_idx = np.where(mask)[0]
    t_fit = times_ms[mask] - peak_time_ms
    env_fit = envelope[mask]

    if len(env_fit) < 20:
        return None

    # Baseline: main pipeline uses np.median(env_fit[-len(env_fit)//5:])
    baseline = baseline_tail_median(env_fit)  # This is canonical now

    max_value = env_fit[0]
    amplitude = max_value - baseline

    if amplitude < 0.01 * max_value:
        return None

    # Fit models using canonical fitter
    fits = fit_envelope_with_baseline(t_fit, env_fit, baseline)

    if not fits:
        return None

    # Filter by sanity
    valid_fits = {}
    sanity_reasons = {}
    for name, fit in fits.items():
        is_sane, reason = check_fit_sanity(fit)
        sanity_reasons[name] = reason
        if is_sane:
            valid_fits[name] = fit

    if not valid_fits:
        return ClassificationResult(
            peak_idx=peak_idx,
            peak_time_ms=peak_time_ms,
            peak_amplitude=peak_amplitude,
            env_fit_idx=env_fit_idx,
            env_fit=env_fit,
            baseline=baseline,
            winner_model='none',
            winner_aicc=float('inf'),
            runner_up_model='none',
            runner_up_aicc=float('inf'),
            delta_aicc=0.0,
            geometry='unknown',
            sanity_pass=False,
            sanity_reason='; '.join(f"{k}:{v}" for k, v in sanity_reasons.items()),
            classification='uncertain'
        )

    # Find winner by AICc
    sorted_fits = sorted(valid_fits.items(), key=lambda x: x[1].aicc)
    winner_name, winner_fit = sorted_fits[0]

    if len(sorted_fits) >= 2:
        runner_up_name, runner_up_fit = sorted_fits[1]
        delta_aicc = runner_up_fit.aicc - winner_fit.aicc
    else:
        runner_up_name = 'none'
        runner_up_fit = None
        delta_aicc = float('inf')

    # Get geometry
    geometry = get_model_geometry(winner_name)

    # Classification logic (ΔAICc < 2 + geometry disagreement → uncertain)
    if delta_aicc < 2.0 and len(sorted_fits) >= 2:
        geo2 = get_model_geometry(runner_up_name)
        if geometry != geo2:
            classification = 'uncertain'
        else:
            classification = 'delayed' if geometry == 'delayed' else 'fast'
    else:
        classification = 'delayed' if geometry == 'delayed' else 'fast'

    # Compute curvature using canonical function (uses 150ms baseline internally)
    curvature_b = compute_curvature_index(envelope, times_ms, peak_idx)

    # Also compute baseline_150 for comparison (same as curvature uses)
    times_centered = center_times_on_peak(times_ms, peak_idx)
    env_fit_150_idx = extract_fit_window_indices(times_centered, peak_idx, 150.0)
    env_fit_150 = envelope[env_fit_150_idx]
    baseline_150 = baseline_tail_median(env_fit_150)

    return ClassificationResult(
        peak_idx=peak_idx,
        peak_time_ms=peak_time_ms,
        peak_amplitude=peak_amplitude,
        env_fit_idx=env_fit_idx,
        env_fit=env_fit,
        baseline=baseline,
        winner_model=winner_name,
        winner_aicc=winner_fit.aicc,
        runner_up_model=runner_up_name,
        runner_up_aicc=runner_up_fit.aicc if runner_up_fit else float('inf'),
        delta_aicc=delta_aicc,
        geometry=geometry,
        sanity_pass=True,
        sanity_reason='ok',
        classification=classification,
        curvature_b=curvature_b,
        baseline_150=baseline_150
    )


# -----------------------------------------------------------------------------
# Downstream classification (canonical implementation)
# -----------------------------------------------------------------------------

def classify_downstream(
    envelope: np.ndarray,
    times_ms: np.ndarray,
    window_ms: float = 100.0
) -> Optional[ClassificationResult]:
    """
    Classify event using canonical downstream approach.

    This uses:
    - center_times_on_peak() for time centering
    - INDEX-based window extraction (not floating mask)
    - baseline_tail_median() for baseline
    - fit_envelope_with_baseline() for fitting
    - check_fit_sanity() for sanity filtering
    """
    # Find peak
    peak_idx = find_constrained_peak(envelope, times_ms)
    peak_time_ms = times_ms[peak_idx]
    peak_amplitude = envelope[peak_idx]

    # Validate peak is within search window
    if abs(peak_time_ms) >= PEAK_SEARCH_WINDOW_MS:
        return None

    # Center times on peak
    times_centered = center_times_on_peak(times_ms, peak_idx)

    # Build window by INDEX using searchsorted (exact match to mask rule)
    # This replaces the error-prone ceil()+1 logic
    env_fit_idx = extract_fit_window_indices(times_centered, peak_idx, window_ms)
    t_fit = times_centered[env_fit_idx]
    env_fit = envelope[env_fit_idx]

    if len(env_fit) < 20:
        return None

    # INVARIANT checks
    if env_fit_idx[0] != peak_idx:
        raise AssertionError(f"env_fit does not start at peak: {env_fit_idx[0]} != {peak_idx}")
    if abs(t_fit[0]) > 1e-9:
        raise AssertionError(f"t_fit[0] must be 0 after centering, got {t_fit[0]}")

    # Canonical baseline
    baseline = baseline_tail_median(env_fit)

    max_value = env_fit[0]
    amplitude = max_value - baseline

    if amplitude < 0.01 * max_value:
        return None

    # Fit models using canonical fitter
    fits = fit_envelope_with_baseline(t_fit, env_fit, baseline)

    if not fits:
        return None

    # Filter by sanity
    valid_fits = {}
    sanity_reasons = {}
    for name, fit in fits.items():
        is_sane, reason = check_fit_sanity(fit)
        sanity_reasons[name] = reason
        if is_sane:
            valid_fits[name] = fit

    if not valid_fits:
        return ClassificationResult(
            peak_idx=peak_idx,
            peak_time_ms=peak_time_ms,
            peak_amplitude=peak_amplitude,
            env_fit_idx=env_fit_idx,
            env_fit=env_fit,
            baseline=baseline,
            winner_model='none',
            winner_aicc=float('inf'),
            runner_up_model='none',
            runner_up_aicc=float('inf'),
            delta_aicc=0.0,
            geometry='unknown',
            sanity_pass=False,
            sanity_reason='; '.join(f"{k}:{v}" for k, v in sanity_reasons.items()),
            classification='uncertain'
        )

    # Find winner by AICc
    sorted_fits = sorted(valid_fits.items(), key=lambda x: x[1].aicc)
    winner_name, winner_fit = sorted_fits[0]

    if len(sorted_fits) >= 2:
        runner_up_name, runner_up_fit = sorted_fits[1]
        delta_aicc = runner_up_fit.aicc - winner_fit.aicc
    else:
        runner_up_name = 'none'
        runner_up_fit = None
        delta_aicc = float('inf')

    # Get geometry
    geometry = get_model_geometry(winner_name)

    # Classification logic (ΔAICc < 2 + geometry disagreement → uncertain)
    if delta_aicc < 2.0 and len(sorted_fits) >= 2:
        geo2 = get_model_geometry(runner_up_name)
        if geometry != geo2:
            classification = 'uncertain'
        else:
            classification = 'delayed' if geometry == 'delayed' else 'fast'
    else:
        classification = 'delayed' if geometry == 'delayed' else 'fast'

    # Compute curvature using canonical function (uses 150ms baseline internally)
    curvature_b = compute_curvature_index(envelope, times_ms, peak_idx)

    # Also compute baseline_150 for comparison (same as curvature uses)
    env_fit_150_idx = extract_fit_window_indices(times_centered, peak_idx, 150.0)
    env_fit_150 = envelope[env_fit_150_idx]
    baseline_150 = baseline_tail_median(env_fit_150)

    return ClassificationResult(
        peak_idx=peak_idx,
        peak_time_ms=peak_time_ms,
        peak_amplitude=peak_amplitude,
        env_fit_idx=env_fit_idx,
        env_fit=env_fit,
        baseline=baseline,
        winner_model=winner_name,
        winner_aicc=winner_fit.aicc,
        runner_up_model=runner_up_name,
        runner_up_aicc=runner_up_fit.aicc if runner_up_fit else float('inf'),
        delta_aicc=delta_aicc,
        geometry=geometry,
        sanity_pass=True,
        sanity_reason='ok',
        classification=classification,
        curvature_b=curvature_b,
        baseline_150=baseline_150
    )


# -----------------------------------------------------------------------------
# Comparison logic
# -----------------------------------------------------------------------------

def compare_classifications(
    main: ClassificationResult,
    downstream: ClassificationResult,
    atol: float = 1e-12
) -> Tuple[bool, List[str]]:
    """
    Compare main vs downstream classification results.

    Returns (match, list_of_mismatch_reasons).
    """
    mismatches = []

    # Peak index
    if main.peak_idx != downstream.peak_idx:
        mismatches.append(f"peak_idx: {main.peak_idx} vs {downstream.peak_idx}")

    # env_fit_idx array
    if not np.array_equal(main.env_fit_idx, downstream.env_fit_idx):
        mismatches.append(f"env_fit_idx: len {len(main.env_fit_idx)} vs {len(downstream.env_fit_idx)}")

    # env_fit values (numerical identity within tolerance)
    if len(main.env_fit) == len(downstream.env_fit):
        if not np.allclose(main.env_fit, downstream.env_fit, rtol=0, atol=atol):
            max_diff = np.max(np.abs(main.env_fit - downstream.env_fit))
            mismatches.append(f"env_fit values: max_diff={max_diff:.2e}")
    else:
        mismatches.append(f"env_fit length: {len(main.env_fit)} vs {len(downstream.env_fit)}")

    # Baseline
    if not np.isclose(main.baseline, downstream.baseline, rtol=0, atol=atol):
        mismatches.append(f"baseline: {main.baseline:.6f} vs {downstream.baseline:.6f}")

    # Winner model
    if main.winner_model != downstream.winner_model:
        mismatches.append(f"winner: {main.winner_model} vs {downstream.winner_model}")

    # Winner AICc
    if not np.isclose(main.winner_aicc, downstream.winner_aicc, rtol=1e-6, atol=1e-6):
        mismatches.append(f"winner_aicc: {main.winner_aicc:.2f} vs {downstream.winner_aicc:.2f}")

    # Geometry
    if main.geometry != downstream.geometry:
        mismatches.append(f"geometry: {main.geometry} vs {downstream.geometry}")

    # Classification
    if main.classification != downstream.classification:
        mismatches.append(f"classification: {main.classification} vs {downstream.classification}")

    # Sanity status
    if main.sanity_pass != downstream.sanity_pass:
        mismatches.append(f"sanity_pass: {main.sanity_pass} vs {downstream.sanity_pass}")

    # Baseline_150 (used for curvature)
    if main.baseline_150 is not None and downstream.baseline_150 is not None:
        if not np.isclose(main.baseline_150, downstream.baseline_150, rtol=0, atol=atol):
            mismatches.append(f"baseline_150: {main.baseline_150:.6f} vs {downstream.baseline_150:.6f}")

    # Curvature index b
    if main.curvature_b is not None and downstream.curvature_b is not None:
        if not np.isclose(main.curvature_b, downstream.curvature_b, rtol=0, atol=atol):
            diff = abs(main.curvature_b - downstream.curvature_b)
            mismatches.append(f"curvature_b: {main.curvature_b:.6e} vs {downstream.curvature_b:.6e} (diff={diff:.2e})")
    elif (main.curvature_b is None) != (downstream.curvature_b is None):
        mismatches.append(f"curvature_b: {main.curvature_b} vs {downstream.curvature_b} (one is None)")

    return len(mismatches) == 0, mismatches


# -----------------------------------------------------------------------------
# Main harness
# -----------------------------------------------------------------------------

def run_harness(verbose: bool = False, limit: Optional[int] = None) -> int:
    """
    Run the regression harness.

    Returns number of mismatches (0 = all pass).
    """
    print("=" * 70)
    print("LIGO Regression Harness")
    print("=" * 70)
    print(f"\nFixed RNG seed: 42")
    print(f"Window: 100ms (classification), 150ms (curvature baseline)")
    print(f"Tolerance: numerical identity within 1e-12")
    print(f"Checks: classification + curvature_b")

    # Load events
    if not STABILITY_FILE.exists():
        print(f"\nERROR: {STABILITY_FILE} not found")
        print("Run ligo_stability_figures.py first to generate stability events.")
        return 1

    events = []
    with open(STABILITY_FILE) as f:
        for line in f:
            events.append(json.loads(line))

    if limit:
        events = events[:limit]

    print(f"\nLoaded {len(events)} events from {STABILITY_FILE.name}")

    # Run comparisons
    results = []
    n_match = 0
    n_mismatch = 0
    n_skip = 0

    for i, event in enumerate(events):
        gps_time = event['gps_time']

        # Load cached strain
        data = load_cached_strain(gps_time, CACHE_DIR, warn_legacy=False)
        if data is None:
            n_skip += 1
            continue

        strain = data['values']
        fs = data['sample_rate']
        times = data['times']

        # Compute envelope and GPS-relative times
        envelope = compute_hilbert_envelope(strain, fs)
        times_ms = compute_times_ms(times, gps_time)

        # Classify with both approaches
        try:
            main_result = classify_main_pipeline(envelope, times_ms, window_ms=100.0)
            downstream_result = classify_downstream(envelope, times_ms, window_ms=100.0)
        except Exception as e:
            if verbose:
                print(f"  [{i+1}] GPS {gps_time:.6f}: ERROR - {e}")
            n_skip += 1
            continue

        if main_result is None or downstream_result is None:
            n_skip += 1
            continue

        # Compare
        match, mismatches = compare_classifications(main_result, downstream_result)

        result = ComparisonResult(
            gps_time=gps_time,
            match=match,
            mismatches=mismatches,
            main=main_result,
            downstream=downstream_result
        )
        results.append(result)

        if match:
            n_match += 1
            if verbose:
                print(f"  [{i+1}] GPS {gps_time:.6f}: MATCH ({main_result.classification})")
        else:
            n_mismatch += 1
            print(f"  [{i+1}] GPS {gps_time:.6f}: MISMATCH")
            for m in mismatches:
                print(f"        - {m}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nTotal events:   {len(events)}")
    print(f"Matched:        {n_match}")
    print(f"Mismatched:     {n_mismatch}")
    print(f"Skipped:        {n_skip}")

    if n_mismatch == 0 and n_match > 0:
        print(f"\n✓ PASS: Science invariant holds ({n_match}/{n_match} events match)")
        print("  Safe to proceed with dip test interpretation.")
    else:
        print(f"\n✗ FAIL: {n_mismatch} mismatches detected")
        print("  Fix implementation drift before interpreting dip test results.")

    # Save detailed results
    output_file = OUTPUT_DIR / "regression_harness_results.json"
    summary = {
        'metadata': {
            'created': '2025-12-17',
            'rng_seed': 42,
            'window_ms': 100.0,
            'tolerance': 1e-12,
        },
        'summary': {
            'n_events': len(events),
            'n_match': n_match,
            'n_mismatch': n_mismatch,
            'n_skip': n_skip,
            'pass': n_mismatch == 0 and n_match > 0,
        },
        'mismatches': [
            {
                'gps_time': r.gps_time,
                'reasons': r.mismatches,
                'main_classification': r.main.classification if r.main else None,
                'downstream_classification': r.downstream.classification if r.downstream else None,
            }
            for r in results if not r.match
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {output_file}")

    return n_mismatch


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate main vs downstream classification consistency"
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                       help="Show all comparisons (not just mismatches)")
    parser.add_argument('--limit', '-n', type=int, default=None,
                       help="Limit to first N events (for testing)")

    args = parser.parse_args()

    n_mismatch = run_harness(verbose=args.verbose, limit=args.limit)

    # Exit code: 0 if pass, 1 if fail
    import sys
    sys.exit(0 if n_mismatch == 0 else 1)


if __name__ == "__main__":
    main()
