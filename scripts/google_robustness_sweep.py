#!/usr/bin/env python3
"""
Robustness Sweep for Model-Based IOF Classification
====================================================

Runs the model-based classification on frozen events with varying inference
parameters to demonstrate that classification stability is robust to
reasonable methodological choices.

Input: frozen_events.json (created by google_mcewan_analysis.py --freeze-only)
Data source: McEwen et al. (2022), Figshare DOI 10.6084/m9.figshare.16673851

Parameter sweep blocks:
- Block 1: Core Inference Sweep (12 configurations)
  - window_ms: [60, 100, 150] ms
  - alignment: [argmin, threshold_crossing]
  - fix_baseline: [True, False]

- Block 2: Model-Set Robustness (3 configurations)
  - Full model set (exponential, power_law, sigmoid, delayed)
  - Without power_law
  - Without sigmoid

Output:
- output/robustness_sweep/run_summaries.json: Per-run classification summaries
- output/robustness_sweep/stability_table.json: Event-level stability statistics
- output/robustness_sweep/detailed_results.json: Full results for all events

Stability classification:
- Stable IOF: >= 80% of runs classify as IOF (delayed geometry)
- Stable STD: >= 80% of runs classify as Standard (fast geometry)
- Flip: Event switches between IOF and STD across runs
- Uncertain: Event has uncertain classification in most runs

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import hashlib
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import from iof_metrics
from iof_metrics import (
    FitParams, ClassificationParams, FitResult, NumpyEncoder,
    exponential_recovery, exponential_recovery_fixed,
    power_law_recovery, logistic_sigmoid, delayed_exponential,
    compute_aic, compute_aicc, fit_model,
    compute_model_t_peak, get_model_geometry
)


# =============================================================================
# Alignment Methods
# =============================================================================

def align_argmin(t_window: np.ndarray, y_window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align to minimum value (current default)."""
    min_idx = np.argmin(y_window)
    return t_window[min_idx:] - t_window[min_idx], y_window[min_idx:]


def align_threshold_crossing(
    t_window: np.ndarray,
    y_window: np.ndarray,
    baseline: float,
    threshold_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align to threshold crossing on recovery.

    t=0 is when signal first crosses threshold_frac * amplitude above minimum.
    """
    min_idx = np.argmin(y_window)
    min_val = y_window[min_idx]
    amplitude = baseline - min_val

    # Guard against non-positive amplitude (pathological case)
    if amplitude <= 0:
        return align_argmin(t_window, y_window)

    threshold = min_val + threshold_frac * amplitude

    # Find first crossing after minimum
    for i in range(min_idx, len(y_window)):
        if y_window[i] >= threshold:
            t_aligned = t_window[i:] - t_window[i]
            y_aligned = y_window[i:]
            return t_aligned, y_aligned

    # Fallback to argmin if no crossing found
    return align_argmin(t_window, y_window)


# =============================================================================
# Model Fitting with Configurable Model Set
# =============================================================================

def fit_competing_models_configurable(
    t_ms: np.ndarray,
    y: np.ndarray,
    baseline_estimate: float,
    params: FitParams,
    model_set: Set[str],
    min_value_true: Optional[float] = None
) -> Dict[str, FitResult]:
    """
    Fit competing models with configurable model set.

    Args:
        t_ms: Time array in milliseconds
        y: Data array
        baseline_estimate: Canonical baseline (tail-median of frozen window)
        params: Fitting parameters
        model_set: Set of model names to include
        min_value_true: True minimum from frozen event (for stable amplitude)

    Returns:
        Dictionary of model_name -> FitResult
    """
    results = {}

    if len(y) < 10:
        return results

    # Amplitude gate uses frozen min_value_true for INVARIANT membership decisions.
    # This prevents alignment-dependent event filtering.
    # NO abs() - require positive amplitude (upward recovery: baseline > min)
    # Negative amplitude indicates data/anchoring problem, should fail fast.
    if min_value_true is not None:
        amplitude_gate = baseline_estimate - min_value_true
    else:
        amplitude_gate = baseline_estimate - np.min(y)

    # HARD FAIL on negative amplitude (pathological baseline/anchoring)
    # Must raise BEFORE the "too small" skip, or negatives slip through silently
    if amplitude_gate <= 0:
        raise AssertionError(
            f"Amplitude sanity failed: baseline={baseline_estimate:.2f}, min_value_true={min_value_true}"
        )

    if amplitude_gate < 1:
        # Too small for reliable fitting - skip event (but not pathological)
        return results

    # For optimizer initialization (p0), use aligned-window values.
    # This keeps the solver near reality even when threshold alignment
    # discards the true minimum.
    min_value = np.min(y)
    amplitude = baseline_estimate - min_value  # p0 amplitude (may be smaller under threshold align)

    # Guard: if aligned amplitude went negative/tiny, use gate amplitude for p0
    if amplitude < 0.1:
        amplitude = max(amplitude_gate, 0.1)

    # Baseline is INVARIANT: always use the frozen canonical baseline.
    baseline_used = baseline_estimate

    # Compute tail median ONCE using identical formula as fit_models() in main analysis
    # Formula: last 20% of window (y[int(0.8 * len(y)):])
    tail_start = int(0.8 * len(y))
    tail_median = np.median(y[tail_start:]) if tail_start < len(y) else np.median(y)

    # TRIPWIRE: Baseline sanity - should be reasonably close to tail of window
    assert baseline_used >= tail_median * 0.5, \
        f"Baseline sanity failed: baseline={baseline_used:.1f} but tail_median={tail_median:.1f}"

    # Data-derived lower bound for fitted baseline (blocks "baseline teleportation")
    # Identical formula to fit_models() in main analysis
    baseline_lower = max(min_value + 1, tail_median * 0.8)

    # TRIPWIRE: Bounds sanity - ensure lower < upper
    assert baseline_lower < baseline_used * 2, \
        f"Invalid bounds: baseline_lower={baseline_lower:.2f} >= 2*baseline_used={2*baseline_used:.2f}"

    # --- Exponential (free baseline) ---
    if 'exponential' in model_set:
        results['exponential'] = fit_model(
            t_ms, y,
            exponential_recovery,
            p0=[amplitude, 10.0, baseline_used],
            bounds=([0, 0.1, baseline_lower], [amplitude * 2, 200, baseline_used * 2]),
            model_name='exponential',
            n_params=3,
            maxfev=params.maxfev
        )

    # --- Exponential (fixed baseline) ---
    if 'exponential_fixed' in model_set:
        results['exponential_fixed'] = fit_model(
            t_ms, y,
            exponential_recovery_fixed(baseline_used),
            p0=[amplitude, 10.0],
            bounds=([0, 0.1], [amplitude * 2, 200]),
            model_name='exponential_fixed',
            n_params=2,
            maxfev=params.maxfev
        )

    # --- Power Law (free baseline) ---
    if 'power_law' in model_set:
        results['power_law'] = fit_model(
            t_ms, y,
            power_law_recovery,
            p0=[amplitude, 5.0, baseline_used],
            bounds=([0, 0.1, baseline_lower], [amplitude * 2, 200, baseline_used * 2]),
            model_name='power_law',
            n_params=3,
            maxfev=params.maxfev
        )

    # --- Sigmoid (IOF) ---
    if 'sigmoid' in model_set:
        results['sigmoid'] = fit_model(
            t_ms, y,
            logistic_sigmoid,
            p0=[amplitude, 0.2, 15.0, min_value],
            bounds=([0, 0.01, 1, 0], [amplitude * 2, 2, 80, baseline_used]),
            model_name='sigmoid',
            n_params=4,
            maxfev=params.maxfev
        )

    # --- Delayed Exponential (IOF) ---
    if 'delayed' in model_set:
        results['delayed'] = fit_model(
            t_ms, y,
            delayed_exponential,
            p0=[amplitude, 15.0, 5.0, min_value],
            bounds=([0, 0.1, 0, 0], [amplitude * 2, 200, 50, baseline_used]),
            model_name='delayed',
            n_params=4,
            maxfev=params.maxfev
        )

    return results


def classify_event_model_based_configurable(
    fits: Dict[str, FitResult]
) -> Tuple[str, str, str, float, float]:
    """
    Classify event using model-based approach.

    Returns:
        classification: 'standard', 'iof', or 'uncertain'
        reason: Explanation string
        winning_model: Name of best-fit model
        model_t_peak: t_peak from winning model (ms)
        delta_aicc: AICc difference between best and second-best
    """
    # Get successful fits only
    valid_fits = {k: v for k, v in fits.items() if v.success}

    if not valid_fits:
        return 'uncertain', 'no successful fits', 'none', 0.0, 0.0

    # Find best model by AICc
    sorted_fits = sorted(valid_fits.items(), key=lambda x: x[1].aicc)
    best_name, best_fit = sorted_fits[0]

    # Compute delta AICc to second-best (if available)
    if len(sorted_fits) >= 2:
        second_name, second_fit = sorted_fits[1]
        delta_aicc = second_fit.aicc - best_fit.aicc
    else:
        delta_aicc = float('inf')

    # Compute model-based t_peak from winning model
    model_t_peak = compute_model_t_peak(best_name, best_fit.params)

    # Get model geometry
    geometry = get_model_geometry(best_name)

    # Classification logic
    reasons = []

    # Check if model selection is confident (delta_aicc >= 2)
    if delta_aicc < 2.0:
        # Models are comparable - check if they agree on geometry
        geometries = [get_model_geometry(name) for name, _ in sorted_fits[:2]]
        if geometries[0] == geometries[1]:
            # Both top models have same geometry - confident classification
            reasons.append(f"top models agree on geometry ({geometry})")
        else:
            # Models disagree on geometry - uncertain
            reasons.append(f"model selection ambiguous (dAICc={delta_aicc:.1f}<2)")
            return 'uncertain', "; ".join(reasons), best_name, model_t_peak, delta_aicc

    # Classify based on geometry
    if geometry == 'fast':
        classification = 'standard'
        reasons.append(f"fast geometry ({best_name})")
    elif geometry == 'delayed':
        classification = 'iof'
        reasons.append(f"delayed geometry ({best_name})")
    else:
        classification = 'uncertain'
        reasons.append(f"unknown geometry ({best_name})")

    reasons.append(f"dAICc={delta_aicc:.1f}")

    return classification, "; ".join(reasons), best_name, model_t_peak, delta_aicc


# =============================================================================
# Sweep Configuration
# =============================================================================

@dataclass
class SweepRunConfig:
    """Configuration for a single sweep run."""
    run_id: str
    block: str
    window_ms: float
    alignment: str  # 'argmin' or 'threshold'
    fix_baseline: bool
    model_set: Set[str]

    def to_dict(self) -> dict:
        return {
            'run_id': self.run_id,
            'block': self.block,
            'window_ms': self.window_ms,
            'alignment': self.alignment,
            'fix_baseline': self.fix_baseline,
            'model_set': list(self.model_set)
        }


@dataclass
class EventResult:
    """Result for a single event in a single run."""
    event_id: str
    classification: str
    winning_model: str
    model_t_peak: float
    delta_aicc: float
    reason: str


@dataclass
class RunSummary:
    """Summary statistics for a single run."""
    config: SweepRunConfig
    n_events: int
    n_iof: int
    n_standard: int
    n_uncertain: int
    iof_fraction: float
    iof_fraction_of_classified: float
    median_delta_aicc: float
    strong_evidence_frac: float  # dAICc >= 10
    moderate_evidence_frac: float  # 4 <= dAICc < 10
    weak_evidence_frac: float  # dAICc < 2


def generate_sweep_configs() -> List[SweepRunConfig]:
    """Generate all sweep configurations for Block 1 and Block 2."""
    configs = []

    # Model sets depend on fix_baseline:
    # - fix_baseline=True: use exponential_fixed (2-param), no free-baseline exponential
    # - fix_baseline=False: use exponential (3-param) + power_law (free baseline)
    # IOF models (sigmoid, delayed) always included
    fixed_baseline_models = {'exponential_fixed', 'sigmoid', 'delayed'}
    free_baseline_models = {'exponential', 'power_law', 'sigmoid', 'delayed'}

    # Block 1: Core Inference Sweep (12 runs)
    windows = [60, 100, 150]
    alignments = ['argmin', 'threshold']
    fix_baselines = [True, False]

    run_idx = 1
    for window in windows:
        for alignment in alignments:
            for fix_bl in fix_baselines:
                # fix_baseline now ACTUALLY changes the model set
                model_set = fixed_baseline_models.copy() if fix_bl else free_baseline_models.copy()
                configs.append(SweepRunConfig(
                    run_id=f"B1_{run_idx:02d}",
                    block="1",
                    window_ms=float(window),
                    alignment=alignment,
                    fix_baseline=fix_bl,
                    model_set=model_set
                ))
                run_idx += 1

    # Block 2: Model-Set Robustness (3 runs at default settings)
    # Default: 100ms, argmin, fix_baseline=True (uses fixed_baseline_models)
    # Full model set for comparison
    full_model_set = {'exponential', 'exponential_fixed', 'power_law', 'sigmoid', 'delayed'}

    configs.append(SweepRunConfig(
        run_id="B2_01_full",
        block="2",
        window_ms=100.0,
        alignment='argmin',
        fix_baseline=True,
        model_set=full_model_set.copy()
    ))

    configs.append(SweepRunConfig(
        run_id="B2_02_no_powerlaw",
        block="2",
        window_ms=100.0,
        alignment='argmin',
        fix_baseline=True,
        model_set={'exponential', 'exponential_fixed', 'sigmoid', 'delayed'}
    ))

    configs.append(SweepRunConfig(
        run_id="B2_03_no_sigmoid",
        block="2",
        window_ms=100.0,
        alignment='argmin',
        fix_baseline=True,
        model_set={'exponential', 'exponential_fixed', 'power_law', 'delayed'}
    ))

    return configs


# =============================================================================
# Main Sweep Logic
# =============================================================================

def run_single_config(
    events: List[dict],
    config: SweepRunConfig
) -> Tuple[List[EventResult], RunSummary]:
    """
    Run classification on all events with a single configuration.

    Args:
        events: List of frozen event dictionaries
        config: Sweep configuration

    Returns:
        List of EventResult, RunSummary
    """
    results = []
    fit_params = FitParams(fix_baseline=config.fix_baseline)

    delta_aiccs = []

    for evt in events:
        event_id = f"{evt['dataset']}_{evt['event_index']}"

        try:
            t_window = np.array(evt['t_window'])
            y_window = np.array(evt['y_window'])
            baseline = evt['baseline']

            # Apply alignment
            if config.alignment == 'argmin':
                t_aligned, y_aligned = align_argmin(t_window, y_window)
            else:  # threshold
                t_aligned, y_aligned = align_threshold_crossing(t_window, y_window, baseline)

            # TRIPWIRE: Alignment must produce t[0] == 0 (within float tolerance)
            if len(t_aligned) > 0:
                assert abs(t_aligned[0]) < 1e-6, f"Alignment failed: t[0]={t_aligned[0]}"

            # Truncate to requested window
            mask = t_aligned <= config.window_ms
            t_fit = t_aligned[mask]
            y_fit = y_aligned[mask]

            if len(t_fit) < 10:
                results.append(EventResult(
                    event_id=event_id,
                    classification='uncertain',
                    winning_model='none',
                    model_t_peak=0.0,
                    delta_aicc=0.0,
                    reason='insufficient data after alignment'
                ))
                continue

            # Fit models (pass frozen min_value_true for stable amplitude)
            min_value_true = evt.get('min_value_true')
            fits = fit_competing_models_configurable(
                t_fit, y_fit, baseline, fit_params, config.model_set,
                min_value_true=min_value_true
            )

            # Classify
            classification, reason, winning_model, model_t_peak, delta_aicc = \
                classify_event_model_based_configurable(fits)

            results.append(EventResult(
                event_id=event_id,
                classification=classification,
                winning_model=winning_model,
                model_t_peak=model_t_peak,
                delta_aicc=delta_aicc,
                reason=reason
            ))

            # Only collect finite delta_aicc for summary stats (inf from single-model fits)
            if np.isfinite(delta_aicc):
                delta_aiccs.append(delta_aicc)

        except AssertionError:
            # Sanity check failures are FATAL - don't silently convert to "uncertain"
            raise
        except Exception as e:
            results.append(EventResult(
                event_id=event_id,
                classification='uncertain',
                winning_model='none',
                model_t_peak=0.0,
                delta_aicc=0.0,
                reason=f'error: {str(e)}'
            ))

    # Compute summary statistics
    n_iof = sum(1 for r in results if r.classification == 'iof')
    n_standard = sum(1 for r in results if r.classification == 'standard')
    n_uncertain = sum(1 for r in results if r.classification == 'uncertain')
    n_classified = n_iof + n_standard

    # Evidence strength distribution
    if delta_aiccs:
        strong_evidence = sum(1 for d in delta_aiccs if d >= 10) / len(delta_aiccs)
        moderate_evidence = sum(1 for d in delta_aiccs if 4 <= d < 10) / len(delta_aiccs)
        weak_evidence = sum(1 for d in delta_aiccs if d < 2) / len(delta_aiccs)
        median_delta = np.median(delta_aiccs)
    else:
        strong_evidence = moderate_evidence = weak_evidence = 0.0
        median_delta = 0.0

    summary = RunSummary(
        config=config,
        n_events=len(results),
        n_iof=n_iof,
        n_standard=n_standard,
        n_uncertain=n_uncertain,
        iof_fraction=n_iof / len(results) if results else 0.0,
        iof_fraction_of_classified=n_iof / n_classified if n_classified > 0 else 0.0,
        median_delta_aicc=median_delta,
        strong_evidence_frac=strong_evidence,
        moderate_evidence_frac=moderate_evidence,
        weak_evidence_frac=weak_evidence
    )

    return results, summary


def compute_stability_table(
    all_results: Dict[str, List[EventResult]]
) -> Dict[str, dict]:
    """
    Compute event-level stability across all runs.

    Determines how consistently each event is classified across different
    parameter configurations. Events are categorized as stable_iof, stable_std,
    flip (switches between IOF/STD), or uncertain_fluctuate.

    Returns:
        Dictionary with stability statistics and per-event details
    """
    # Collect classifications per event across all runs
    event_classifications = defaultdict(list)

    for run_id, results in all_results.items():
        for r in results:
            event_classifications[r.event_id].append(r.classification)

    # Compute stability metrics
    n_events = len(event_classifications)
    stable_iof = 0  # IOF in >= 80% of runs
    stable_std = 0  # STD in >= 80% of runs
    fluctuate_uncertain = 0  # Fluctuates but only within uncertain
    true_flip = 0  # Flips between IOF <-> STD

    event_stability = {}
    n_runs = len(all_results)
    threshold = 0.8  # 80% threshold for stability

    for event_id, classifications in event_classifications.items():
        n_iof = sum(1 for c in classifications if c == 'iof')
        n_std = sum(1 for c in classifications if c == 'standard')
        n_unc = sum(1 for c in classifications if c == 'uncertain')
        n_total = len(classifications)

        iof_frac = n_iof / n_total
        std_frac = n_std / n_total

        if iof_frac >= threshold:
            stable_iof += 1
            stability = 'stable_iof'
        elif std_frac >= threshold:
            stable_std += 1
            stability = 'stable_std'
        elif n_iof > 0 and n_std > 0:
            # Both IOF and STD appear - true flip
            true_flip += 1
            stability = 'flip'
        else:
            # Fluctuates but no IOF<->STD flip
            fluctuate_uncertain += 1
            stability = 'uncertain_fluctuate'

        event_stability[event_id] = {
            'n_iof': n_iof,
            'n_std': n_std,
            'n_uncertain': n_unc,
            'iof_frac': iof_frac,
            'std_frac': std_frac,
            'stability': stability
        }

    summary = {
        'n_events': n_events,
        'n_runs': n_runs,
        'stable_iof': stable_iof,
        'stable_iof_pct': 100 * stable_iof / n_events if n_events > 0 else 0,
        'stable_std': stable_std,
        'stable_std_pct': 100 * stable_std / n_events if n_events > 0 else 0,
        'fluctuate_uncertain': fluctuate_uncertain,
        'fluctuate_uncertain_pct': 100 * fluctuate_uncertain / n_events if n_events > 0 else 0,
        'true_flip': true_flip,
        'true_flip_pct': 100 * true_flip / n_events if n_events > 0 else 0,
        'event_details': event_stability
    }

    return summary


def main():
    """Run the full robustness sweep."""
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'output' / 'robustness_sweep'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen events
    frozen_path = script_dir / 'output' / 'frozen_events.json'
    print(f"Loading frozen events from {frozen_path}...")

    with open(frozen_path, 'r') as f:
        frozen_data = json.load(f)

    events = frozen_data['events']
    print(f"Loaded {len(events)} frozen events")

    # Generate sweep configurations
    configs = generate_sweep_configs()
    print(f"\nRunning {len(configs)} sweep configurations...")
    print("=" * 70)

    # Run all configurations
    all_results = {}
    all_summaries = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running {config.run_id}...")
        print(f"  window={config.window_ms}ms, align={config.alignment}, "
              f"fix_bl={config.fix_baseline}, models={len(config.model_set)}")

        results, summary = run_single_config(events, config)
        all_results[config.run_id] = results
        all_summaries.append(summary)

        print(f"  Results: IOF={summary.n_iof} ({100*summary.iof_fraction:.1f}%), "
              f"STD={summary.n_standard} ({100*summary.n_standard/summary.n_events:.1f}%), "
              f"UNC={summary.n_uncertain}")
        print(f"  IOF of classified: {100*summary.iof_fraction_of_classified:.1f}%")
        print(f"  Median dAICc: {summary.median_delta_aicc:.1f}")

    # Compute stability table
    print("\n" + "=" * 70)
    print("Computing event-level stability table...")
    stability = compute_stability_table(all_results)

    # Print stability summary
    print("\n" + "=" * 70)
    print("EVENT-LEVEL STABILITY SUMMARY")
    print("=" * 70)
    print(f"Total events: {stability['n_events']}")
    print(f"Total runs: {stability['n_runs']}")
    print()
    print(f"Stable IOF (>=80% of runs):     {stability['stable_iof']:3d} ({stability['stable_iof_pct']:.1f}%)")
    print(f"Stable STD (>=80% of runs):     {stability['stable_std']:3d} ({stability['stable_std_pct']:.1f}%)")
    print(f"Fluctuate (uncertain only):     {stability['fluctuate_uncertain']:3d} ({stability['fluctuate_uncertain_pct']:.1f}%)")
    print(f"TRUE FLIP (IOF <-> STD):        {stability['true_flip']:3d} ({stability['true_flip_pct']:.1f}%)")

    # Print per-run summary table
    print("\n" + "=" * 70)
    print("PER-RUN SUMMARY")
    print("=" * 70)
    print(f"{'Run ID':<20} {'IOF%':>6} {'STD%':>6} {'UNC%':>6} {'IOF/cls':>8} {'med_dAICc':>10}")
    print("-" * 70)

    for summary in all_summaries:
        iof_pct = 100 * summary.iof_fraction
        std_pct = 100 * summary.n_standard / summary.n_events if summary.n_events > 0 else 0
        unc_pct = 100 * summary.n_uncertain / summary.n_events if summary.n_events > 0 else 0
        iof_cls = 100 * summary.iof_fraction_of_classified

        print(f"{summary.config.run_id:<20} {iof_pct:>5.1f}% {std_pct:>5.1f}% {unc_pct:>5.1f}% "
              f"{iof_cls:>7.1f}% {summary.median_delta_aicc:>10.1f}")

    # Compute aggregate statistics
    iof_fractions = [s.iof_fraction for s in all_summaries]
    iof_classified_fractions = [s.iof_fraction_of_classified for s in all_summaries]

    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)
    print(f"IOF fraction across {len(configs)} runs:")
    print(f"  Mean:   {100*np.mean(iof_fractions):.1f}%")
    print(f"  Std:    {100*np.std(iof_fractions):.1f}%")
    print(f"  Min:    {100*np.min(iof_fractions):.1f}%")
    print(f"  Max:    {100*np.max(iof_fractions):.1f}%")
    print(f"  Range:  {100*(np.max(iof_fractions) - np.min(iof_fractions)):.1f}%")
    print()
    print(f"IOF fraction of classified:")
    print(f"  Mean:   {100*np.mean(iof_classified_fractions):.1f}%")
    print(f"  Std:    {100*np.std(iof_classified_fractions):.1f}%")
    print(f"  Range:  {100*(np.max(iof_classified_fractions) - np.min(iof_classified_fractions)):.1f}%")

    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")

    # Save per-run summaries
    summaries_data = []
    for summary in all_summaries:
        summaries_data.append({
            'config': summary.config.to_dict(),
            'n_events': summary.n_events,
            'n_iof': summary.n_iof,
            'n_standard': summary.n_standard,
            'n_uncertain': summary.n_uncertain,
            'iof_fraction': summary.iof_fraction,
            'iof_fraction_of_classified': summary.iof_fraction_of_classified,
            'median_delta_aicc': summary.median_delta_aicc,
            'strong_evidence_frac': summary.strong_evidence_frac,
            'moderate_evidence_frac': summary.moderate_evidence_frac,
            'weak_evidence_frac': summary.weak_evidence_frac
        })

    with open(output_dir / 'run_summaries.json', 'w') as f:
        json.dump(summaries_data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved run summaries to {output_dir / 'run_summaries.json'}")

    # Save stability table
    with open(output_dir / 'stability_table.json', 'w') as f:
        json.dump(stability, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved stability table to {output_dir / 'stability_table.json'}")

    # Save detailed per-event results
    detailed_results = {}
    for run_id, results in all_results.items():
        detailed_results[run_id] = [
            {
                'event_id': r.event_id,
                'classification': r.classification,
                'winning_model': r.winning_model,
                'model_t_peak': r.model_t_peak,
                'delta_aicc': r.delta_aicc,
                'reason': r.reason
            }
            for r in results
        ]

    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved detailed results to {output_dir / 'detailed_results.json'}")

    print("\n" + "=" * 70)
    print("Robustness sweep complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
