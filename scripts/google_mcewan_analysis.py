#!/usr/bin/env python3
"""
IOF Forensic Analysis Script
============================

Analyzes cosmic ray recovery data from the McEwen et al. (2022) dataset to detect
signatures of the Ignorant Observer Framework (IOF).

Data source: Figshare DOI 10.6084/m9.figshare.16673851
Reference: McEwen, M., et al. (2022). Resolving catastrophic error bursts from
           cosmic rays in large arrays of superconducting qubits. Nature Physics.

Based on the Forensic Protocol:
- Compares Standard Physics (exponential recovery) vs IOF Physics (sigmoid recovery)
- Looks for "instability plateaus" in the first 5-15ms after impact
- Identifies systematic residual patterns as potential IOF signatures

NOTE: Cosmic rays cause error count to DROP (qubits reset to ground state),
then the system RECOVERS back to baseline. We analyze this upward recovery.

Author: Aernoud Dekker
Date: November 2025
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Suppress optimization warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=np.exceptions.VisibleDeprecationWarning if hasattr(np.exceptions, 'VisibleDeprecationWarning') else UserWarning)


# =============================================================================
# Configuration
# =============================================================================

# Data parameters for MAIN datasets (from README.txt)
SAMPLING_INTERVAL_US = 100  # microseconds (MAIN datasets)
SAMPLING_INTERVAL_MS = SAMPLING_INTERVAL_US / 1000  # = 0.1 milliseconds

# Analysis window
RECOVERY_WINDOW_MS = 100  # Analyze first 100ms after impact (per Forensic Protocol)
RECOVERY_WINDOW_ROWS = int(RECOVERY_WINDOW_MS / SAMPLING_INTERVAL_MS)  # = 1000 rows

# Event detection - looking for DROPS in error count
DROP_THRESHOLD_SIGMA = 5  # Detect drops > 5 sigma below baseline
MIN_DROP_DEPTH = 10  # Minimum absolute decrease in error count
MIN_EVENT_SEPARATION_MS = 500  # Minimum time between separate events

# Column indices (0-indexed, for MAIN datasets with 30 columns)
TIME_COL = 1  # Time in microseconds
ERROR_COUNT_COL = 29  # Last column = total '1's


# =============================================================================
# Model Functions for UPWARD Recovery
# =============================================================================

def exponential_recovery(t, A, tau, baseline):
    """
    Standard Physics Model A: Exponential recovery to baseline

    The standard theory predicts error rate recovers as quasiparticle
    density returns to equilibrium. Recovery is exponential.

    y(t) = baseline - A * exp(-t/tau)

    At t=0: y = baseline - A (minimum)
    As t→∞: y → baseline

    Parameters:
        t: time (ms)
        A: amplitude (baseline - minimum)
        tau: recovery time constant (ms)
        baseline: equilibrium error rate (high)
    """
    return baseline - A * np.exp(-t / tau)


def power_law_recovery(t, A, tau, baseline):
    """
    Standard Physics Model A': Power-law recovery (1/t behavior)

    If recombination dominates: x_qp ∝ 1/t, recovery follows 1/(1+t/τ)

    Parameters:
        t: time (ms)
        A: amplitude
        tau: characteristic time scale (ms)
        baseline: equilibrium error rate
    """
    return baseline - A / (1 + t / tau)


def logistic_sigmoid(t, A, k, t0, minimum):
    """
    IOF Physics Model B: Logistic sigmoid recovery

    The IOF predicts the controller must "hunt" to re-acquire the signal.
    This produces an S-curve with initial plateau before rapid recovery.

    y(t) = minimum + A / (1 + exp(-k*(t - t0)))

    At t=0: y ≈ minimum (if t0 > 0)
    At t=t0: y = minimum + A/2 (midpoint)
    As t→∞: y → minimum + A

    Parameters:
        t: time (ms)
        A: total amplitude of recovery
        k: steepness of transition (1/ms)
        t0: midpoint time of transition (ms)
        minimum: starting value (low point after cosmic ray)
    """
    return minimum + A / (1 + np.exp(-k * (t - t0)))


def delayed_recovery(t, A, tau, delay, minimum):
    """
    IOF Physics Model B': Delayed exponential recovery

    Exponential recovery that only begins after an initial plateau/delay.
    This represents the "instability plateau" where the controller is
    saturated and cannot track the basis.

    Parameters:
        t: time (ms)
        A: amplitude of recovery
        tau: recovery time constant (ms)
        delay: initial plateau duration (ms)
        minimum: starting value at minimum
    """
    baseline = minimum + A
    return np.where(
        t < delay,
        minimum,  # Plateau phase - stuck at minimum
        baseline - A * np.exp(-(t - delay) / tau)  # Recovery phase
    )


# =============================================================================
# Data Loading and Event Detection
# =============================================================================

def load_main_data(filepath):
    """
    Load a MAIN dataset CSV file.

    Returns:
        data: numpy array with all columns
        error_counts: array of total error counts
        time_ms: array of time values in milliseconds
    """
    print(f"Loading {filepath}...")
    data = np.loadtxt(filepath)

    error_counts = data[:, ERROR_COUNT_COL]
    time_us = data[:, TIME_COL]
    time_ms = time_us / 1000  # Convert to milliseconds

    print(f"  Loaded {len(data)} rows ({time_ms[-1]/1000:.1f} seconds total)")
    print(f"  Error count range: {error_counts.min():.0f} - {error_counts.max():.0f}")

    return data, error_counts, time_ms


def detect_cosmic_ray_events(error_counts, time_ms):
    """
    Detect cosmic ray impact events as DROPS in error count.

    Cosmic rays cause qubits to reset to ground state, so error count
    drops from baseline (~23) to much lower values.

    Returns:
        events: list of (min_index, min_value, baseline) tuples
    """
    # Calculate baseline statistics (using robust median)
    baseline = np.median(error_counts)
    mad = np.median(np.abs(error_counts - baseline))
    sigma = 1.4826 * mad  # Scale to standard deviation equivalent

    print(f"\nBaseline statistics:")
    print(f"  Median error count: {baseline:.2f}")
    print(f"  Robust sigma: {sigma:.2f}")

    # Find drops below threshold
    threshold = baseline - DROP_THRESHOLD_SIGMA * sigma
    threshold = min(threshold, baseline - MIN_DROP_DEPTH)

    print(f"  Detection threshold (looking for drops below): {threshold:.2f}")

    # Simple event detection - find contiguous regions below threshold
    events = []
    below_threshold = error_counts < threshold
    min_separation_rows = int(MIN_EVENT_SEPARATION_MS / SAMPLING_INTERVAL_MS)

    in_event = False
    event_start = 0
    last_event_end = -min_separation_rows

    for i in range(len(error_counts)):
        if below_threshold[i] and not in_event:
            if i - last_event_end >= min_separation_rows:
                in_event = True
                event_start = i
        elif not below_threshold[i] and in_event:
            in_event = False
            last_event_end = i
            # Find minimum within this event
            event_region = error_counts[event_start:i]
            min_idx = event_start + np.argmin(event_region)
            min_val = error_counts[min_idx]
            events.append((min_idx, min_val, baseline))

    # Handle edge case: event at end of file (in_event still True)
    if in_event:
        event_region = error_counts[event_start:]
        min_idx = event_start + np.argmin(event_region)
        min_val = error_counts[min_idx]
        events.append((min_idx, min_val, baseline))

    print(f"\nDetected {len(events)} cosmic ray event(s)")
    for i, (idx, val, bl) in enumerate(events):
        print(f"  Event {i+1}: t = {time_ms[idx]:.1f} ms, minimum = {val:.0f}, baseline = {bl:.1f}")

    return events


# =============================================================================
# Event Freezing (for robustness analysis)
# =============================================================================

FREEZE_WINDOW_MS = 150  # Extended window for robustness sweep (150ms = 1500 rows)
FREEZE_WINDOW_ROWS = int(FREEZE_WINDOW_MS / SAMPLING_INTERVAL_MS)


def freeze_events_from_datasets(data_dir, output_path):
    """
    Extract all detected events and save to frozen_events.json for robustness analysis.

    This creates a reproducible snapshot of all detected events that can be used
    by google_robustness_sweep.py to run stability analysis across different inference
    parameter configurations.

    Parameters:
        data_dir: Path to directory containing MAIN_DATASET_*.csv files
        output_path: Path for frozen_events.json output
    """
    import json

    csv_files = sorted(data_dir.glob("MAIN_DATASET_*.csv"))
    if not csv_files:
        print(f"No datasets found in {data_dir}")
        return

    print(f"\nFreezing events from {len(csv_files)} datasets...")

    all_events = []
    for csv_file in csv_files:
        dataset_name = csv_file.stem

        # Load data (same format as analyze_dataset: space-delimited, no header)
        data = np.loadtxt(csv_file)
        time_us = data[:, TIME_COL]
        error_counts = data[:, ERROR_COUNT_COL]
        time_ms = time_us / 1000.0

        # Detect events (suppress print output for batch processing)
        baseline = np.median(error_counts)
        mad = np.median(np.abs(error_counts - baseline))
        sigma = 1.4826 * mad
        threshold = baseline - DROP_THRESHOLD_SIGMA * sigma
        threshold = min(threshold, baseline - MIN_DROP_DEPTH)

        # Simple event detection
        events = []
        below_threshold = error_counts < threshold
        min_separation_rows = int(MIN_EVENT_SEPARATION_MS / SAMPLING_INTERVAL_MS)

        in_event = False
        event_start = 0
        last_event_end = -min_separation_rows

        for i in range(len(error_counts)):
            if below_threshold[i] and not in_event:
                if i - last_event_end >= min_separation_rows:
                    in_event = True
                    event_start = i
            elif not below_threshold[i] and in_event:
                in_event = False
                last_event_end = i
                event_region = error_counts[event_start:i]
                min_idx = event_start + np.argmin(event_region)
                min_val = error_counts[min_idx]
                events.append((min_idx, min_val, baseline))

        # Handle edge case: event at end of file
        if in_event:
            event_region = error_counts[event_start:]
            min_idx = event_start + np.argmin(event_region)
            min_val = error_counts[min_idx]
            events.append((min_idx, min_val, baseline))

        for min_idx, min_val, baseline in events:
            # Extract extended window (150ms)
            end_idx = min(min_idx + FREEZE_WINDOW_ROWS, len(error_counts))
            t_window = (time_ms[min_idx:end_idx] - time_ms[min_idx]).tolist()
            y_window = error_counts[min_idx:end_idx].tolist()

            # Compute canonical baseline: tail-median of last 20% of window
            # This matches LIGO pipeline convention and avoids global-median drift
            y_arr = np.array(y_window)
            tail_start = int(len(y_arr) * 0.8)
            baseline_tail = float(np.median(y_arr[tail_start:])) if tail_start < len(y_arr) else float(baseline)

            # Store true minimum within window (for amplitude calculations)
            min_value_true = float(np.min(y_arr))
            min_idx_in_window = int(np.argmin(y_arr))

            # Corrected event index: global position of true minimum
            # (event_index + min_idx_in_window gives the exact array index)
            corrected_event_index = min_idx + min_idx_in_window

            all_events.append({
                'dataset': dataset_name,
                'event_index': int(min_idx),
                'corrected_event_index': int(corrected_event_index),  # Global index of true min
                'event_time_ms': float(time_ms[min_idx]),
                'corrected_event_time_ms': float(time_ms[corrected_event_index]),  # Time at true min
                'event_value': float(min_val),
                'baseline_global': float(baseline),  # Legacy: global dataset median
                'baseline': float(baseline_tail),     # Canonical: tail-median of window
                'min_value_true': min_value_true,     # True minimum in window
                'min_idx_in_window': min_idx_in_window,
                't_window': t_window,
                'y_window': y_window
            })

    # Save frozen events
    frozen_data = {
        'detection_params': {
            'threshold_sigma': float(DROP_THRESHOLD_SIGMA),
            'min_drop_depth': float(MIN_DROP_DEPTH),
            'min_separation_ms': float(MIN_EVENT_SEPARATION_MS),
            'detection_direction': 'drop'
        },
        'n_events': len(all_events),
        'events': all_events
    }

    with open(output_path, 'w') as f:
        json.dump(frozen_data, f)

    print(f"Frozen {len(all_events)} events to {output_path}")
    return all_events


# =============================================================================
# Model Fitting
# =============================================================================

def extract_recovery_window(error_counts, time_ms, min_idx):
    """
    Extract the recovery window starting from minimum (t=0) to t=100ms.

    Returns:
        t_recovery: time array (starting from 0)
        y_recovery: error count array
    """
    end_idx = min(min_idx + RECOVERY_WINDOW_ROWS, len(error_counts))

    t_recovery = time_ms[min_idx:end_idx] - time_ms[min_idx]
    y_recovery = error_counts[min_idx:end_idx]

    return t_recovery, y_recovery


def fit_models(t, y, baseline_estimate):
    """
    Fit both standard and IOF models to the recovery data.

    Note: We're fitting UPWARD recovery from minimum toward baseline.

    Returns:
        results: dict with fit parameters, residuals, and statistics
    """
    results = {}
    min_value = y[0]
    amplitude = baseline_estimate - min_value  # Positive: how much to recover

    # GUARDRAIL: baseline must exceed minimum (catches noise/truncation pathology)
    if baseline_estimate <= min_value + 0.5:
        print(f"Bad baseline: baseline={baseline_estimate:.1f} <= min={min_value:.1f}, skipping")
        return results

    # Skip if no significant drop
    if amplitude < 5:
        print("Insufficient amplitude for fitting")
        return results

    ss_tot = np.sum((y - np.mean(y))**2)
    if ss_tot == 0:
        print("No variance in data")
        return results

    # Compute data-derived lower bound for fitted baseline
    # Prevents optimizer from choosing baseline below tail (unphysical "recovery")
    tail_median = np.median(y[int(0.8 * len(y)):]) if len(y) >= 5 else np.median(y)
    baseline_lower = max(min_value + 1, tail_median * 0.8)  # Must exceed minimum, near tail

    # --- Model A: Exponential Recovery ---
    try:
        p0_exp = [amplitude, 10.0, baseline_estimate]
        # Dynamic bounds with data-derived baseline floor (blocks "baseline teleportation")
        bounds_exp = ([0, 0.1, baseline_lower], [amplitude * 2, 200, baseline_estimate * 2])

        popt_exp, _ = curve_fit(
            exponential_recovery, t, y,
            p0=p0_exp, bounds=bounds_exp, maxfev=10000
        )

        y_fit_exp = exponential_recovery(t, *popt_exp)
        residuals_exp = y - y_fit_exp
        ss_res_exp = np.sum(residuals_exp**2)
        r2_exp = 1 - (ss_res_exp / ss_tot)

        n = len(y)
        k = 3
        aic_exp = n * np.log(ss_res_exp / n) + 2 * k
        # AICc correction for small samples
        if n - k - 1 > 0:
            aicc_exp = aic_exp + (2 * k * (k + 1)) / (n - k - 1)
        else:
            aicc_exp = aic_exp

        results['exponential'] = {
            'params': {'A': popt_exp[0], 'tau': popt_exp[1], 'baseline': popt_exp[2]},
            'fit': y_fit_exp,
            'residuals': residuals_exp,
            'ss_res': ss_res_exp,
            'r2': r2_exp,
            'aicc': aicc_exp
        }
        print(f"\nExponential recovery: A={popt_exp[0]:.2f}, tau={popt_exp[1]:.2f}ms, R²={r2_exp:.4f}")

    except Exception as e:
        print(f"Exponential fit failed: {e}")

    # --- Model A': Power Law Recovery ---
    try:
        p0_pow = [amplitude, 5.0, baseline_estimate]
        # Dynamic bounds with data-derived baseline floor (blocks "baseline teleportation")
        bounds_pow = ([0, 0.1, baseline_lower], [amplitude * 2, 200, baseline_estimate * 2])

        popt_pow, _ = curve_fit(
            power_law_recovery, t, y,
            p0=p0_pow, bounds=bounds_pow, maxfev=10000
        )

        y_fit_pow = power_law_recovery(t, *popt_pow)
        residuals_pow = y - y_fit_pow
        ss_res_pow = np.sum(residuals_pow**2)
        r2_pow = 1 - (ss_res_pow / ss_tot)

        n = len(y)
        k = 3
        aic_pow = n * np.log(ss_res_pow / n) + 2 * k
        # AICc correction
        if n - k - 1 > 0:
            aicc_pow = aic_pow + (2 * k * (k + 1)) / (n - k - 1)
        else:
            aicc_pow = aic_pow

        results['power_law'] = {
            'params': {'A': popt_pow[0], 'tau': popt_pow[1], 'baseline': popt_pow[2]},
            'fit': y_fit_pow,
            'residuals': residuals_pow,
            'ss_res': ss_res_pow,
            'r2': r2_pow,
            'aicc': aicc_pow
        }
        print(f"Power law recovery: A={popt_pow[0]:.2f}, tau={popt_pow[1]:.2f}ms, R²={r2_pow:.4f}")

    except Exception as e:
        print(f"Power law fit failed: {e}")

    # --- Model B: Logistic Sigmoid (IOF) ---
    try:
        # Sigmoid: starts at minimum, rises to minimum + A
        p0_sig = [amplitude, 0.2, 15.0, min_value]  # midpoint ~15ms
        # Upper bound for minimum tied to baseline, not magic constant
        bounds_sig = ([0, 0.01, 1, 0], [amplitude * 2, 2, 80, baseline_estimate])

        popt_sig, _ = curve_fit(
            logistic_sigmoid, t, y,
            p0=p0_sig, bounds=bounds_sig, maxfev=10000
        )

        y_fit_sig = logistic_sigmoid(t, *popt_sig)
        residuals_sig = y - y_fit_sig
        ss_res_sig = np.sum(residuals_sig**2)
        r2_sig = 1 - (ss_res_sig / ss_tot)

        n = len(y)
        k = 4
        aic_sig = n * np.log(ss_res_sig / n) + 2 * k
        # AICc correction
        if n - k - 1 > 0:
            aicc_sig = aic_sig + (2 * k * (k + 1)) / (n - k - 1)
        else:
            aicc_sig = aic_sig

        results['sigmoid'] = {
            'params': {'A': popt_sig[0], 'k': popt_sig[1], 't0': popt_sig[2], 'minimum': popt_sig[3]},
            'fit': y_fit_sig,
            'residuals': residuals_sig,
            'ss_res': ss_res_sig,
            'r2': r2_sig,
            'aicc': aicc_sig
        }
        print(f"Sigmoid (IOF): A={popt_sig[0]:.2f}, k={popt_sig[1]:.3f}, t0={popt_sig[2]:.2f}ms, R²={r2_sig:.4f}")

    except Exception as e:
        print(f"Sigmoid fit failed: {e}")

    # --- Model B': Delayed Recovery (IOF) ---
    try:
        p0_del = [amplitude, 15.0, 5.0, min_value]  # 5ms delay
        # Upper bound for minimum tied to baseline, not magic constant
        bounds_del = ([0, 0.1, 0, 0], [amplitude * 2, 200, 50, baseline_estimate])

        popt_del, _ = curve_fit(
            delayed_recovery, t, y,
            p0=p0_del, bounds=bounds_del, maxfev=10000
        )

        y_fit_del = delayed_recovery(t, *popt_del)
        residuals_del = y - y_fit_del
        ss_res_del = np.sum(residuals_del**2)
        r2_del = 1 - (ss_res_del / ss_tot)

        n = len(y)
        k = 4
        aic_del = n * np.log(ss_res_del / n) + 2 * k
        # AICc correction
        if n - k - 1 > 0:
            aicc_del = aic_del + (2 * k * (k + 1)) / (n - k - 1)
        else:
            aicc_del = aic_del

        results['delayed'] = {
            'params': {'A': popt_del[0], 'tau': popt_del[1], 'delay': popt_del[2], 'minimum': popt_del[3]},
            'fit': y_fit_del,
            'residuals': residuals_del,
            'ss_res': ss_res_del,
            'r2': r2_del,
            'aicc': aicc_del
        }
        print(f"Delayed recovery (IOF): A={popt_del[0]:.2f}, tau={popt_del[1]:.2f}ms, delay={popt_del[2]:.2f}ms, R²={r2_del:.4f}")

    except Exception as e:
        print(f"Delayed recovery fit failed: {e}")

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_recovery_analysis(t, y, results, event_num, output_dir, dataset_name):
    """
    Generate comprehensive plots for recovery analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name} - Event {event_num}: Cosmic Ray Recovery Analysis', fontsize=12)

    # --- Plot 1: Raw data with all fits ---
    ax1 = axes[0, 0]
    ax1.scatter(t, y, s=2, alpha=0.4, color='gray', label='Data')

    colors = {'exponential': 'blue', 'power_law': 'cyan', 'sigmoid': 'red', 'delayed': 'orange'}
    labels = {'exponential': 'Exponential (Standard)', 'power_law': 'Power Law (Standard)',
              'sigmoid': 'Sigmoid (IOF)', 'delayed': 'Delayed (IOF)'}

    for model_name, color in colors.items():
        if results.get(model_name):
            ax1.plot(t, results[model_name]['fit'], color=color,
                    linewidth=2, label=f"{labels[model_name]} (R²={results[model_name]['r2']:.4f})")

    ax1.set_xlabel('Time since minimum (ms)')
    ax1.set_ylabel('Error count')
    ax1.set_title('Recovery Curve with Model Fits')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Early time zoom (first 20ms) - IOF window ---
    ax2 = axes[0, 1]
    mask_early = t <= 20
    if np.any(mask_early):
        ax2.scatter(t[mask_early], y[mask_early], s=4, alpha=0.5, color='gray', label='Data')

        for model_name, color in colors.items():
            if results.get(model_name):
                fit = results[model_name]['fit']
                ax2.plot(t[mask_early], fit[mask_early], color=color, linewidth=2, label=labels[model_name])

        ax2.axvspan(5, 15, alpha=0.15, color='red', label='IOF plateau window (5-15ms)')
        ax2.set_xlabel('Time since minimum (ms)')
        ax2.set_ylabel('Error count')
        ax2.set_title('Early Recovery (First 20ms) - IOF Signature Window')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

    # --- Plot 3: Residuals comparison ---
    ax3 = axes[1, 0]

    for model_name, color in colors.items():
        if results.get(model_name):
            residuals = results[model_name]['residuals']
            # Smooth residuals for visibility
            # NOTE: 'valid' convolution - correct time axis to window centers
            window = min(20, max(1, len(residuals) // 20))
            if window % 2 == 0:
                window += 1  # Force odd to avoid half-sample bias
            if window > 1:
                smoothed = np.convolve(residuals, np.ones(window)/window, mode='valid')
                offset = (window - 1) // 2
                t_smooth = t[offset:offset + len(smoothed)]
                ax3.plot(t_smooth, smoothed, color=color, linewidth=1.5,
                        label=f"{model_name}", alpha=0.8)

    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax3.axvspan(0, 20, alpha=0.1, color='yellow', label='Early recovery region')
    ax3.set_xlabel('Time since minimum (ms)')
    ax3.set_ylabel('Residual (Data - Fit)')
    ax3.set_title('Residual Analysis (Smoothed) - Look for systematic "hump"')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Model comparison summary ---
    ax4 = axes[1, 1]

    model_names = []
    aic_values = []
    r2_values = []

    for model_name in ['exponential', 'power_law', 'sigmoid', 'delayed']:
        if results.get(model_name):
            model_names.append(model_name)
            aic_values.append(results[model_name]['aicc'])
            r2_values.append(results[model_name]['r2'])

    if model_names:
        x_pos = np.arange(len(model_names))
        width = 0.35

        ax4_r2 = ax4
        ax4_aic = ax4.twinx()

        ax4_r2.bar(x_pos - width/2, r2_values, width, label='R²', color='steelblue', alpha=0.7)
        ax4_r2.set_ylabel('R² (higher is better)', color='steelblue')
        ax4_r2.tick_params(axis='y', labelcolor='steelblue')
        ax4_r2.set_ylim(0, 1)

        # Normalize AICc for visualization (relative to min)
        aicc_min = min(aic_values)
        aicc_rel = [a - aicc_min for a in aic_values]
        ax4_aic.bar(x_pos + width/2, aicc_rel, width, label='ΔAICc', color='coral', alpha=0.7)
        ax4_aic.set_ylabel('ΔAICc (lower is better)', color='coral')
        ax4_aic.tick_params(axis='y', labelcolor='coral')

        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([labels.get(m, m) for m in model_names], rotation=15, ha='right', fontsize=9)
        ax4.set_title('Model Comparison: R² and AICc')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f'{dataset_name}_event_{event_num}_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    plt.close()


def plot_derivative_analysis(t, y, results, event_num, output_dir, dataset_name):
    """
    Plot the derivative (rate of change) to identify slope changes.

    For UPWARD recovery:
    - Standard physics: Steepest upward slope at t≈0
    - IOF signature: Steepest upward slope DELAYED (t > 5ms)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate numerical derivative (smoothed)
    # NOTE: 'valid' convolution drops (window-1) samples; output corresponds to window centers
    window = max(5, len(y) // 50)
    if window % 2 == 0:
        window += 1  # Force odd to avoid half-sample bias
    y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
    # Correct time axis: smoothed output corresponds to t[offset:offset+len], not t[:len]
    offset = (window - 1) // 2
    t_smooth = t[offset:offset + len(y_smooth)]

    if len(t_smooth) < 2:
        print("Insufficient data for derivative analysis")
        plt.close()
        return 0.0

    dy = np.gradient(y_smooth, t_smooth)

    # --- Plot 1: Derivative of data ---
    ax1 = axes[0]
    ax1.plot(t_smooth, dy, color='black', linewidth=1, alpha=0.7, label='d(error)/dt')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.axvspan(0, 20, alpha=0.1, color='red', label='IOF signature window')

    ax1.set_xlabel('Time since minimum (ms)')
    ax1.set_ylabel('Rate of change (errors/ms)')
    ax1.set_title(f'{dataset_name} Event {event_num}: Recovery Rate (Derivative)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Early derivative zoom ---
    ax2 = axes[1]
    mask_early = t_smooth <= 30
    early_t = t_smooth[mask_early]
    early_dy = dy[mask_early]

    if len(early_t) > 0:
        ax2.plot(early_t, early_dy, color='black', linewidth=1.5, label='Data derivative')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        # Find MAXIMUM derivative (steepest UPWARD slope for recovery)
        max_idx = np.argmax(early_dy)
        max_t = early_t[max_idx]
        max_dy = early_dy[max_idx]
        ax2.axvline(x=max_t, color='green', linestyle=':', alpha=0.7)
        ax2.annotate(f'Steepest: {max_t:.1f}ms', xy=(max_t, max_dy),
                    xytext=(max_t+3, max_dy*0.8), fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

        ax2.set_xlabel('Time since minimum (ms)')
        ax2.set_ylabel('Rate of change (errors/ms)')
        ax2.set_title('Early Recovery Rate - IOF Test')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add interpretation text
        interpretation = (
            "STANDARD PHYSICS: Steepest UPWARD slope at t≈0\n"
            "IOF SIGNATURE: Steepest slope DELAYED (t > 5ms)"
        )
        ax2.text(0.98, 0.02, interpretation, transform=ax2.transAxes,
                 fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        steepest_time = max_t
    else:
        steepest_time = 0.0

    plt.tight_layout()

    output_path = output_dir / f'{dataset_name}_event_{event_num}_derivative.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved derivative plot to {output_path}")
    plt.close()

    return steepest_time


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_dataset(filepath, output_dir):
    """
    Run complete IOF forensic analysis on a single MAIN dataset.
    """
    dataset_name = filepath.stem
    print("\n" + "=" * 70)
    print(f"IOF FORENSIC ANALYSIS: {dataset_name}")
    print("=" * 70)

    # Load data
    data, error_counts, time_ms = load_main_data(filepath)

    # Detect events
    events = detect_cosmic_ray_events(error_counts, time_ms)

    if not events:
        print("\nNo cosmic ray events detected in this dataset.")
        return None

    # Analyze each event
    summary = []

    for i, (min_idx, min_val, baseline_global) in enumerate(events, 1):
        print(f"\n{'='*50}")
        print(f"ANALYZING EVENT {i}")
        print(f"{'='*50}")

        # Extract raw 150ms window (before re-anchoring)
        end_idx_150 = min(min_idx + FREEZE_WINDOW_ROWS, len(error_counts))
        y_150_raw = error_counts[min_idx:end_idx_150]
        t_150_raw = time_ms[min_idx:end_idx_150] - time_ms[min_idx]

        # Compute canonical baseline from RAW window BEFORE re-anchoring
        # This makes baseline bit-identical with freeze_events_from_datasets
        tail_start_raw = int(len(y_150_raw) * 0.8)
        baseline = float(np.median(y_150_raw[tail_start_raw:])) if tail_start_raw < len(y_150_raw) else baseline_global

        # Re-anchor to local argmin within 150ms window (for fitting)
        j = int(np.argmin(y_150_raw))
        if j > 0:
            t_150 = t_150_raw[j:] - t_150_raw[j]
            y_150 = y_150_raw[j:]
            corrected_min_idx = min_idx + j
        else:
            t_150 = t_150_raw
            y_150 = y_150_raw
            corrected_min_idx = min_idx

        min_val = y_150[0]  # Now guaranteed to be the true minimum

        # Truncate to 100ms for fitting (but baseline is from 150ms window)
        mask_100 = t_150 <= RECOVERY_WINDOW_MS
        t = t_150[mask_100]
        y = y_150[mask_100]

        print(f"\nRecovery window: {len(t)} points (100ms fit), baseline from 150ms")
        print(f"Starting value (local min): {min_val:.1f}, Final value: {y[-1]:.1f}")
        print(f"Baseline (150ms tail-median): {baseline:.1f} (global was {baseline_global:.1f})")

        # Fit models
        results = fit_models(t, y, baseline)

        if not results:
            print("Skipping event - insufficient data for fitting")
            continue

        # Generate plots
        plot_recovery_analysis(t, y, results, i, output_dir, dataset_name)
        steepest_time = plot_derivative_analysis(t, y, results, i, output_dir, dataset_name)

        # Normalize data for aggregation (0=min, 1=baseline)
        # Re-interpolate to a common time grid for averaging later
        common_t = np.linspace(0, 100, 1000)  # 0 to 100ms, 1000 points

        # Normalize y: (y - min) / (baseline - min)
        y_norm_raw = (y - min_val) / (baseline - min_val)

        # Interpolate to common grid
        y_interp = np.interp(common_t, t, y_norm_raw)

        # Compute curvature index (quadratic fit over first 20ms)
        curvature_b = None
        try:
            mask = t <= 20.0
            if np.sum(mask) >= 10:
                t_curv = t[mask]
                y_curv = y_norm_raw[mask]  # Use mask directly, not [:len] slice
                coeffs = np.polyfit(t_curv, y_curv, 2)
                curvature_b = coeffs[0]
        except:
            pass

        # Determine stability classification based on steepest ascent time
        # >5ms = delayed geometry, <=5ms = fast geometry
        # NOTE: This is a simple derivative-based heuristic for exploratory analysis.
        # The manuscript uses the proper AICc model-based 15-run stability classification
        # from google_robustness_sweep.py which gives different (correct) counts:
        # n=196 with 13 Stable Delayed, 152 Stable Fast, 31 Flip.
        if steepest_time > 5:
            stability = 'Stable Delayed'
        else:
            stability = 'Stable Fast'

        # Compile summary (use corrected_min_idx for accurate timestamp)
        # Include dataset + event_index to match frozen event IDs for stability lookup
        event_summary = {
            'dataset': dataset_name,           # stem, matches frozen evt['dataset']
            'event_index': int(min_idx),       # matches frozen evt['event_index']
            'event_num': i,
            'min_time_ms': time_ms[corrected_min_idx],
            'min_value': min_val,
            'baseline': baseline,
            'steepest_ascent_ms': steepest_time,
            'stability': stability,
            'curvature_b': curvature_b,
            'results': results,
            'curve_t': common_t,
            'curve_y': y_interp
        }
        summary.append(event_summary)

        # IOF signature check
        print(f"\n--- IOF SIGNATURE CHECK ---")
        if steepest_time > 5:
            print(f"  [DELAYED] Steepest recovery at {steepest_time:.1f}ms (>5ms)")
        else:
            print(f"  [FAST] Steepest recovery at {steepest_time:.1f}ms (<=5ms)")

        # Compare AIC
        if results.get('exponential') and results.get('sigmoid'):
            aicc_diff = results['exponential']['aicc'] - results['sigmoid']['aicc']
            if aicc_diff > 10:
                print(f"  Sigmoid model preferred (ΔAICc = {aicc_diff:.1f})")
            elif aicc_diff < -10:
                print(f"  Exponential model preferred (ΔAICc = {aicc_diff:.1f})")
            else:
                print(f"  Models comparable (ΔAICc = {aicc_diff:.1f})")

    return summary


def generate_summary_report(all_summaries, output_dir):
    """
    Generate a text summary report of all analyses.
    """
    report_path = output_dir / 'analysis_summary.txt'

    total_events = 0
    iof_candidates = 0

    with open(report_path, 'w') as f:
        f.write("IOF FORENSIC ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write("McEwen Cosmic Ray Dataset Analysis\n")
        f.write("=" * 70 + "\n\n")

        for dataset_name, summaries in all_summaries.items():
            f.write(f"\nDATASET: {dataset_name}\n")
            f.write("-" * 50 + "\n")

            if summaries is None:
                f.write("No events detected\n")
                continue

            for s in summaries:
                total_events += 1
                f.write(f"\nEvent {s['event_num']}:\n")
                f.write(f"  Time of minimum: {s['min_time_ms']:.1f} ms\n")
                f.write(f"  Minimum error count: {s['min_value']:.0f}\n")
                f.write(f"  Baseline: {s['baseline']:.1f}\n")
                f.write(f"  Steepest recovery: {s['steepest_ascent_ms']:.1f} ms\n")

                results = s['results']
                f.write(f"\n  Model fits:\n")
                for model in ['exponential', 'power_law', 'sigmoid', 'delayed']:
                    if results.get(model):
                        f.write(f"    {model}: R²={results[model]['r2']:.4f}, AICc={results[model]['aicc']:.1f}\n")

                f.write(f"\n  IOF Assessment:\n")
                if s['steepest_ascent_ms'] > 5:
                    iof_candidates += 1
                    f.write(f"    [DELAYED] Delayed steepest recovery (>5ms)\n")
                else:
                    f.write(f"    [FAST] Immediate steepest recovery (<=5ms)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write(f"Total events analyzed: {total_events}\n")
        f.write(f"Potential IOF signatures: {iof_candidates}\n")
        f.write("=" * 70 + "\n")

    print(f"\nSummary report saved to {report_path}")
    return total_events, iof_candidates


def generate_curvature_plot(all_summaries, output_dir):
    """
    Generate curvature index plot for Google/McEwan data.
    Shows that curvature does NOT discriminate (unlike LIGO).

    Uses model-based stability labels from stability_table.json (from the 15-run
    robustness sweep) instead of the derivative-based heuristic.
    """
    from scipy.stats import mannwhitneyu
    import json

    print("\nGenerating Curvature Index Plot...")

    # Try to load model-based stability labels from robustness sweep
    script_dir = Path(__file__).parent
    stability_path = script_dir / 'output' / 'robustness_sweep' / 'stability_table.json'

    stability_lookup = {}
    if stability_path.exists():
        print(f"  Loading stability labels from {stability_path}")
        with open(stability_path, 'r') as f:
            stability_data = json.load(f)
        # Map sweep labels to plot labels
        label_map = {
            'stable_std': 'Stable Fast',
            'stable_iof': 'Stable Delayed',
            'flip': 'Flip',
            'uncertain_fluctuate': 'Flip'  # Group with Flip (parameter-sensitive)
        }
        for event_id, details in stability_data.get('event_details', {}).items():
            sweep_label = details.get('stability', '')
            stability_lookup[event_id] = label_map.get(sweep_label, 'unknown')
        print(f"  Loaded {len(stability_lookup)} event labels")
    else:
        print("  Warning: stability_table.json not found, falling back to derivative heuristic")

    # Collect curvature values by stability class
    curvature_by_class = {'Stable Fast': [], 'Flip': [], 'Stable Delayed': []}

    # Visibility check: count lookup hits/misses
    lookup_hits = 0
    lookup_misses = 0
    total_events = 0

    for dataset_name, summaries in all_summaries.items():
        if summaries is None:
            continue
        for event in summaries:
            total_events += 1
            # Construct event_id to look up model-based stability
            # Must match sweep format: "{dataset}_{event_index}" (stem, original min_idx)
            event_id = f"{event['dataset']}_{event['event_index']}"

            if stability_lookup:
                if event_id in stability_lookup:
                    stability = stability_lookup[event_id]
                    lookup_hits += 1
                else:
                    stability = 'unknown'
                    lookup_misses += 1
            else:
                # Fallback to derivative-based heuristic
                stability = event.get('stability', 'unknown')

            curvature = event.get('curvature_b')
            if curvature is not None and stability in curvature_by_class:
                curvature_by_class[stability].append(curvature * 1000)  # Scale to 10^-3

    # Report lookup statistics
    if stability_lookup:
        print(f"  Stability lookup: {lookup_hits}/{total_events} matched, {lookup_misses} unknown")
        if lookup_misses > 0:
            print(f"  WARNING: {lookup_misses} events fell back to 'unknown' - check event_id dialect")
        # HARD FAIL for publication runs: dialect mismatch = silent data corruption
        assert lookup_misses == 0, \
            f"Event ID dialect mismatch: {lookup_misses}/{total_events} events not found in stability_table"

    # Check we have data
    stable_fast = np.array(curvature_by_class['Stable Fast'])
    stable_delayed = np.array(curvature_by_class['Stable Delayed'])
    flip = np.array(curvature_by_class['Flip'])

    if len(stable_fast) == 0 or len(stable_delayed) == 0:
        print("Warning: Not enough curvature data for plot")
        return

    # Mann-Whitney test
    stat, pval = mannwhitneyu(stable_delayed, stable_fast, alternative='two-sided')

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Boxplot
    ax = axes[0]
    data_to_plot = [stable_fast, flip, stable_delayed]
    labels = [f'Stable Fast\n(n={len(stable_fast)})',
              f'Flip\n(n={len(flip)})',
              f'Stable Delayed\n(n={len(stable_delayed)})']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Curvature index $b$ (×10⁻³)', fontsize=11)
    ax.set_title(f'Curvature by Stability Class\n(Mann-Whitney p = {pval:.2f})', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Right: Density plot
    ax = axes[1]
    from scipy.stats import gaussian_kde

    for data, label, color in [(stable_fast, 'Stable Fast', '#2ecc71'),
                                (stable_delayed, 'Stable Delayed', '#e74c3c')]:
        if len(data) > 5:
            kde = gaussian_kde(data, bw_method=0.3)
            x_range = np.linspace(min(data) - 1, max(data) + 1, 200)
            ax.fill_between(x_range, kde(x_range), alpha=0.4, color=color, label=label)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Curvature index $b$ (×10⁻³)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Curvature Distributions\n(Substantial Overlap)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "mcewan_curvature_index.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved curvature plot to {out_path}")
    print(f"  Mann-Whitney p = {pval:.4f} (curvature does NOT discriminate)")
    plt.close()


def main():
    """
    Main entry point for IOF forensic analysis.

    Options:
        --freeze: Create frozen_events.json for robustness analysis
        --freeze-only: Create frozen_events.json and exit (skip full analysis)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='IOF Forensic Analysis of Google/McEwan cosmic ray data'
    )
    parser.add_argument('--freeze', action='store_true',
                        help='Create frozen_events.json for robustness analysis')
    parser.add_argument('--freeze-only', action='store_true',
                        help='Only create frozen_events.json (skip full analysis)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to directory containing MAIN_DATASET_*.csv files')
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    # Data path: use --data_dir if provided, otherwise default location
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = script_dir.parent.parent.parent / "Datasets" / "Google_Quantum_AI" / "data" / "16673851" / "MAIN_DATASETS"
    output_dir = script_dir / "output"
    figures_dir = script_dir.parent / "figures" / "google"

    # Create output directories
    output_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Freeze events if requested
    if args.freeze or args.freeze_only:
        frozen_path = output_dir / "frozen_events.json"
        freeze_events_from_datasets(data_dir, frozen_path)
        if args.freeze_only:
            print("\nFreeze complete. Use google_robustness_sweep.py for stability analysis.")
            return

    # Find all MAIN datasets
    csv_files = sorted(data_dir.glob("MAIN_DATASET_*.csv"))

    if not csv_files:
        print(f"ERROR: No MAIN_DATASET_*.csv files found in {data_dir}")
        return

    print(f"\nFound {len(csv_files)} dataset(s)")

    # Analyze each dataset
    all_summaries = {}

    for csv_file in csv_files:
        try:
            summary = analyze_dataset(csv_file, output_dir)
            all_summaries[csv_file.stem] = summary  # stem (no .csv) to match frozen event IDs
        except Exception as e:
            print(f"Error analyzing {csv_file.name}: {e}")
            all_summaries[csv_file.stem] = None

    # Generate summary report
    total, candidates = generate_summary_report(all_summaries, output_dir)

    # Generate the curvature index plot (save to figures/google/)
    # Uses model-based stability labels from stability_table.json if available
    # (falls back to derivative heuristic if sweep hasn't been run yet)
    generate_curvature_plot(all_summaries, figures_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Total events: {total}")
    print(f"Potential IOF signatures: {candidates}")
    print("=" * 70)


if __name__ == "__main__":
    main()
