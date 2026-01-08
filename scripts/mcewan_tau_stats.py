#!/usr/bin/env python3
"""
Compute McEwen Tau Statistics
=============================

Fits exponential recovery model to frozen events and computes tau statistics.
Outputs mcewan_tau_stats.json for generate_macros.py to read.

Model: y(t) = baseline - A * exp(-t/tau)

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

SCRIPT_DIR = Path(__file__).parent
FROZEN_FILE = SCRIPT_DIR / "output" / "frozen_events.json"
OUTPUT_FILE = SCRIPT_DIR / "output" / "robustness_sweep" / "mcewan_tau_stats.json"


def exponential_recovery(t, A, tau, baseline):
    """y(t) = baseline - A * exp(-t/tau)"""
    return baseline - A * np.exp(-t / tau)


def fit_event(event):
    """Fit exponential to a single event, return tau or None."""
    t = np.array(event['t_window'])
    y = np.array(event['y_window'])
    baseline = event['baseline']
    min_val = event['min_value_true']

    # Initial guess: A = baseline - min_val, tau = 10 ms
    A_init = baseline - min_val
    if A_init <= 0:
        return None

    try:
        popt, _ = curve_fit(
            exponential_recovery,
            t, y,
            p0=[A_init, 10.0, baseline],
            bounds=([0, 0.5, baseline - 5], [A_init * 2, 180, baseline + 5]),
            maxfev=1000
        )
        tau = popt[1]
        # Sanity check: tau should be reasonable
        if 0.5 <= tau <= 180:
            return tau
    except (RuntimeError, ValueError):
        pass

    return None


def main():
    print("=" * 60)
    print("Computing McEwen Tau Statistics")
    print("=" * 60)

    # Load frozen events
    with open(FROZEN_FILE) as f:
        data = json.load(f)

    events = data['events']
    print(f"Loaded {len(events)} frozen events")

    # Fit each event and collect tau values
    taus = []
    failed = 0

    for event in events:
        tau = fit_event(event)
        if tau is not None:
            taus.append(tau)
        else:
            failed += 1

    print(f"Successful fits: {len(taus)}")
    print(f"Failed fits: {failed}")

    if len(taus) < 10:
        print("WARNING: Too few successful fits for reliable statistics")
        return

    taus = np.array(taus)

    # Compute statistics
    stats = {
        'model': 'exponential: y(t) = baseline - A * exp(-t/tau)',
        'subset': 'all frozen events with successful fit',
        'tau_bounds_ms': [0.5, 180],
        'n_events': len(taus),
        'n_failed': failed,
        'tau_median_ms': round(float(np.median(taus)), 1),
        'tau_iqr_lo_ms': round(float(np.percentile(taus, 25)), 1),
        'tau_iqr_hi_ms': round(float(np.percentile(taus, 75)), 1),
        'tau_mean_ms': round(float(np.mean(taus)), 1),
        'tau_std_ms': round(float(np.std(taus)), 1),
    }

    print(f"\nTau Statistics:")
    print(f"  Median: {stats['tau_median_ms']} ms")
    print(f"  IQR: [{stats['tau_iqr_lo_ms']}, {stats['tau_iqr_hi_ms']}] ms")
    print(f"  Mean: {stats['tau_mean_ms']} ± {stats['tau_std_ms']} ms")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
