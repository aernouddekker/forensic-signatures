#!/usr/bin/env python3
"""
Chinese 63-Qubit Cosmic Ray Analysis
=====================================

Analyzes cosmic ray recovery dynamics from Li et al. (2025) 63-qubit
superconducting processor data to identify capacity-wins regime signatures.

Data source: Figshare DOI 10.6084/m9.figshare.28815434
Download and extract to a local directory, then set DATA_DIR below or use --data_dir.

This script processes:
- SI_Fig8a: Charge-parity jump probability (P_MQSCPJ) - 5.6 μs resolution
- SI_Fig12a: Bit flip probability (P_MQSBF) - 56 ms resolution

Expected data structure:
    <data_dir>/
        SI_Fig8/SI_Fig8a.npz
        SI_Fig12/SI_Fig12a.npz

Output: figures/chinese/chinese_cosmic_ray_events.png
        figures/chinese/chinese_bitflip_events.png

Reference:
Li, Y., et al. (2025). Cosmic-ray-induced correlated errors in superconducting
qubit array. Nature Communications, 16, 4677.

Author: Aernoud Dekker
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import os

# Default paths - override with --data_dir argument
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "chinese_cosmic_ray"  # Default: scripts/data/chinese_cosmic_ray/
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "chinese"
FROZEN_DIR = SCRIPT_DIR / "output" / "chinese_frozen_events"


# -----------------------------------------------------------------------------
# Robust estimation helpers
# -----------------------------------------------------------------------------

def robust_baseline(data, quantile_range=(0.1, 0.5)):
    """
    Estimate baseline from lower quantiles (avoiding burst peaks).
    Uses median of data between quantile_range[0] and quantile_range[1].
    """
    lo, hi = np.quantile(data, quantile_range)
    mask = (data >= lo) & (data <= hi)
    if mask.sum() > 0:
        return float(np.median(data[mask]))
    return float(np.median(data))


def robust_threshold(data, baseline, n_sigma=5.0):
    """
    Estimate threshold as baseline + n_sigma * MAD-based sigma.
    MAD (median absolute deviation) is robust to outliers.
    """
    mad = np.median(np.abs(data - baseline))
    sigma_est = 1.4826 * mad  # MAD to sigma conversion for normal distribution
    threshold = baseline + n_sigma * sigma_est
    return float(threshold), float(sigma_est)


def ms_to_samples(ms, dt_seconds):
    """Convert milliseconds to sample count given dt in seconds."""
    return int(round(ms / 1000.0 / dt_seconds))


def samples_to_ms(samples, dt_seconds):
    """Convert sample count to milliseconds given dt in seconds."""
    return samples * dt_seconds * 1000.0


def analyze_cosmic_ray_events(freeze=False):
    """
    Analyze charge-parity jump probability data (SI_Fig8a).
    High-resolution (5.6 μs) cosmic ray burst detection.

    Args:
        freeze: If True, save frozen events to JSON for robustness analysis.
    """
    print("=== Charge-Parity Jump Analysis (5.6 μs resolution) ===")

    data_path = DATA_DIR / "SI_Fig8" / "SI_Fig8a.npz"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Download from Figshare DOI 10.6084/m9.figshare.28815434")
        return

    # Load data
    data = np.load(data_path, allow_pickle=True)
    time = data['time']
    P_MQSCPJ = data['P_MQSCPJ']

    # Measure dt from data (don't assume 5.6 μs)
    dt = time[1] - time[0]  # seconds
    dt_us = dt * 1e6  # microseconds

    print(f"Duration: {time.max():.1f} s")
    print(f"Sampling interval: {dt_us:.2f} μs (measured from data)")
    print(f"Total points: {len(time):,}")

    # Tripwire: check dt consistency
    dt_diff = np.diff(time)
    dt_jitter = np.std(dt_diff) / np.mean(dt_diff)
    if dt_jitter > 0.01:
        print(f"WARNING: dt jitter = {dt_jitter*100:.2f}% (non-uniform sampling)")
    assert dt_jitter < 0.1, f"dt jitter too large: {dt_jitter*100:.1f}%"

    # Robust baseline and threshold estimation
    baseline = robust_baseline(P_MQSCPJ)
    threshold, sigma_est = robust_threshold(P_MQSCPJ, baseline, n_sigma=5.0)

    print(f"Robust baseline: {baseline:.4f}")
    print(f"Robust sigma (MAD-based): {sigma_est:.4f}")
    print(f"Detection threshold (baseline + 5σ): {threshold:.4f}")

    # Find burst events (peaks above threshold)
    peaks = np.where(P_MQSCPJ > threshold)[0]

    # Cluster nearby peaks into events (within 1ms)
    cluster_window_ms = 1.0
    cluster_samples = ms_to_samples(cluster_window_ms, dt)

    if len(peaks) == 0:
        print("No events found above threshold")
        return

    events = []
    current_event = [peaks[0]]
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i-1] < cluster_samples:
            current_event.append(peaks[i])
        else:
            events.append(current_event)
            current_event = [peaks[i]]
    events.append(current_event)

    print(f"Found {len(events)} burst events")
    print(f"  (cluster window: {cluster_window_ms} ms = {cluster_samples} samples)")

    # Window parameters (derived from dt)
    window_before_ms = 0.5
    window_after_ms = 2.0
    samples_before = ms_to_samples(window_before_ms, dt)
    samples_after = ms_to_samples(window_after_ms, dt)

    # Plot first 4 events with recovery analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (ax, event) in enumerate(zip(axes.flat, events[:4])):
        peak_idx = event[np.argmax(P_MQSCPJ[event])]

        # Window: 0.5ms before to 2ms after peak
        start = max(0, peak_idx - samples_before)
        end = min(len(time), peak_idx + samples_after)

        t_window = (time[start:end] - time[peak_idx]) * 1000  # ms relative to peak
        p_window = P_MQSCPJ[start:end]

        ax.plot(t_window, p_window, 'b-', linewidth=0.5)
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'baseline={baseline:.3f}')
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'threshold={threshold:.3f}')
        ax.axvline(x=0, color='orange', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time relative to peak (ms)')
        ax.set_ylabel('P_MQSCPJ')
        ax.set_title(f'Event {i+1}: t = {time[peak_idx]:.2f} s, peak = {P_MQSCPJ[peak_idx]:.3f}')
        ax.set_xlim(-window_before_ms, window_after_ms)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f'Chinese 63-Qubit: Cosmic Ray Events ({dt_us:.1f} μs resolution)', fontsize=12)
    plt.tight_layout()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "chinese_cosmic_ray_events.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")

    # Recovery analysis
    print("\n=== Recovery Analysis ===")
    frozen_events = []

    for i, event in enumerate(events):
        peak_idx = event[np.argmax(P_MQSCPJ[event])]
        peak_val = P_MQSCPJ[peak_idx]

        # Find 50% recovery point
        half_recovery = baseline + (peak_val - baseline) * 0.5

        # Look at 2ms window after peak
        end = min(len(time), peak_idx + samples_after)
        recovery_window = P_MQSCPJ[peak_idx:end]

        # Find time to 50% recovery
        below_half = np.where(recovery_window < half_recovery)[0]
        if len(below_half) > 0:
            t_half_us = below_half[0] * dt_us
        else:
            t_half_us = None

        if i < 5:
            if t_half_us is not None:
                print(f"Event {i+1}: Peak={peak_val:.3f}, 50% recovery at {t_half_us:.0f} μs")
            else:
                print(f"Event {i+1}: Peak={peak_val:.3f}, no 50% recovery in {window_after_ms}ms window")

        # Store frozen event for potential robustness analysis
        if freeze:
            # Extract window around peak for freezing
            freeze_before_ms = 1.0
            freeze_after_ms = 5.0
            freeze_start = max(0, peak_idx - ms_to_samples(freeze_before_ms, dt))
            freeze_end = min(len(time), peak_idx + ms_to_samples(freeze_after_ms, dt))

            frozen_events.append({
                'event_id': f'SI_Fig8a_{i}',
                'event_index': int(i),
                'peak_idx_in_window': int(peak_idx - freeze_start),
                'peak_time_s': float(time[peak_idx]),
                'peak_value': float(peak_val),
                'baseline': float(baseline),
                'threshold': float(threshold),
                't_half_recovery_us': float(t_half_us) if t_half_us is not None else None,
                'dt_us': float(dt_us),
                'window_ms': [float(freeze_before_ms), float(freeze_after_ms)],
                'time_ms': [float(x) for x in ((time[freeze_start:freeze_end] - time[peak_idx]) * 1000)],
                'values': [float(x) for x in P_MQSCPJ[freeze_start:freeze_end]],
            })

    # Save frozen events if requested
    if freeze and frozen_events:
        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        frozen_path = FROZEN_DIR / "SI_Fig8a_frozen.json"
        with open(frozen_path, 'w') as f:
            json.dump({
                'source': 'SI_Fig8a',
                'description': 'Charge-parity jump probability (P_MQSCPJ)',
                'n_events': len(frozen_events),
                'dt_us': dt_us,
                'baseline': baseline,
                'threshold': threshold,
                'sigma_est': sigma_est,
                'events': frozen_events,
            }, f, indent=2)
        print(f"\nFrozen {len(frozen_events)} events to {frozen_path}")


def analyze_bitflip_events(freeze=False):
    """
    Analyze bit flip probability data (SI_Fig12a).
    Longer timescale monitoring.

    SI_Fig12a has non-uniform sampling (~428% jitter), so all windowing and
    clustering is done in physical time, not sample-index space.

    Args:
        freeze: If True, save frozen events to JSON for robustness analysis.
    """
    print("\n=== Bit Flip Probability Analysis ===")

    data_path = DATA_DIR / "SI_Fig12" / "SI_Fig12a.npz"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Download from Figshare DOI 10.6084/m9.figshare.28815434")
        return

    # Load data
    data = np.load(data_path, allow_pickle=True)
    time = data['time']
    P_MQSBF = data['P_MQSBF']

    print(f"Duration: {time.max():.1f} s ({time.max()/3600:.2f} hours)")
    print(f"Total points: {len(time):,}")

    # -------------------------------------------------------------------------
    # Measure dt statistics (but do NOT assume uniform sampling)
    # -------------------------------------------------------------------------
    dt_diff = np.diff(time)

    # Tripwires: data integrity
    assert np.all(np.isfinite(dt_diff)), "Non-finite dt detected"
    assert np.all(dt_diff >= 0), "Time must be non-decreasing (dt < 0 found)"

    # Handle duplicate timestamps (precision loss in original data)
    n_dups = int(np.sum(dt_diff == 0))
    if n_dups > 0:
        dup_frac = n_dups / len(time)
        print(f"NOTE: {n_dups} duplicate timestamps ({dup_frac*100:.3f}%) - deduplicating...")
        # Keep first occurrence of each timestamp
        _, unique_idx = np.unique(time, return_index=True)
        unique_idx = np.sort(unique_idx)  # preserve order
        time = time[unique_idx]
        P_MQSBF = P_MQSBF[unique_idx]
        dt_diff = np.diff(time)
        print(f"  After deduplication: {len(time):,} points")
        assert np.all(dt_diff > 0), "Deduplication failed: still have non-increasing time"

    dt_med = float(np.median(dt_diff))
    dt_mean = float(np.mean(dt_diff))
    dt_ms_med = dt_med * 1000.0
    dt_ms_mean = dt_mean * 1000.0

    # Jitter relative to mean
    dt_jitter = float(np.std(dt_diff) / dt_mean) if dt_mean > 0 else float("inf")
    irregular = dt_jitter > 0.10

    # Format dt display based on magnitude
    if dt_ms_med < 0.1:
        print(f"Sampling: median dt = {dt_ms_med*1000:.1f} μs, mean dt = {dt_ms_mean:.2f} ms")
    else:
        print(f"Sampling: median dt = {dt_ms_med:.1f} ms, mean dt = {dt_ms_mean:.1f} ms")
    if irregular:
        print(f"WARNING: Non-uniform sampling (jitter = {dt_jitter*100:.0f}%). Using time-based windows.")
    else:
        # Keep original strictness when uniform
        assert dt_jitter < 0.1, f"dt jitter too large: {dt_jitter*100:.1f}%"

    # Display dt is median (robust to outliers)
    dt_ms = dt_ms_med

    # -------------------------------------------------------------------------
    # Robust baseline and threshold estimation
    # -------------------------------------------------------------------------
    baseline = robust_baseline(P_MQSBF)
    threshold, sigma_est = robust_threshold(P_MQSBF, baseline, n_sigma=3.0)

    print(f"P_MQSBF range: {P_MQSBF.min():.4f} to {P_MQSBF.max():.4f}")
    print(f"Robust baseline: {baseline:.4f}")
    print(f"Robust sigma (MAD-based): {sigma_est:.4f}")
    print(f"Detection threshold (baseline + 3σ): {threshold:.4f}")

    peaks = np.where(P_MQSBF > threshold)[0]
    print(f"Points above threshold: {len(peaks)}")

    if len(peaks) == 0:
        print("No events found above threshold")
        return

    # -------------------------------------------------------------------------
    # Cluster into events using TIME gaps (robust to irregular sampling)
    # -------------------------------------------------------------------------
    cluster_window_s = 0.5  # physical time

    events = []
    current_event = [peaks[0]]
    for i in range(1, len(peaks)):
        gap_s = float(time[peaks[i]] - time[peaks[i-1]])
        if gap_s < cluster_window_s:
            current_event.append(peaks[i])
        else:
            events.append(current_event)
            current_event = [peaks[i]]
    events.append(current_event)

    print(f"Burst events found: {len(events)}")
    print(f"  (cluster window: {cluster_window_s:.2f} s, time-based)")
    print(f"Rate: {len(events)/time.max()*3600:.1f} events/hour")

    # -------------------------------------------------------------------------
    # Window parameters (physical time, robust to irregular sampling)
    # -------------------------------------------------------------------------
    window_before_s = 5.0
    window_after_s = 20.0

    # Plot first 4 events
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (ax, event) in enumerate(zip(axes.flat, events[:4])):
        peak_idx = event[np.argmax(P_MQSBF[event])]
        t0 = float(time[peak_idx])

        # Time-based slice using searchsorted
        start_t = t0 - window_before_s
        end_t = t0 + window_after_s
        start = int(np.searchsorted(time, start_t, side="left"))
        end = int(np.searchsorted(time, end_t, side="right"))

        t_window = time[start:end] - t0
        p_window = P_MQSBF[start:end]

        ax.plot(t_window, p_window, 'b-', linewidth=0.8)
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'baseline={baseline:.4f}')
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'threshold={threshold:.4f}')
        ax.axvline(x=0, color='orange', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time relative to peak (s)')
        ax.set_ylabel('P_MQSBF')
        ax.set_title(f'Event {i+1}: t = {time[peak_idx]:.1f} s')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f'Chinese 63-Qubit: Bit Flip Events (median dt ≈ {dt_ms:.0f} ms)', fontsize=12)
    plt.tight_layout()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "chinese_bitflip_events.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nSaved: {output_path}")

    # -------------------------------------------------------------------------
    # Save frozen events if requested (time-based extraction)
    # -------------------------------------------------------------------------
    if freeze:
        frozen_events = []
        for i, event in enumerate(events):
            peak_idx = event[np.argmax(P_MQSBF[event])]
            peak_val = float(P_MQSBF[peak_idx])
            t0 = float(time[peak_idx])

            # Freeze window in physical time
            freeze_before_s = 2.0
            freeze_after_s = 30.0
            start = int(np.searchsorted(time, t0 - freeze_before_s, side="left"))
            end = int(np.searchsorted(time, t0 + freeze_after_s, side="right"))

            frozen_events.append({
                'event_id': f'SI_Fig12a_{i}',
                'event_index': int(i),
                'peak_time_s': t0,
                'peak_value': peak_val,
                'baseline': float(baseline),
                'threshold': float(threshold),
                'dt_median_ms': dt_ms_med,
                'dt_mean_ms': dt_ms_mean,
                'dt_jitter': dt_jitter,
                'irregular_sampling': irregular,
                'window_s': [float(freeze_before_s), float(freeze_after_s)],
                'time_ms': [float((t - t0) * 1000) for t in time[start:end]],
                'values': [float(x) for x in P_MQSBF[start:end]],
            })

        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        frozen_path = FROZEN_DIR / "SI_Fig12a_frozen.json"
        with open(frozen_path, 'w') as f:
            json.dump({
                'source': 'SI_Fig12a',
                'description': 'Bit flip probability (P_MQSBF)',
                'n_events': len(frozen_events),
                'dt_median_ms': dt_ms_med,
                'dt_mean_ms': dt_ms_mean,
                'dt_jitter': dt_jitter,
                'irregular_sampling': irregular,
                'baseline': float(baseline),
                'threshold': float(threshold),
                'sigma_est': float(sigma_est),
                'events': frozen_events,
            }, f, indent=2)
        print(f"Frozen {len(frozen_events)} events to {frozen_path}")


def main():
    """Main analysis routine."""
    global DATA_DIR

    parser = argparse.ArgumentParser(
        description="Analyze Chinese 63-qubit cosmic ray data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data source: Figshare DOI 10.6084/m9.figshare.28815434

Download the SI_Fig8 and SI_Fig12 directories from Figshare and place them in:
    scripts/data/chinese_cosmic_ray/
Or specify a custom path with --data_dir.

Pipeline:
    1. python chinese_cosmic_ray_analysis.py --freeze  # Extract events
    2. python chinese_robustness_sweep.py              # Run model competition
    3. python chinese_stability_diagnostics.py         # Compute evidence strength
        """
    )
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to directory containing SI_Fig8/ and SI_Fig12/')
    parser.add_argument('--freeze', action='store_true',
                        help='Save frozen events to JSON for robustness analysis')
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = Path(args.data_dir)

    print("=" * 60)
    print("Chinese 63-Qubit Cosmic Ray Analysis")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    if args.freeze:
        print(f"Freeze mode: events will be saved to {FROZEN_DIR}")

    analyze_cosmic_ray_events(freeze=args.freeze)
    analyze_bitflip_events(freeze=args.freeze)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    if args.freeze:
        print(f"Frozen events saved to: {FROZEN_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
