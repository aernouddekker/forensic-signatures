#!/usr/bin/env python3
"""
Pipeline Consistency Regression Test
=====================================

Verifies that peak/baseline/z/curvature values are consistent across
the canonical ligo_pipeline_common.py module.

Run this after any pipeline changes to catch drift.

Usage:
    python test_pipeline_consistency.py

Author: Aernoud Dekker
Date: December 2025
"""

import json
import pickle
import numpy as np
from pathlib import Path

from ligo_pipeline_common import (
    compute_times_ms,
    compute_hilbert_envelope,
    find_constrained_peak,
    baseline_from_postpeak_window,
    extract_fit_window,
    compute_curvature_index,
    load_cached_strain,
    ANALYSIS_WINDOWS_MS,
    PEAK_SEARCH_WINDOW_MS,
)

SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = SCRIPT_DIR / "strain_cache"
RESULTS_FILE = SCRIPT_DIR / "output" / "ligo_envelope" / "ligo_envelope_Extremely_Loud_results.jsonl"


def pick_10_gps():
    """Select 10 OK events for testing."""
    gps = []
    if not RESULTS_FILE.exists():
        return gps
    with open(RESULTS_FILE) as f:
        for line in f:
            e = json.loads(line)
            if e.get("status") == "ok":
                gps.append(float(e["gps_time"]))
            if len(gps) >= 10:
                break
    return gps


def main():
    print("=" * 70)
    print("Pipeline Consistency Regression Test")
    print("=" * 70)

    gps_list = pick_10_gps()
    if not gps_list:
        print("No OK events found. Run ligo_glitch_analysis.py first.")
        return 1

    print(f"\nTesting {len(gps_list)} events...")
    print(f"Peak search window: +/-{PEAK_SEARCH_WINDOW_MS} ms")
    print(f"Analysis windows: {ANALYSIS_WINDOWS_MS} ms")
    print()

    rows = []
    for gps in gps_list:
        d = load_cached_strain(gps, CACHE_DIR)
        if d is None:
            print(f"  GPS {gps:.6f}: cache miss")
            continue

        times = np.asarray(d["times"])
        strain = np.asarray(d["values"])
        fs = float(d["sample_rate"])

        # Canonical pipeline operations
        times_ms = compute_times_ms(times, gps)
        env = compute_hilbert_envelope(strain, fs, bandpass=True)
        peak_idx = find_constrained_peak(env, times_ms)
        peak_t = float(times_ms[peak_idx])
        peak_val = float(env[peak_idx])

        # Per-window baseline + z values
        perW = {}
        for W in ANALYSIS_WINDOWS_MS:
            t_fit, z_fit, _, baseline = extract_fit_window(env, times_ms, peak_idx, W)
            perW[W] = {
                "baseline": float(baseline),
                "z0": float(z_fit[0]) if len(z_fit) else None,
                "z_end": float(z_fit[-1]) if len(z_fit) else None,
                "n": int(len(z_fit)),
            }

        # Curvature index
        b = compute_curvature_index(env, times_ms, peak_idx)

        rows.append({
            "gps": gps,
            "peak_t_ms": peak_t,
            "peak_val": peak_val,
            "b": None if b is None else float(b),
            "perW": perW
        })

    # Print compact diff-friendly output
    print("-" * 70)
    for r in rows:
        b_str = f"{r['b']:.6f}" if r['b'] is not None else "None"
        print(f"\nGPS {r['gps']:.6f}  peak_t={r['peak_t_ms']:+7.2f} ms  b={b_str}")
        for W in ANALYSIS_WINDOWS_MS:
            x = r["perW"][W]
            z0_str = f"{x['z0']:.4f}" if x['z0'] is not None else "None"
            zend_str = f"{x['z_end']:.4f}" if x['z_end'] is not None else "None"
            print(f"  W={W:3d}ms  baseline={x['baseline']:.4e}  z0={z0_str}  zend={zend_str}  n={x['n']}")

    print("\n" + "-" * 70)
    print("\nConsistency checks:")

    # Check 1: peak_t should be within search window
    peak_violations = [r for r in rows if abs(r['peak_t_ms']) > PEAK_SEARCH_WINDOW_MS]
    if peak_violations:
        print(f"  [FAIL] {len(peak_violations)} events have peak outside search window")
    else:
        print(f"  [PASS] All peaks within +/-{PEAK_SEARCH_WINDOW_MS} ms of GPS time")

    # Check 2: z0 should be near 0
    z0_violations = []
    for r in rows:
        for W in ANALYSIS_WINDOWS_MS:
            z0 = r['perW'][W]['z0']
            if z0 is not None and abs(z0) > 0.1:
                z0_violations.append((r['gps'], W, z0))
    if z0_violations:
        print(f"  [WARN] {len(z0_violations)} window(s) have z0 > 0.1 (may indicate normalization issue)")
    else:
        print(f"  [PASS] All z0 values near 0 (correct normalization)")

    # Check 3: baseline should be positive and smaller than peak
    baseline_violations = []
    for r in rows:
        for W in ANALYSIS_WINDOWS_MS:
            bl = r['perW'][W]['baseline']
            if bl <= 0 or bl >= r['peak_val']:
                baseline_violations.append((r['gps'], W, bl))
    if baseline_violations:
        print(f"  [WARN] {len(baseline_violations)} baseline(s) are non-positive or >= peak")
    else:
        print(f"  [PASS] All baselines are positive and < peak")

    print("\n" + "=" * 70)
    print("Test complete. Save this output to compare after pipeline changes.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
