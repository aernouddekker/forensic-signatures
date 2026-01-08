#!/usr/bin/env python3
"""
Hesitation Phase Diagram Generator
===================================

Generates the hesitation phase diagram showing delay fraction D vs curvature
index b for both McEwan (Google Sycamore) and LIGO platforms.

The diagram visualizes two distinct modes of Ignorance-Originated Fluctuation
(IOF) behavior:
- McEwan: High delay fraction D, normal curvature (temporal hesitation)
- LIGO: Low delay fraction D, high curvature (geometric hesitation)

Data sources:
- McEwan: robustness_sweep/stability_table.json (15-run stability analysis)
- LIGO: ligo_envelope/stability_events.jsonl (3-window stability analysis)

Author: Aernoud Dekker
Date: December 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"

# McEwan (Google Sycamore) data paths
MCEWAN_STABILITY_FILE = OUTPUT_DIR / "robustness_sweep" / "stability_table.json"
MCEWAN_DETAILED_FILE = OUTPUT_DIR / "robustness_sweep" / "detailed_results.json"
MCEWAN_FROZEN_FILE = OUTPUT_DIR / "frozen_events.json"

# LIGO data paths
LIGO_EVENTS_FILE = OUTPUT_DIR / "ligo_envelope" / "stability_events.jsonl"
LIGO_RESULTS_FILE = OUTPUT_DIR / "ligo_envelope" / "ligo_envelope_Extremely_Loud_results.jsonl"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Curvature computation
# -----------------------------------------------------------------------------

def compute_curvature_index(t, y, window_ms=20.0):
    """
    Compute early-time curvature index b over the specified window.

    Parameters
    ----------
    t : array-like
        Time values in milliseconds (relative to peak, can include negatives)
    y : array-like
        Normalized recovery coordinate z(t)
    window_ms : float
        Window size for quadratic fit (default: 20 ms)

    Returns
    -------
    float or None
        Quadratic coefficient b from polynomial fit, or None if insufficient data
    """
    # Post-peak only: t >= 0 ensures we fit recovery, not pre-peak segment
    mask = (t >= 0) & (t <= window_ms)
    if np.sum(mask) < 10:
        return None
    t_fit = t[mask]
    y_fit = y[mask]
    try:
        coeffs = np.polyfit(t_fit, y_fit, 2)
        return coeffs[0]
    except Exception:
        return None


# -----------------------------------------------------------------------------
# McEwan data loading
# -----------------------------------------------------------------------------

def load_mcewan_data():
    """
    Load McEwan (Google Sycamore) data with stability classifications.

    Returns
    -------
    list of dict
        Each dict contains: event_id, stability, D, curvature_b, platform
    """
    print("Loading McEwan data...")

    with open(MCEWAN_STABILITY_FILE) as f:
        stability_data = json.load(f)

    with open(MCEWAN_DETAILED_FILE) as f:
        detailed_data = json.load(f)

    with open(MCEWAN_FROZEN_FILE) as f:
        frozen_data = json.load(f)

    # Index raw event data by event_id
    raw_events = {}
    for event in frozen_data['events']:
        event_id = f"{event['dataset']}_{event['event_index']}"
        raw_events[event_id] = event

    # Use default analysis run (B1_01: 60ms window, argmin alignment)
    default_run = detailed_data.get('B1_01', [])
    event_results = {r['event_id']: r for r in default_run}

    # Stability label mapping (include both 'flip' and 'true_flip' for compatibility)
    stability_labels = {
        'stable_iof': 'Stable Delayed',
        'stable_std': 'Stable Fast',
        'flip': 'Flip',
        'true_flip': 'Flip',  # backward-compat alias (summary counters use this)
        'fluctuate_uncertain': None,  # exclude uncertain fluctuators
        'uncertain_fluctuate': None,  # alternate spelling of fluctuate_uncertain
    }

    # Tripwire: check for unknown or missing labels
    known_labels = set(stability_labels.keys())
    seen_labels = {v.get('stability') for v in stability_data['event_details'].values()}
    assert None not in seen_labels, "Missing 'stability' key in some McEwan event_details entries"
    unknown_labels = sorted(seen_labels - known_labels)
    assert not unknown_labels, f"Unknown McEwan stability labels: {unknown_labels}"

    # Sanity print: label counts
    label_counts = Counter(v.get('stability') for v in stability_data['event_details'].values())
    print(f"  McEwan stability label counts: {dict(label_counts)}")

    events = []
    for event_id, stability_info in stability_data['event_details'].items():
        stability = stability_info['stability']
        label = stability_labels.get(stability)
        if label is None:
            continue

        result = event_results.get(event_id)
        if result is None:
            continue

        # Model-implied inflection time
        t_inf = result.get('model_t_peak', 0.0)

        raw = raw_events.get(event_id)
        if raw is None:
            continue

        t = np.array(raw['t_window'])
        y = np.array(raw['y_window'])
        baseline = raw['baseline']

        # Normalize to recovery coordinate z(t) in [0, 1]
        min_val = np.min(y)
        if baseline > min_val:
            z = (y - min_val) / (baseline - min_val)
        else:
            continue

        curvature_b = compute_curvature_index(t, z)
        if curvature_b is None:
            continue

        # Characteristic timescale (median from ensemble)
        tau = 11.0  # ms

        # Delay fraction
        D = t_inf / tau if tau > 0 else 0

        events.append({
            'event_id': event_id,
            'stability': label,
            'D': D,
            'curvature_b': curvature_b,
            'platform': 'McEwan'
        })

    print(f"  Loaded {len(events)} McEwan events")
    return events


# -----------------------------------------------------------------------------
# LIGO data loading
# -----------------------------------------------------------------------------

def load_ligo_data():
    """
    Load LIGO data with 3-window stability classifications.

    Primary source: stability_events.jsonl (3-window stability analysis)
    Fallback: single-window classification from results file

    Returns
    -------
    list of dict
        Each dict contains: event_id, stability, D, curvature_b, platform
    """
    print("Loading LIGO data...")

    stability_labels = {
        'stable_iof': 'Stable Delayed',
        'stable_std': 'Stable Fast',
        'flip': 'Flip',
        'failed': None
    }

    if LIGO_EVENTS_FILE.exists():
        print(f"  Using 3-window stability from {LIGO_EVENTS_FILE.name}")
        events = []
        with open(LIGO_EVENTS_FILE) as f:
            for line in f:
                e = json.loads(line)
                label = stability_labels.get(e['stability'])
                if label is None:
                    continue

                events.append({
                    'event_id': f"LIGO_{int(e['gps_time'])}",
                    'stability': label,
                    'D': e['D'],
                    'curvature_b': e['curvature_b'],
                    'platform': 'LIGO'
                })

        n_delayed = sum(1 for e in events if e['stability'] == 'Stable Delayed')
        n_fast = sum(1 for e in events if e['stability'] == 'Stable Fast')
        n_flip = sum(1 for e in events if e['stability'] == 'Flip')
        print(f"  3-window stability: {n_delayed} Delayed, {n_fast} Fast, {n_flip} Flip")

        return events

    # Fallback to single-window classification
    print(f"  {LIGO_EVENTS_FILE.name} not found, using single-window fallback")

    events = []
    with open(LIGO_RESULTS_FILE) as f:
        for line in f:
            event = json.loads(line)
            if event.get('status') != 'ok':
                continue

            classification = event.get('classification', 'uncertain')
            if classification == 'iof':
                label = 'Stable Delayed'
            elif classification == 'standard':
                label = 'Stable Fast'
            else:
                continue

            t_inf = event.get('model_t_peak_ms', 0)
            tau = event.get('tau_ms', 20)
            D = t_inf / tau if tau > 0 else 0

            events.append({
                'event_id': f"LIGO_{int(event['gps_time'])}",
                'stability': label,
                'D': D,
                'curvature_b': 0,
                'platform': 'LIGO'
            })

    n_delayed = sum(1 for e in events if e['stability'] == 'Stable Delayed')
    n_fast = sum(1 for e in events if e['stability'] == 'Stable Fast')
    print(f"  Single-window fallback: {n_delayed} Delayed, {n_fast} Fast")

    return events


# -----------------------------------------------------------------------------
# Phase diagram generation
# -----------------------------------------------------------------------------

def generate_phase_diagram(mcewan_events, ligo_events):
    """
    Generate the hesitation phase diagram showing D vs curvature b.

    Parameters
    ----------
    mcewan_events : list of dict
        McEwan event data
    ligo_events : list of dict
        LIGO event data
    """
    print("\nGenerating phase diagram...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Separate by platform and stability class
    mcewan_std = [e for e in mcewan_events if e['stability'] == 'Stable Fast']
    mcewan_iof = [e for e in mcewan_events if e['stability'] == 'Stable Delayed']
    mcewan_flip = [e for e in mcewan_events if e['stability'] == 'Flip']
    ligo_std = [e for e in ligo_events if e['stability'] == 'Stable Fast']
    ligo_iof = [e for e in ligo_events if e['stability'] == 'Stable Delayed']
    ligo_flip = [e for e in ligo_events if e['stability'] == 'Flip']

    print(f"\nEvent counts:")
    print(f"  McEwan STD: {len(mcewan_std)}, IOF: {len(mcewan_iof)}, Flip: {len(mcewan_flip)}")
    print(f"  LIGO STD: {len(ligo_std)}, IOF: {len(ligo_iof)}, Flip: {len(ligo_flip)}")

    def scale_b(events):
        """Scale curvature to 10^-3 units for display."""
        return [e['curvature_b'] * 1000 for e in events]

    def get_D(events):
        """Extract delay fraction values."""
        return [e['D'] for e in events]

    # McEwan standard recovery (circles, green)
    if mcewan_std:
        ax.scatter(get_D(mcewan_std), scale_b(mcewan_std),
                   c='lightgreen', marker='o', s=60, alpha=0.7, edgecolors='green',
                   label=f'McEwan STD (n={len(mcewan_std)})')

    # McEwan IOF (circles, red)
    if mcewan_iof:
        ax.scatter(get_D(mcewan_iof), scale_b(mcewan_iof),
                   c='coral', marker='o', s=80, alpha=0.8, edgecolors='red',
                   label=f'McEwan IOF (n={len(mcewan_iof)})')

    # LIGO standard recovery (squares, green)
    if ligo_std:
        ax.scatter(get_D(ligo_std), scale_b(ligo_std),
                   c='lightgreen', marker='s', s=60, alpha=0.7, edgecolors='green',
                   label=f'LIGO STD (n={len(ligo_std)})')

    # LIGO IOF (squares, red)
    if ligo_iof:
        ax.scatter(get_D(ligo_iof), scale_b(ligo_iof),
                   c='coral', marker='s', s=80, alpha=0.8, edgecolors='red',
                   label=f'LIGO IOF (n={len(ligo_iof)})')

    # McEwan Flip (circles, gray - ambiguous/boundary events)
    if mcewan_flip:
        ax.scatter(get_D(mcewan_flip), scale_b(mcewan_flip),
                   c='lightgray', marker='o', s=40, alpha=0.4, edgecolors='gray',
                   label=f'McEwan Flip (n={len(mcewan_flip)})')

    # LIGO Flip (squares, gray - ambiguous/boundary events)
    if ligo_flip:
        ax.scatter(get_D(ligo_flip), scale_b(ligo_flip),
                   c='lightgray', marker='s', s=40, alpha=0.4, edgecolors='gray',
                   label=f'LIGO Flip (n={len(ligo_flip)})')

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # Interpretation annotations
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7)

    ax.annotate('LIGO IOF:\nHigh curvature\nLow delay',
                xy=(0.15, 0.5), fontsize=9, bbox=bbox_props)

    ax.annotate('McEwan IOF:\nHigh delay\nNormal curvature',
                xy=(0.15, -2.0), fontsize=9, bbox=bbox_props)

    # Axis labels
    ax.set_xlabel(r'Delay fraction $D = t_{\mathrm{inf}}/\tau$', fontsize=12)
    ax.set_ylabel(r'Curvature index $b$ ($\times 10^3$)', fontsize=12)
    ax.set_title('Hesitation Phase Diagram: Two Modes of IOF', fontsize=14)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    # Axis limits based on data range with padding
    all_D = (get_D(mcewan_std) + get_D(mcewan_iof) + get_D(mcewan_flip) +
             get_D(ligo_std) + get_D(ligo_iof) + get_D(ligo_flip))
    max_D = max(all_D) if all_D else 1
    ax.set_xlim(-0.1, max_D * 1.15)

    plt.tight_layout()

    # Save outputs
    output_path = FIGURES_DIR / "hesitation_phase_diagram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    output_path2 = OUTPUT_DIR / "hesitation_phase_diagram.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")

    plt.close()


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main():
    """Generate hesitation phase diagram from McEwan and LIGO data."""
    print("=" * 70)
    print("Hesitation Phase Diagram Generator")
    print("=" * 70)
    print()

    mcewan_events = load_mcewan_data()
    ligo_events = load_ligo_data()

    generate_phase_diagram(mcewan_events, ligo_events)

    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
