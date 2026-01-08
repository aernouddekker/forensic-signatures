#!/usr/bin/env python3
"""
Chinese Robustness Sweep (Model Competition)
===========================================

Consumes frozen events from chinese_cosmic_ray_analysis.py --freeze and runs
a configuration sweep with two models:

  STD: y(t) = baseline + A * exp(-t/tau)
  IOF: y(t) = baseline + A * exp(-max(0, t - t0)/tau)   (dead-time / hesitation)

Baseline is fixed from frozen metadata to prevent "baseline teleportation".

Outputs:
  scripts/output/chinese_robustness_sweep/detailed_results.json
  scripts/output/chinese_robustness_sweep/stability_table.json

Author: Aernoud Dekker
Date: Dec 2025
"""

import json
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

SCRIPT_DIR = Path(__file__).parent
FROZEN_DIR = SCRIPT_DIR / "output" / "chinese_frozen_events"
OUT_DIR = SCRIPT_DIR / "output" / "chinese_robustness_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SI_Fig12a excluded from model competition due to:
#   - Bimodal sampling (429% jitter) inflates AICc via sample count (r=0.786)
#   - 94% of IOF t0 values peg at grid maximum (non-identifiable)
# See audit notes in stability_table.json
DEFAULT_FILES = [
    FROZEN_DIR / "SI_Fig8a_frozen.json",
    # FROZEN_DIR / "SI_Fig12a_frozen.json",  # EXCLUDED - see note above
]


# -----------------------------------------------------------------------------
# AICc
# -----------------------------------------------------------------------------

def aicc_from_sse(sse: float, n: int, k: int, eps: float = 1e-15) -> float:
    """
    AICc for Gaussian residuals (up to an additive constant):
      AIC  = n*ln(SSE/n) + 2k
      AICc = AIC + 2k(k+1)/(n-k-1)
    """
    sse = max(float(sse), eps)
    n = int(n)
    k = int(k)
    aic = n * math.log(sse / n) + 2 * k
    if n - k - 1 <= 0:
        return float("inf")
    return aic + (2 * k * (k + 1)) / (n - k - 1)


# -----------------------------------------------------------------------------
# Models & fitting (grid search, baseline fixed)
# -----------------------------------------------------------------------------

def _best_amp(y, baseline, x):
    """
    Solve y ≈ baseline + A*x in least squares (A>=0 constraint).
    """
    r = y - baseline
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    A = float(np.dot(r, x) / denom)
    return max(0.0, A)


def fit_std(t, y, baseline, tau_grid_ms):
    """
    Fit STD model by grid search over tau; solve for A analytically each tau.
    Returns dict with best params + SSE.
    """
    best = {"sse": float("inf"), "tau_ms": None, "A": None}
    for tau in tau_grid_ms:
        x = np.exp(-t / tau)
        A = _best_amp(y, baseline, x)
        yhat = baseline + A * x
        sse = float(np.sum((y - yhat) ** 2))
        if sse < best["sse"]:
            best = {"sse": sse, "tau_ms": float(tau), "A": float(A)}
    return best


def fit_iof_deadtime(t, y, baseline, tau_grid_ms, t0_grid_ms):
    """
    Fit IOF "dead-time" model by grid search over (t0, tau); solve for A analytically.
    """
    best = {"sse": float("inf"), "tau_ms": None, "t0_ms": None, "A": None}
    for t0 in t0_grid_ms:
        td = np.maximum(0.0, t - t0)
        for tau in tau_grid_ms:
            x = np.exp(-td / tau)
            A = _best_amp(y, baseline, x)
            yhat = baseline + A * x
            sse = float(np.sum((y - yhat) ** 2))
            if sse < best["sse"]:
                best = {"sse": sse, "tau_ms": float(tau), "t0_ms": float(t0), "A": float(A)}
    return best


# -----------------------------------------------------------------------------
# Sweep configuration
# -----------------------------------------------------------------------------

def build_configs_for_source(source_name: str):
    """
    Define a small but meaningful sweep for each source.

    SI_Fig8a is microsecond-scale; windows in ms.
    SI_Fig12a is seconds-scale; windows in s (coarser grid for speed).
    """
    if source_name == "SI_Fig8a":
        # fit_end_ms values (post-peak) — keep modest for speed.
        fit_ends = [0.3, 0.5, 1.0, 2.0, 5.0]
        # grids (ms)
        tau_grid = np.geomspace(0.02, 5.0, 80)     # 20 µs .. 5 ms (80 points)
        t0_grid = np.linspace(0.0, 1.0, 41)        # 0 .. 1 ms (41 points)
        max_events = None  # all events
        return fit_ends, tau_grid, t0_grid, "ms", max_events
    else:
        # SI_Fig12a: coarser grid since data has lower resolution
        fit_ends = [5, 10, 20, 30, 60]
        tau_grid = np.geomspace(50.0, 20_000.0, 50)    # 50 ms .. 20 s in ms (50 points)
        t0_grid = np.linspace(0.0, 2000.0, 31)         # 0 .. 2 s in ms (31 points)
        max_events = 200  # subsample for tractability
        return fit_ends, tau_grid, t0_grid, "s", max_events


def classify(delta_aicc_std_minus_iof: float, eps: float = 1e-9):
    """
    delta = AICc_STD - AICc_IOF
      delta > 0  => IOF better
      delta < 0  => STD better
    """
    if delta_aicc_std_minus_iof > eps:
        return "iof"
    elif delta_aicc_std_minus_iof < -eps:
        return "standard"
    return "tie"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def load_frozen(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("Chinese Robustness Sweep (Model Competition)")
    print("=" * 70)

    frozen_files = [p for p in DEFAULT_FILES if p.exists()]
    if not frozen_files:
        raise SystemExit(f"No frozen files found in {FROZEN_DIR}. Run: python chinese_cosmic_ray_analysis.py --freeze")

    detailed_results = {}
    per_event_votes = defaultdict(list)  # event_id -> list of class labels across configs

    for fp in frozen_files:
        blob = load_frozen(fp)
        source = blob.get("source")
        events = blob.get("events", [])
        if not source or not events:
            print(f"Skipping {fp.name}: missing source/events")
            continue

        fit_ends, tau_grid_ms, t0_grid_ms, window_units, max_events = build_configs_for_source(source)

        # Subsample if needed
        n_original = len(events)
        if max_events is not None and len(events) > max_events:
            np.random.seed(42)  # reproducible
            indices = np.random.choice(len(events), max_events, replace=False)
            events = [events[i] for i in sorted(indices)]
            print(f"\n=== {source}: {n_original} events -> subsampled to {len(events)} ===")
        else:
            print(f"\n=== {source}: {len(events)} frozen events ===")

        print(f"Sweep windows ({window_units}): {fit_ends}")
        print(f"Grid: tau={len(tau_grid_ms)} pts, t0={len(t0_grid_ms)} pts")

        # build config keys like C01..C05
        for ci, fit_end in enumerate(fit_ends, start=1):
            config_key = f"{source}_C{ci:02d}"
            detailed_results[config_key] = []
            print(f"  {config_key}: fitting {len(events)} events...", end="", flush=True)

            for e in events:
                event_id = e["event_id"]
                baseline = float(e["baseline"])

                # Use post-peak only (t>=0)
                t = np.array(e["time_ms"], dtype=float)
                y = np.array(e["values"], dtype=float)

                mask = t >= 0.0
                t = t[mask]
                y = y[mask]

                # Apply fit window
                if window_units == "ms":
                    tmax = float(fit_end)
                else:
                    tmax = float(fit_end) * 1000.0  # seconds -> ms
                mask2 = t <= tmax
                t2 = t[mask2]
                y2 = y[mask2]

                # Tripwire: require enough samples
                if len(t2) < 12:
                    continue

                # STD fit
                std = fit_std(t2, y2, baseline, tau_grid_ms)
                aicc_std = aicc_from_sse(std["sse"], n=len(t2), k=2)  # A, tau

                # IOF fit
                iof = fit_iof_deadtime(t2, y2, baseline, tau_grid_ms, t0_grid_ms)
                aicc_iof = aicc_from_sse(iof["sse"], n=len(t2), k=3)  # A, tau, t0

                delta = aicc_std - aicc_iof
                winner = classify(delta)
                per_event_votes[event_id].append(winner)

                detailed_results[config_key].append({
                    "event_id": event_id,
                    "source": source,
                    "fit_end_ms": tmax,
                    "n": int(len(t2)),
                    "baseline": baseline,
                    "aicc_std": float(aicc_std),
                    "aicc_iof": float(aicc_iof),
                    "delta_aicc_std_minus_iof": float(delta),
                    "abs_delta_aicc": float(abs(delta)),
                    "winner": winner,
                    "std_tau_ms": std["tau_ms"],
                    "std_A": std["A"],
                    "iof_tau_ms": iof["tau_ms"],
                    "iof_t0_ms": iof["t0_ms"],
                    "iof_A": iof["A"],
                })

            print(f" done ({len(detailed_results[config_key])} fits)")

    # Build stability table (across all configs that produced a winner)
    event_details = {}
    stability_counts = Counter()

    for event_id, votes in per_event_votes.items():
        # Drop ties from voting; they indicate true ambiguity or numerical tie
        votes2 = [v for v in votes if v in ("iof", "standard")]
        if len(votes2) == 0:
            stability = "fluctuate_uncertain"
        else:
            s = set(votes2)
            if s == {"iof"}:
                stability = "stable_iof"
            elif s == {"standard"}:
                stability = "stable_std"
            else:
                stability = "flip"

        stability_counts[stability] += 1
        event_details[event_id] = {
            "stability": stability,
            "n_votes_total": int(len(votes)),
            "n_votes_used": int(len(votes2)),
            "votes": votes,
        }

    stability_table = {
        "stability_counts": dict(stability_counts),
        "event_details": event_details,
        "n_events": int(len(event_details)),
        "n_configs": int(len(detailed_results)),
        "notes": {
            "delta_definition": "delta = AICc_STD - AICc_IOF; delta>0 favors IOF",
            "models": {
                "STD": "baseline + A*exp(-t/tau)",
                "IOF": "baseline + A*exp(-max(0,t-t0)/tau) (dead-time/hesitation)",
            },
            "baseline": "fixed from frozen metadata (prevents optimizer cheating)",
            "excluded_sources": {
                "SI_Fig12a": "Excluded due to (1) bimodal sampling inflating AICc "
                             "(corr(n,|dAICc|)=0.786, n up to 49k samples), "
                             "(2) 94% of IOF t0 values peg at grid max (non-identifiable). "
                             "Naive sample-wise AICc not meaningful for irregular sampling."
            },
        }
    }

    # Save outputs
    detailed_path = OUT_DIR / "detailed_results.json"
    with open(detailed_path, "w") as f:
        json.dump(detailed_results, f, indent=2)

    stability_path = OUT_DIR / "stability_table.json"
    with open(stability_path, "w") as f:
        json.dump(stability_table, f, indent=2)

    print("\n" + "=" * 70)
    print("Chinese robustness sweep complete")
    print("=" * 70)
    print(f"Saved: {detailed_path}")
    print(f"Saved: {stability_path}")
    print(f"Stability counts: {dict(stability_counts)}")


if __name__ == "__main__":
    main()
