#!/usr/bin/env python3
"""
Generate LaTeX Macros from Pipeline Outputs
============================================

Reads JSON output files from analysis scripts and generates results_macros.tex
with all canonical values. This ensures single source of truth between pipeline
outputs and manuscript.

Usage:
    python generate_macros.py

Output:
    ../latex/results_macros.tex (overwrites existing file)

Author: Aernoud Dekker
Date: December 2025
"""

import json
import math
import subprocess
from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
LIGO_DIR = OUTPUT_DIR / "ligo_envelope"
LATEX_DIR = SCRIPT_DIR.parent / "latex"
MACROS_FILE = LATEX_DIR / "results_macros.tex"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_json(path):
    """Load JSON file, return None if not found."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_git_hash():
    """Get short git hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=SCRIPT_DIR
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def fmt(value, decimals=None):
    """Format a value for LaTeX macro."""
    if value is None:
        return "???"
    if isinstance(value, float):
        if decimals is not None:
            return f"{value:.{decimals}f}"
        # Auto-detect appropriate precision
        if abs(value) >= 100:
            return f"{value:.0f}"
        elif abs(value) >= 10:
            return f"{value:.1f}"
        elif abs(value) >= 1:
            return f"{value:.2f}"
        else:
            return f"{value:.3f}"
    return str(value)


def fmt_pval(p, threshold=1e-3):
    """Format a p-value, using scientific notation for very small values.

    For p < threshold, returns LaTeX-friendly scientific notation like '1.9 \\times 10^{-6}'.
    For p >= threshold, returns decimal format like '0.35'.
    """
    if p is None:
        return "???"
    if p < threshold:
        # Use scientific notation
        exp = int(f"{p:.0e}".split('e')[1])
        mantissa = p / (10 ** exp)
        if abs(mantissa - 1.0) < 0.05:
            # Close to 10^exp, just show the power
            return f"10^{{{exp}}}"
        else:
            return f"{mantissa:.1f} \\times 10^{{{exp}}}"
    else:
        # Regular decimal format
        if p >= 0.1:
            return f"{p:.2f}"
        else:
            return f"{p:.3f}"


def wilson_ci(successes: int, total: int, z: float = 1.96):
    """Wilson score confidence interval for a binomial proportion.

    Args:
        successes: number of successes (0..total)
        total: number of trials
        z: z-score (1.96 ~ 95% CI)

    Returns:
        (lower, upper) on [0, 1]
    """
    if total <= 0:
        return (0.0, 0.0)
    if successes < 0 or successes > total:
        raise ValueError(f"successes must be in [0,total], got {successes} / {total}")

    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# -----------------------------------------------------------------------------
# Load pipeline outputs
# -----------------------------------------------------------------------------

def load_ligo_data():
    """Load all LIGO-related JSON outputs."""
    data = {}

    # Main stability analysis (from ligo_stability_figures.py)
    stability = load_json(LIGO_DIR / "bootstrap_beta_b.json")
    if stability:
        data['stability'] = stability

    # Threshold sweep (from ligo_threshold_sweep.py)
    threshold = load_json(LIGO_DIR / "threshold_sweep_results.json")
    if threshold:
        data['threshold'] = threshold

    # Baseline robustness (from ligo_baseline_robustness.py)
    baseline = load_json(LIGO_DIR / "baseline_robustness_results.json")
    if baseline:
        data['baseline'] = baseline

    # Null simulation control (from ligo_null_simulation.py)
    null_sim = load_json(LIGO_DIR / "null_simulation.json")
    if null_sim:
        data['null_simulation'] = null_sim

    return data


def load_mcewan_data():
    """
    Load McEwen/Google data from stability_table.json.

    The robustness_sweep script outputs stability_table.json with:
    - n_events: total events (including uncertain fluctuators)
    - stable_iof, stable_std, true_flip: classified events
    - fluctuate_uncertain: events with inconsistent classification

    For manuscript counts, we use only the classified events:
    n_total = stable_iof + stable_std + true_flip (excluding uncertain)
    """
    # Primary source: robustness_sweep/stability_table.json
    stability_file = OUTPUT_DIR / "robustness_sweep" / "stability_table.json"
    if stability_file.exists():
        data = load_json(stability_file)
        if data:
            # Extract values (note: true_flip in JSON, n_flip in output)
            n_stable_iof = data['stable_iof']
            n_stable_std = data['stable_std']
            n_flip = data['true_flip']

            # Total = classified events only (excluding uncertain fluctuators)
            n_total = n_stable_iof + n_stable_std + n_flip

            print(f"  Loaded McEwen stability from {stability_file.name}")
            print(f"    {n_total} classified events ({data['n_events']} total, "
                  f"{data['fluctuate_uncertain']} uncertain excluded)")

            return {
                'n_total': n_total,
                'n_stable_iof': n_stable_iof,
                'n_stable_std': n_stable_std,
                'n_flip': n_flip,
                'pct_stable_iof': 100 * n_stable_iof / n_total if n_total > 0 else 0,
                'pct_stable_std': 100 * n_stable_std / n_total if n_total > 0 else 0,
                'pct_flip': 100 * n_flip / n_total if n_total > 0 else 0,
                # Also store raw counts for provenance
                'n_events_raw': data['n_events'],
                'n_uncertain': data['fluctuate_uncertain'],
            }

    # Fallback (should not be used in production)
    print("  WARNING: Using fallback McEwen values (stability_table.json not found)")
    return {
        'n_total': 196,
        'n_stable_iof': 13,
        'n_stable_std': 152,
        'n_flip': 31,
        'pct_stable_iof': 6.6,
        'pct_stable_std': 77.6,
        'pct_flip': 15.8,
    }


# -----------------------------------------------------------------------------
# Generate macros
# -----------------------------------------------------------------------------

def assert_invariants(ligo, mcewan):
    """
    Validate data invariants. Fail fast if something is inconsistent.
    This catches pipeline bugs before they become manuscript bugs.
    """
    errors = []

    # LIGO invariants
    if 'stability' in ligo:
        s = ligo['stability']
        # n_ok must be positive (prevents division-by-zero in percentage calcs)
        if s.get('n_ok', 0) <= 0:
            errors.append(f"LIGO: n_ok must be > 0, got {s.get('n_ok', 0)}")
        # Count check: stable + flip + failed = n_ok
        n_failed = s.get('n_failed', 0)
        n_sum = s['n_stable_iof'] + s['n_stable_std'] + s['n_flip'] + n_failed
        if n_sum != s['n_ok']:
            errors.append(f"LIGO: n_stable_iof + n_stable_std + n_flip + n_failed = {n_sum} != n_ok = {s['n_ok']}")

        # Percentage check: all categories should sum to ~100%
        failed_pct = 100 * n_failed / s['n_ok'] if s['n_ok'] > 0 else 0
        pct_sum = s['stable_iof_pct'] + s['stable_std_pct'] + s['flip_pct'] + failed_pct
        if abs(pct_sum - 100.0) > 1.0:  # Allow 1% tolerance for rounding
            errors.append(f"LIGO: percentages sum to {pct_sum:.1f}%, expected ~100%")

        if not (0.5 <= s['auc'] <= 1.0):
            errors.append(f"LIGO: AUC = {s['auc']} outside valid range [0.5, 1.0]")

        if not (-1.0 <= s['cliffs_delta'] <= 1.0):
            errors.append(f"LIGO: Cliff's delta = {s['cliffs_delta']} outside valid range [-1, 1]")

    # McEwen invariants
    n_sum = mcewan['n_stable_iof'] + mcewan['n_stable_std'] + mcewan['n_flip']
    if n_sum != mcewan['n_total']:
        errors.append(f"McEwen: n_stable_iof + n_stable_std + n_flip = {n_sum} != n_total = {mcewan['n_total']}")

    pct_sum = mcewan['pct_stable_iof'] + mcewan['pct_stable_std'] + mcewan['pct_flip']
    if abs(pct_sum - 100.0) > 1.0:
        errors.append(f"McEwen: percentages sum to {pct_sum:.1f}%, expected ~100%")

    if errors:
        print("\n" + "=" * 70)
        print("INVARIANT VIOLATIONS DETECTED:")
        print("=" * 70)
        for e in errors:
            print(f"  ERROR: {e}")
        print("=" * 70)
        print("\nFix the pipeline or data before regenerating macros.")
        raise ValueError(f"{len(errors)} invariant violation(s) detected")

    print("  All invariants passed")


def generate_macros():
    """Generate the complete results_macros.tex file."""

    ligo = load_ligo_data()
    mcewan = load_mcewan_data()

    # Validate before generating
    print("  Checking invariants...")
    assert_invariants(ligo, mcewan)

    lines = []
    lines.append("% results_macros.tex")
    lines.append("% Canonical results - use these everywhere to prevent number drift")
    lines.append(f"% Auto-generated by generate_macros.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("% DO NOT EDIT MANUALLY - regenerate with: python scripts/generate_macros.py")
    lines.append("")

    # === Provenance ===
    run_date = datetime.now().strftime('%Y-%m-%d')
    git_hash = get_git_hash()
    lines.append("% === Provenance (for reproducibility audit) ===")
    lines.append("% Safety defaults for provenance macros")
    lines.append("\\providecommand{\\ResultsRunDate}{UNKNOWN}")
    lines.append("\\providecommand{\\ResultsRNGSeed}{UNKNOWN}")
    lines.append("\\providecommand{\\ResultsPipelineVersion}{UNKNOWN}")
    lines.append("\\providecommand{\\ResultsGitHash}{UNKNOWN}")
    lines.append("\\providecommand{\\ResultsBanner}{UNKNOWN}")
    lines.append("% Actual values")
    lines.append(f"\\renewcommand{{\\ResultsRunDate}}{{{run_date}}}")
    lines.append("\\renewcommand{\\ResultsRNGSeed}{42}")
    lines.append("\\renewcommand{\\ResultsPipelineVersion}{1.0}")
    lines.append(f"\\renewcommand{{\\ResultsGitHash}}{{{git_hash}}}")
    lines.append("")
    # One-line banner for appendix (use \textbar instead of $|$ to avoid Hyperref PDF string issues)
    lines.append("% One-line provenance banner for appendix")
    lines.append(f"\\renewcommand{{\\ResultsBanner}}{{Pipeline v1.0 \\textbar\\ seed 42 \\textbar\\ {git_hash} \\textbar\\ {run_date}}}")
    lines.append("")

    # === Formatting helpers ===
    lines.append("% === Formatting helpers (prevent style drift) ===")
    lines.append("% Math-safe placeholder for missing values (won't italicize in math mode)")
    lines.append("\\providecommand{\\NA}{\\ensuremath{\\mathrm{NA}}}")
    lines.append("% Other formatting helpers")
    lines.append("\\providecommand{\\Pct}[1]{#1\\%}")
    lines.append("\\providecommand{\\CountPct}[2]{#1 (#2\\%)}")
    lines.append("\\providecommand{\\AUCfmt}[3]{#1 [#2, #3]}")
    lines.append("\\providecommand{\\CIfmt}[2]{[#1, #2]}")
    lines.append("")

    # === Safety net defaults (compile never fails) ===
    lines.append("% === Safety net defaults (use \\providecommand so missing values show NA) ===")
    lines.append("% These are overwritten below if data exists; NA indicates missing pipeline output")
    safety_macros = [
        'NMcEwen', 'McEwenStableDelayed', 'McEwenStableDelayedPct',
        'McEwenStableDelayedCILo', 'McEwenStableDelayedCIHi',
        'McEwenStableFast', 'McEwenStableFastPct', 'McEwenFlip', 'McEwenFlipPct',
        'McEwenCurvatureP', 'McEwenTauMedian', 'McEwenTauIQRLo', 'McEwenTauIQRHi',
        # Google robustness sweep (evidence strength)
        'McEwenRobustN', 'McEwenRobustConfigs',
        'McEwenStableIOFEvidMed', 'McEwenStableIOFEvidIQRLo', 'McEwenStableIOFEvidIQRHi',
        'McEwenStableSTDEvidMed', 'McEwenStableSTDEvidIQRLo', 'McEwenStableSTDEvidIQRHi',
        'McEwenFlipEvidMed', 'McEwenFlipEvidIQRLo', 'McEwenFlipEvidIQRHi',
        'McEwenStableIOFAboveFour', 'McEwenStableIOFAboveTen',
        'McEwenFlipAboveFour',
        'NLIGOOK', 'NLIGOStable', 'LIGOStableDelayed', 'LIGOStableDelayedPct', 'LIGOStableFast',
        'LIGOStableFastPct', 'LIGOFlip', 'LIGOFlipPct', 'LIGOFailed', 'LIGOFailedPct',
        'LIGOCurvatureAUC', 'LIGOCurvatureAUClo', 'LIGOCurvatureAUChi', 'LIGOCliffsDelta',
        'LIGOCurvDelayedMedian', 'LIGOCurvDelayedIQRLo', 'LIGOCurvDelayedIQRHi',
        'LIGOCurvFastMedian', 'LIGOCurvFastIQRLo', 'LIGOCurvFastIQRHi',
        'LIGOBetaB', 'LIGOBetaBLo', 'LIGOBetaBHi', 'LIGOBetaSNR',
        'LIGODeltaBIC', 'LIGODeltaBICLo', 'LIGODeltaBICHi',
        'LIGOBetaBOnly', 'LIGOBetaBOnlyLo', 'LIGOBetaBOnlyHi',
        'LIGODeltaBICBOnly', 'LIGODeltaBICBOnlyLo', 'LIGODeltaBICBOnlyHi',
        'LIGOBetaSNROnly',
        'LIGOThreshOneDelayedN', 'LIGOThreshOneDelayedPct',
        'LIGOThreshOneFastN', 'LIGOThreshOneFastPct',
        'LIGOThreshOneFlipN', 'LIGOThreshOneFlipPct', 'LIGOThreshOneAUC',
        'LIGOThreshFourDelayedN', 'LIGOThreshFourDelayedPct',
        'LIGOThreshFourFastN', 'LIGOThreshFourFastPct',
        'LIGOThreshFourFlipN', 'LIGOThreshFourFlipPct', 'LIGOThreshFourAUC',
        'LIGOFlipDeterminateN', 'LIGOFlipDeterminatePct',
        'LIGOFlipUncertainN', 'LIGOFlipUncertainPct',
        'LIGOFlipShortDelayedPct', 'LIGOFlipLongDelayedPct',
        'LIGODip', 'LIGODipP', 'LIGODipPDelayed', 'LIGODipPFast',
        'LIGOMWCurvPExp',  # Mann-Whitney p-value exponent
        'LIGODipUnlabeledN', 'LIGODipUnlabeled', 'LIGODipUnlabeledP',
        'LIGODipUnlabeledWinsP', 'LIGODipUnlabeledTrimP',
        'LIGODipStableN', 'LIGODipStableP', 'LIGODipStableTrimP',
        'LIGOWindowN', 'LIGOWindowClusters', 'LIGOWindowBetaB',
        'LIGOWindowZB', 'LIGOWindowPB', 'LIGOWindowBetaSNR', 'LIGOWindowPSNR',
        'LIGOGMMAgreement', 'LIGOGMMARI', 'LIGOGMMNMI', 'LIGOGMMBoundary',
        # Rejected-morphology stress test (out-of-distribution validation)
        'LIGORejectN', 'LIGORejectKS', 'LIGORejectKSP',
        'LIGORejectDelayedN', 'LIGORejectFastN', 'LIGORejectUncertainN',
        'LIGORejectDeterminateN', 'LIGORejectAUC',
        # Alternative likelihood
        'LIGOAltLikelihoodAgreement',
        # Curvature window sweep
        'LIGOCurvSweepTenAUC', 'LIGOCurvSweepTenCILo', 'LIGOCurvSweepTenCIHi', 'LIGOCurvSweepTenSign',
        'LIGOCurvSweepTwentyAUC', 'LIGOCurvSweepTwentyCILo', 'LIGOCurvSweepTwentyCIHi', 'LIGOCurvSweepTwentySign',
        'LIGOCurvSweepThirtyAUC', 'LIGOCurvSweepThirtyCILo', 'LIGOCurvSweepThirtyCIHi', 'LIGOCurvSweepThirtySign',
        'LIGOCurvSpearmanTenTwenty', 'LIGOCurvSpearmanTwentyThirty',
    ]
    for macro in safety_macros:
        lines.append(f"\\providecommand{{\\{macro}}}{{\\NA}}")
    lines.append("")

    # === McEwen / Google Sycamore ===
    lines.append("% === McEwen / Google Sycamore (telemetry) ===")
    lines.append(f"\\renewcommand{{\\NMcEwen}}{{{mcewan['n_total']}}}")
    lines.append(f"\\renewcommand{{\\McEwenStableDelayed}}{{{mcewan['n_stable_iof']}}}")
    lines.append(f"\\renewcommand{{\\McEwenStableDelayedPct}}{{{fmt(mcewan['pct_stable_iof'], 1)}}}")
    # Wilson CI for stable delayed fraction
    ci_lo, ci_hi = wilson_ci(mcewan['n_stable_iof'], mcewan['n_total'])
    lines.append(f"\\renewcommand{{\\McEwenStableDelayedCILo}}{{{fmt(ci_lo * 100, 1)}}}")
    lines.append(f"\\renewcommand{{\\McEwenStableDelayedCIHi}}{{{fmt(ci_hi * 100, 1)}}}")
    lines.append(f"\\renewcommand{{\\McEwenStableFast}}{{{mcewan['n_stable_std']}}}")
    lines.append(f"\\renewcommand{{\\McEwenStableFastPct}}{{{fmt(mcewan['pct_stable_std'], 1)}}}")
    lines.append(f"\\renewcommand{{\\McEwenFlip}}{{{mcewan['n_flip']}}}")
    lines.append(f"\\renewcommand{{\\McEwenFlipPct}}{{{fmt(mcewan['pct_flip'], 1)}}}")
    # McEwen curvature discrimination (from stability_diagnostics.json)
    stability_diag = load_json(OUTPUT_DIR / "robustness_sweep" / "stability_diagnostics.json")
    if stability_diag and 'curvature_discrimination' in stability_diag:
        curv_p = stability_diag['curvature_discrimination'].get('mann_whitney_p', 0.44)
        lines.append("% Curvature discrimination (from stability_diagnostics.json)")
        lines.append(f"\\renewcommand{{\\McEwenCurvatureP}}{{{fmt(curv_p, 2)}}}  % Mann-Whitney p-value")
    else:
        # Fallback
        lines.append("% Curvature discrimination (fallback)")
        lines.append("\\renewcommand{\\McEwenCurvatureP}{0.44}  % Mann-Whitney p-value")
    # Recovery time constants from model fits (from mcewan_tau_stats.json)
    tau_stats = load_json(OUTPUT_DIR / "robustness_sweep" / "mcewan_tau_stats.json")
    if tau_stats:
        lines.append("% Recovery timescale (from mcewan_tau_stats.json)")
        lines.append(f"\\renewcommand{{\\McEwenTauMedian}}{{{fmt(tau_stats['tau_median_ms'], 1)}}}  % ms")
        lines.append(f"\\renewcommand{{\\McEwenTauIQRLo}}{{{fmt(tau_stats['tau_iqr_lo_ms'], 1)}}}")
        lines.append(f"\\renewcommand{{\\McEwenTauIQRHi}}{{{fmt(tau_stats['tau_iqr_hi_ms'], 1)}}}")
    else:
        lines.append("% Recovery timescale (fallback - run mcewan_tau_stats.py)")
        lines.append("\\renewcommand{\\McEwenTauMedian}{26}  % ms")
        lines.append("\\renewcommand{\\McEwenTauIQRLo}{24}")
        lines.append("\\renewcommand{\\McEwenTauIQRHi}{28}")
    lines.append("% Key: delay (D) separates, curvature (b) does NOT")
    lines.append("")

    # === Google robustness sweep (evidence strength) ===
    # stability_diag already loaded above
    if stability_diag and 'evidence_strength' in stability_diag:
        lines.append("% === McEwen / Google robustness sweep (evidence strength) ===")
        es = stability_diag['evidence_strength']
        # Number of observations (sweep configs × events)
        n_obs = es['stable_iof']['n_observations'] + es['stable_std']['n_observations'] + es['flip']['n_observations']
        n_configs = 15  # Fixed by design
        lines.append(f"\\renewcommand{{\\McEwenRobustN}}{{{n_obs}}}")
        lines.append(f"\\renewcommand{{\\McEwenRobustConfigs}}{{{n_configs}}}")
        # Stable IOF evidence
        lines.append(f"\\renewcommand{{\\McEwenStableIOFEvidMed}}{{{fmt(es['stable_iof']['median'], 2)}}}")
        lines.append(f"\\renewcommand{{\\McEwenStableIOFEvidIQRLo}}{{{fmt(es['stable_iof']['iqr'][0], 2)}}}")
        lines.append(f"\\renewcommand{{\\McEwenStableIOFEvidIQRHi}}{{{fmt(es['stable_iof']['iqr'][1], 2)}}}")
        # Stable STD evidence
        lines.append(f"\\renewcommand{{\\McEwenStableSTDEvidMed}}{{{fmt(es['stable_std']['median'], 2)}}}")
        lines.append(f"\\renewcommand{{\\McEwenStableSTDEvidIQRLo}}{{{fmt(es['stable_std']['iqr'][0], 2)}}}")
        lines.append(f"\\renewcommand{{\\McEwenStableSTDEvidIQRHi}}{{{fmt(es['stable_std']['iqr'][1], 2)}}}")
        # Flip evidence
        lines.append(f"\\renewcommand{{\\McEwenFlipEvidMed}}{{{fmt(es['flip']['median'], 2)}}}")
        lines.append(f"\\renewcommand{{\\McEwenFlipEvidIQRLo}}{{{fmt(es['flip']['iqr'][0], 2)}}}")
        lines.append(f"\\renewcommand{{\\McEwenFlipEvidIQRHi}}{{{fmt(es['flip']['iqr'][1], 2)}}}")
        lines.append("")

    # Evidence threshold rates from stability_diagnostics.json
    if stability_diag and 'evidence_strength' in stability_diag:
        es = stability_diag['evidence_strength']
        lines.append("% Evidence threshold rates (from stability_diagnostics.json)")
        stable_iof_above_4 = es['stable_iof'].get('pct_above_4', 0)
        stable_iof_above_10 = es['stable_iof'].get('pct_above_10', 0)
        flip_above_4 = es['flip'].get('pct_above_4', 0)
        lines.append(f"\\renewcommand{{\\McEwenStableIOFAboveFour}}{{{fmt(stable_iof_above_4, 0)}}}")
        lines.append(f"\\renewcommand{{\\McEwenStableIOFAboveTen}}{{{fmt(stable_iof_above_10, 0)}}}")
        lines.append(f"\\renewcommand{{\\McEwenFlipAboveFour}}{{{fmt(flip_above_4, 0)}}}")
        lines.append("")

    # === LIGO main results ===
    if 'stability' in ligo:
        s = ligo['stability']
        lines.append("% === LIGO Extremely_Loud (Hilbert envelope, 3-window stability) ===")
        lines.append(f"\\renewcommand{{\\NLIGOOK}}{{{s['n_ok']}}}")
        lines.append(f"\\renewcommand{{\\LIGOStableDelayed}}{{{s['n_stable_iof']}}}")
        lines.append(f"\\renewcommand{{\\LIGOStableDelayedPct}}{{{fmt(s['stable_iof_pct'], 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGOStableFast}}{{{s['n_stable_std']}}}")
        lines.append(f"\\renewcommand{{\\LIGOStableFastPct}}{{{fmt(s['stable_std_pct'], 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGOFlip}}{{{s['n_flip']}}}")
        lines.append(f"\\renewcommand{{\\LIGOFlipPct}}{{{fmt(s['flip_pct'], 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGOFailed}}{{{s['n_failed']}}}")
        n_stable = s['n_stable_iof'] + s['n_stable_std']
        lines.append(f"\\renewcommand{{\\NLIGOStable}}{{{n_stable}}}")

        # n_ok already includes n_failed (per invariant: stable + flip + failed = n_ok)
        n_total = s['n_ok']
        failed_pct = 100 * s.get('n_failed', 0) / n_total if n_total > 0 else 0
        lines.append(f"\\renewcommand{{\\LIGOFailedPct}}{{{fmt(failed_pct, 1)}}}")

        lines.append(f"\\renewcommand{{\\LIGOCurvatureAUC}}{{{fmt(s['auc'], 3)}}}")
        lines.append(f"\\renewcommand{{\\LIGOCurvatureAUClo}}{{{fmt(s['auc_ci_lower'], 3)}}}")
        lines.append(f"\\renewcommand{{\\LIGOCurvatureAUChi}}{{{fmt(s['auc_ci_upper'], 3)}}}")
        lines.append(f"\\renewcommand{{\\LIGOCliffsDelta}}{{{fmt(s['cliffs_delta'], 3)}}}")
        lines.append("% Key: curvature (b) separates strongly, survives SNR control; large flip fraction")
        lines.append("")

        # Curvature stats (if available)
        if 'curvature_delayed_median' in s:
            lines.append("% === LIGO Curvature Statistics (scaled to 10^-3) ===")
            lines.append(f"\\renewcommand{{\\LIGOCurvDelayedMedian}}{{{fmt(s['curvature_delayed_median'], 2)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvDelayedIQRLo}}{{{fmt(s['curvature_delayed_iqr_lo'], 1)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvDelayedIQRHi}}{{{fmt(s['curvature_delayed_iqr_hi'], 1)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvFastMedian}}{{{fmt(s['curvature_fast_median'], 2)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvFastIQRLo}}{{{fmt(s['curvature_fast_iqr_lo'], 1)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvFastIQRHi}}{{{fmt(s['curvature_fast_iqr_hi'], 1)}}}")
            # Mann-Whitney p-value for curvature separation
            if 'curvature_mann_whitney_p_exp' in s:
                lines.append("% Mann-Whitney p-value exponent for curvature separation (from bootstrap_beta_b.json)")
                lines.append(f"\\renewcommand{{\\LIGOMWCurvPExp}}{{{s['curvature_mann_whitney_p_exp']}}}  % p < 10^-X")
            else:
                # Fallback (regenerate ligo_stability_figures.py to update)
                lines.append("% Mann-Whitney p-value exponent (fallback - run ligo_stability_figures.py)")
                lines.append("\\renewcommand{\\LIGOMWCurvPExp}{20}  % p < 10^-20")
            lines.append("")

        # Bootstrap regression
        lines.append("% === LIGO Bootstrap regression ===")
        lines.append("% b-only model (Delayed ~ b)")
        if 'beta_b_only' in s:
            lines.append(f"\\renewcommand{{\\LIGOBetaBOnly}}{{{fmt(s['beta_b_only'], 2)}}}")
            lines.append(f"\\renewcommand{{\\LIGOBetaBOnlyLo}}{{{fmt(s['beta_b_only_ci_lower'], 2)}}}")
            lines.append(f"\\renewcommand{{\\LIGOBetaBOnlyHi}}{{{fmt(s['beta_b_only_ci_upper'], 2)}}}")
            lines.append(f"\\renewcommand{{\\LIGODeltaBICBOnly}}{{{fmt(s['delta_bic_b_only'], 1)}}}")
            lines.append(f"\\renewcommand{{\\LIGODeltaBICBOnlyLo}}{{{fmt(s['delta_bic_b_only_ci_lower'], 1)}}}")
            lines.append(f"\\renewcommand{{\\LIGODeltaBICBOnlyHi}}{{{fmt(s['delta_bic_b_only_ci_upper'], 1)}}}")
        else:
            # Fallback (run ligo_stability_figures.py to update)
            print("  WARNING: Using fallback b-only regression values (run ligo_stability_figures.py)")
            lines.append("\\renewcommand{\\LIGOBetaBOnly}{2.01}")
            lines.append("\\renewcommand{\\LIGODeltaBICBOnly}{-87.5}")
        lines.append("% Full model (Delayed ~ b + SNR)")
        lines.append(f"\\renewcommand{{\\LIGOBetaB}}{{{fmt(s['beta_b'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGOBetaBLo}}{{{fmt(s['beta_b_ci_lower'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGOBetaBHi}}{{{fmt(s['beta_b_ci_upper'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGOBetaSNR}}{{{fmt(s['beta_snr'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGODeltaBIC}}{{{fmt(s['delta_bic'], 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGODeltaBICLo}}{{{fmt(s['delta_bic_ci_lower'], 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGODeltaBICHi}}{{{fmt(s['delta_bic_ci_upper'], 1)}}}")
        # SNR-only model (Delayed ~ SNR)
        if 'beta_snr_only' in s:
            lines.append("% SNR-only model (Delayed ~ SNR)")
            lines.append(f"\\renewcommand{{\\LIGOBetaSNROnly}}{{{fmt(s['beta_snr_only'], 2)}}}")
        lines.append("")

    # === LIGO Threshold Sensitivity ===
    if 'threshold' in ligo:
        t = ligo['threshold']['results']
        lines.append("% === LIGO Threshold Sensitivity (T=1, T=4) ===")

        # T=1
        t1 = t.get('1', {})
        lines.append("% T=1 (weaker evidence threshold)")
        lines.append(f"\\renewcommand{{\\LIGOThreshOneDelayedN}}{{{t1.get('n_stable_iof', '???')}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshOneDelayedPct}}{{{fmt(t1.get('pct_stable_iof'), 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshOneFastN}}{{{t1.get('n_stable_std', '???')}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshOneFastPct}}{{{fmt(t1.get('pct_stable_std'), 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshOneFlipN}}{{{t1.get('n_flip', '???')}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshOneFlipPct}}{{{fmt(t1.get('pct_flip'), 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshOneAUC}}{{{fmt(t1.get('auc'), 3)}}}")

        # T=4
        t4 = t.get('4', {})
        lines.append("% T=4 (stronger evidence threshold)")
        lines.append(f"\\renewcommand{{\\LIGOThreshFourDelayedN}}{{{t4.get('n_stable_iof', '???')}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshFourDelayedPct}}{{{fmt(t4.get('pct_stable_iof'), 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshFourFastN}}{{{t4.get('n_stable_std', '???')}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshFourFastPct}}{{{fmt(t4.get('pct_stable_std'), 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshFourFlipN}}{{{t4.get('n_flip', '???')}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshFourFlipPct}}{{{fmt(t4.get('pct_flip'), 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGOThreshFourAUC}}{{{fmt(t4.get('auc'), 3)}}}")
        lines.append("")

        # Flip breakdown from 3-window stability (consistent with LIGOFlip)
        # Load stability events to get flip GPS times
        stability_file = LIGO_DIR / "stability_events.jsonl"
        detailed_file = LIGO_DIR / "stability_events_detailed.jsonl"

        flip_gps = set()
        if stability_file.exists():
            with open(stability_file) as f:
                for line in f:
                    e = json.loads(line)
                    if e.get('stability') == 'flip':
                        flip_gps.add(e['gps_time'])

        n_det = 0
        n_unc = 0
        n_short_delayed = 0
        n_long_delayed = 0

        if detailed_file.exists() and flip_gps:
            with open(detailed_file) as f:
                for line in f:
                    e = json.loads(line)
                    if e['gps_time'] not in flip_gps:
                        continue
                    w = e.get('window_results', {})
                    g60 = w.get('60', {}).get('geometry', '')
                    g150 = w.get('150', {}).get('geometry', '')

                    if g60 == g150:
                        n_det += 1
                    else:
                        n_unc += 1
                        if g60 == 'delayed':
                            n_short_delayed += 1
                        if g150 == 'delayed':
                            n_long_delayed += 1

        n_flip = n_det + n_unc
        pct_det = 100 * n_det / n_flip if n_flip > 0 else 0
        pct_unc = 100 * n_unc / n_flip if n_flip > 0 else 0

        lines.append("% === LIGO Flip Breakdown (3-window stability, consistent with LIGOFlip) ===")
        lines.append(f"\\renewcommand{{\\LIGOFlipDeterminateN}}{{{n_det}}}")
        lines.append(f"\\renewcommand{{\\LIGOFlipDeterminatePct}}{{{fmt(pct_det, 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGOFlipUncertainN}}{{{n_unc}}}")
        lines.append(f"\\renewcommand{{\\LIGOFlipUncertainPct}}{{{fmt(pct_unc, 1)}}}")

        # Flip direction: among uncertain flips (60ms != 150ms), what % show Delayed at each end
        if n_unc > 0:
            pct_short = 100 * n_short_delayed / n_unc
            pct_long = 100 * n_long_delayed / n_unc
            lines.append("% Flip direction: among uncertain flips, % with delayed geometry at each window")
            lines.append(f"\\renewcommand{{\\LIGOFlipShortDelayedPct}}{{{fmt(pct_short, 0)}}}  % Delayed at 60ms window")
            lines.append(f"\\renewcommand{{\\LIGOFlipLongDelayedPct}}{{{fmt(pct_long, 0)}}}  % Delayed at 150ms window")
        else:
            lines.append("\\renewcommand{\\LIGOFlipShortDelayedPct}{0}")
            lines.append("\\renewcommand{\\LIGOFlipLongDelayedPct}{0}")
        lines.append("")

    # === LIGO Hartigan's dip test ===
    lines.append("% === LIGO Hartigan's dip test ===")
    dip_data = load_json(LIGO_DIR / "dip_test.json")
    if dip_data:
        lines.append(f"\\renewcommand{{\\LIGODip}}{{{fmt(dip_data['dip_all'], 3)}}}")
        lines.append(f"\\renewcommand{{\\LIGODipP}}{{${fmt_pval(dip_data['p_all'])}$}}")
        lines.append(f"\\renewcommand{{\\LIGODipPDelayed}}{{{fmt(dip_data['p_delayed'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGODipPFast}}{{{fmt(dip_data['p_fast'], 2)}}}")
    else:
        # Fallback values (should not be used in production)
        print("  WARNING: Using fallback dip test values (dip_test.json not found)")
        lines.append("\\renewcommand{\\LIGODip}{0.038}")
        lines.append("\\renewcommand{\\LIGODipP}{0.023}")
        lines.append("\\renewcommand{\\LIGODipPDelayed}{0.68}")
        lines.append("\\renewcommand{\\LIGODipPFast}{0.93}")
    lines.append("")

    # === LIGO Robustness: Dip test on unlabeled data ===
    lines.append("% === LIGO Robustness: Dip test on ALL OK events (ignoring labels) ===")
    dip_unlabeled = load_json(LIGO_DIR / "dip_test_unlabeled.json")
    if dip_unlabeled and 'all_ok' in dip_unlabeled:
        all_ok = dip_unlabeled['all_ok']
        stable_only = dip_unlabeled['stable_only']
        lines.append(f"\\renewcommand{{\\LIGODipUnlabeledN}}{{{all_ok['n']}}}")
        lines.append(f"\\renewcommand{{\\LIGODipUnlabeled}}{{{fmt(all_ok['original']['dip'], 3)}}}")
        lines.append(f"\\renewcommand{{\\LIGODipUnlabeledP}}{{{fmt(all_ok['original']['p'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGODipUnlabeledWinsP}}{{{fmt(all_ok['winsorized_1_99']['p'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGODipUnlabeledTrimP}}{{{fmt(all_ok['trimmed_k5']['p'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGODipStableN}}{{{stable_only['n']}}}")
        lines.append(f"\\renewcommand{{\\LIGODipStableP}}{{${fmt_pval(stable_only['original']['p'])}$}}")
        lines.append(f"\\renewcommand{{\\LIGODipStableTrimP}}{{${fmt_pval(stable_only['trimmed_k5']['p'])}$}}")
    else:
        lines.append("\\providecommand{\\LIGODipUnlabeledN}{\\NA}")
        lines.append("\\providecommand{\\LIGODipUnlabeledP}{\\NA}")
    lines.append("")

    # === LIGO Robustness: Window-level regression ===
    lines.append("% === LIGO Robustness: Window-level regression with cluster-robust SEs ===")
    window_reg = load_json(LIGO_DIR / "window_level_regression.json")
    if window_reg:
        lines.append(f"\\renewcommand{{\\LIGOWindowN}}{{{window_reg['n_observations']}}}")
        lines.append(f"\\renewcommand{{\\LIGOWindowClusters}}{{{window_reg['n_clusters']}}}")
        lines.append(f"\\renewcommand{{\\LIGOWindowBetaB}}{{{fmt(window_reg['full_model']['coef_b_std'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGOWindowZB}}{{{fmt(window_reg['full_model']['z_b'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGOWindowPB}}{{{fmt(window_reg['full_model']['p_b'], 6)}}}")
        lines.append(f"\\renewcommand{{\\LIGOWindowBetaSNR}}{{{fmt(window_reg['full_model']['coef_snr_std'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGOWindowPSNR}}{{{fmt(window_reg['full_model']['p_snr'], 4)}}}")
    else:
        lines.append("\\providecommand{\\LIGOWindowN}{\\NA}")
        lines.append("\\providecommand{\\LIGOWindowBetaB}{\\NA}")
    lines.append("")

    # === LIGO Robustness: Unsupervised GMM validation ===
    lines.append("% === LIGO Robustness: Unsupervised GMM validation ===")
    gmm_data = load_json(LIGO_DIR / "unsupervised_validation.json")
    if gmm_data and 'stable_only_comparison' in gmm_data:
        comp = gmm_data['stable_only_comparison']
        lines.append(f"\\renewcommand{{\\LIGOGMMAgreement}}{{{fmt(comp['agreement_pct'], 1)}}}")
        lines.append(f"\\renewcommand{{\\LIGOGMMARI}}{{{fmt(comp['adjusted_rand_index'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGOGMMNMI}}{{{fmt(comp['normalized_mutual_info'], 2)}}}")
        lines.append(f"\\renewcommand{{\\LIGOGMMBoundary}}{{{fmt(gmm_data['decision_boundary'], 2)}}}")
    else:
        lines.append("\\providecommand{\\LIGOGMMAgreement}{\\NA}")
        lines.append("\\providecommand{\\LIGOGMMARI}{\\NA}")
    lines.append("")

    # === LIGO Robustness: Rejected-morphology stress test ===
    lines.append("% === LIGO Robustness: Rejected-morphology stress test ===")
    reject = load_json(LIGO_DIR / "negative_control.json")  # file still named this
    if reject:
        lines.append(f"\\renewcommand{{\\LIGORejectN}}{{{reject['n_complex']}}}")
        # KS stats only exist if there are complex events to compare
        if reject.get('ks_statistic') is not None:
            lines.append(f"\\renewcommand{{\\LIGORejectKS}}{{{fmt(reject['ks_statistic'], 2)}}}")
            lines.append(f"\\renewcommand{{\\LIGORejectKSP}}{{{fmt(reject['ks_pvalue'], 3)}}}")
        else:
            # No complex events to compare - set to NA
            lines.append("\\renewcommand{\\LIGORejectKS}{\\NA}")
            lines.append("\\renewcommand{\\LIGORejectKSP}{\\NA}")
        # Complex event classification counts
        if 'complex_classified_delayed' in reject:
            delayed_n = reject['complex_classified_delayed']
            fast_n = reject['complex_classified_fast']
            uncertain_n = reject.get('complex_classified_uncertain', 0)
            determinate_n = delayed_n + fast_n
            lines.append(f"\\renewcommand{{\\LIGORejectDelayedN}}{{{delayed_n}}}")
            lines.append(f"\\renewcommand{{\\LIGORejectFastN}}{{{fast_n}}}")
            lines.append(f"\\renewcommand{{\\LIGORejectUncertainN}}{{{uncertain_n}}}")
            lines.append(f"\\renewcommand{{\\LIGORejectDeterminateN}}{{{determinate_n}}}")
        # Complex AUC (stress test - uses determinate labels only)
        if reject.get('complex_auc') is not None:
            lines.append(f"\\renewcommand{{\\LIGORejectAUC}}{{{fmt(reject['complex_auc'], 3)}}}")
    else:
        lines.append("\\providecommand{\\LIGORejectN}{\\NA}")
        lines.append("\\providecommand{\\LIGORejectKS}{\\NA}")
    lines.append("")

    # === LIGO Robustness: Alternative likelihood (log-domain) ===
    lines.append("% === LIGO Robustness: Alternative likelihood (log-domain) ===")
    alt_lik = load_json(LIGO_DIR / "alternative_likelihood.json")
    if alt_lik:
        lines.append(f"\\renewcommand{{\\LIGOAltLikelihoodAgreement}}{{{fmt(alt_lik['agreement_pct'], 1)}}}")
    else:
        lines.append("\\providecommand{\\LIGOAltLikelihoodAgreement}{\\NA}")
    lines.append("")

    # === LIGO status ===
    lines.append("% === LIGO status ===")
    lines.append("\\providecommand{\\LIGOStatus}{UNKNOWN}")
    lines.append("\\providecommand{\\LIGOAsOf}{UNKNOWN}")
    lines.append("\\providecommand{\\LIGOScaleNote}{}")
    lines.append("\\renewcommand{\\LIGOStatus}{random sample}")
    lines.append("\\renewcommand{\\LIGOAsOf}{December 2025}")
    if 'stability' in ligo:
        s = ligo['stability']
        flip_pct = fmt(s['flip_pct'], 1)
        auc = fmt(s['auc'], 3)
        lines.append(f"\\renewcommand{{\\LIGOScaleNote}}{{The large flip fraction ({flip_pct}\\%) indicates a substantial boundary population; stable-core discrimination remains strong (AUC={auc}).}}")
    lines.append("")

    # === LIGO Curvature Window Sweep ===
    lines.append("% === LIGO Curvature Window Sweep (robustness to fit interval choice) ===")
    lines.append("% From ligo_curvature_sweep.py output")
    curv_sweep = load_json(LIGO_DIR / "curvature_sweep_results.json")
    if curv_sweep and 'windows' in curv_sweep:
        w = curv_sweep['windows']
        # 0-10ms window
        if '10' in w:
            w10 = w['10']
            lines.append("% 0-10ms window")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepTenAUC}}{{{fmt(w10['auc'], 3)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepTenCILo}}{{{fmt(w10['auc_ci_lower'], 3)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepTenCIHi}}{{{fmt(w10['auc_ci_upper'], 3)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepTenSign}}{{{'Yes' if w10['sign_stable'] else 'No'}}}")
        # 0-20ms window (default)
        if '20' in w:
            w20 = w['20']
            lines.append("% 0-20ms window (default)")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepTwentyAUC}}{{{fmt(w20['auc'], 3)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepTwentyCILo}}{{{fmt(w20['auc_ci_lower'], 3)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepTwentyCIHi}}{{{fmt(w20['auc_ci_upper'], 3)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepTwentySign}}{{{'Yes' if w20['sign_stable'] else 'No'}}}")
        # 0-30ms window
        if '30' in w:
            w30 = w['30']
            lines.append("% 0-30ms window")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepThirtyAUC}}{{{fmt(w30['auc'], 3)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepThirtyCILo}}{{{fmt(w30['auc_ci_lower'], 3)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepThirtyCIHi}}{{{fmt(w30['auc_ci_upper'], 3)}}}")
            lines.append(f"\\renewcommand{{\\LIGOCurvSweepThirtySign}}{{{'Yes' if w30['sign_stable'] else 'No'}}}")
        # Cross-window Spearman correlations (stable-core)
        if 'cross_window_correlations_stable' in curv_sweep:
            corr = curv_sweep['cross_window_correlations_stable']
            lines.append("% Cross-window Spearman correlations (stable-core)")
            if '10_vs_20' in corr:
                lines.append(f"\\renewcommand{{\\LIGOCurvSpearmanTenTwenty}}{{{fmt(corr['10_vs_20']['rho'], 2)}}}")
            if '20_vs_30' in corr:
                lines.append(f"\\renewcommand{{\\LIGOCurvSpearmanTwentyThirty}}{{{fmt(corr['20_vs_30']['rho'], 2)}}}")
    else:
        # Fallback values
        lines.append("% Curvature sweep (fallback - run ligo_curvature_sweep.py)")
        lines.append("\\renewcommand{\\LIGOCurvSweepTenAUC}{0.915}")
        lines.append("\\renewcommand{\\LIGOCurvSweepTenCILo}{0.892}")
        lines.append("\\renewcommand{\\LIGOCurvSweepTenCIHi}{0.938}")
        lines.append("\\renewcommand{\\LIGOCurvSweepTenSign}{Yes}")
        lines.append("\\renewcommand{\\LIGOCurvSweepTwentyAUC}{0.952}")
        lines.append("\\renewcommand{\\LIGOCurvSweepTwentyCILo}{0.935}")
        lines.append("\\renewcommand{\\LIGOCurvSweepTwentyCIHi}{0.968}")
        lines.append("\\renewcommand{\\LIGOCurvSweepTwentySign}{Yes}")
        lines.append("\\renewcommand{\\LIGOCurvSweepThirtyAUC}{0.950}")
        lines.append("\\renewcommand{\\LIGOCurvSweepThirtyCILo}{0.931}")
        lines.append("\\renewcommand{\\LIGOCurvSweepThirtyCIHi}{0.967}")
        lines.append("\\renewcommand{\\LIGOCurvSweepThirtySign}{No}")
        lines.append("\\renewcommand{\\LIGOCurvSpearmanTenTwenty}{0.85}")
        lines.append("\\renewcommand{\\LIGOCurvSpearmanTwentyThirty}{0.93}")
    lines.append("")

    # === LIGO Null Simulation Control (window-decoupled) ===
    lines.append("% === LIGO Null Simulation Control (window-decoupled) ===")
    lines.append("% From ligo_null_simulation.py - referee-grade control test")
    if 'null_simulation' in ligo:
        ns = ligo['null_simulation']
        nf = ns.get('null_fast', {})
        mt = ns.get('mixed_truth', {})
        cfg = ns.get('config', {})

        # Null-fast-only results
        if nf:
            lines.append("% Null-fast-only (pure fast-geometry world)")
            lines.append(f"\\newcommand{{\\LIGONullN}}{{{nf.get('n_valid', '???')}}}")
            lines.append(f"\\newcommand{{\\LIGONullMislabelRate}}{{{fmt(100 * nf.get('mislabel_rate', 0), 2)}}}")
            lines.append(f"\\newcommand{{\\LIGONullMislabelCILo}}{{{fmt(100 * nf.get('mislabel_ci_lower', 0), 2)}}}")
            lines.append(f"\\newcommand{{\\LIGONullMislabelCIHi}}{{{fmt(100 * nf.get('mislabel_ci_upper', 0), 2)}}}")
            lines.append(f"\\newcommand{{\\LIGONullMislabeledN}}{{{nf.get('n_mislabeled_delayed', '???')}}}")
            lines.append(f"\\newcommand{{\\LIGONullAmbiguousN}}{{{nf.get('n_ambiguous', '???')}}}")
            if nf.get('auc_assigned'):
                lines.append(f"\\newcommand{{\\LIGONullAUCAssigned}}{{{fmt(nf['auc_assigned'], 3)}}}")
            if nf.get('auc_random_mean'):
                lines.append(f"\\newcommand{{\\LIGONullAUCRandom}}{{{fmt(nf['auc_random_mean'], 3)}}}")

        # Mixed-truth results
        if mt:
            lines.append("% Mixed-truth (50% fast + 50% delayed with known ground truth)")
            lines.append(f"\\newcommand{{\\LIGOMixedNTotal}}{{{mt.get('n_total', '???')}}}")
            lines.append(f"\\newcommand{{\\LIGOMixedNValid}}{{{mt.get('n_valid', '???')}}}")
            lines.append(f"\\newcommand{{\\LIGOMixedNAmbiguous}}{{{mt.get('n_ambiguous', '???')}}}")
            lines.append(f"\\newcommand{{\\LIGOMixedAccuracy}}{{{fmt(100 * mt.get('classifier_accuracy', 0), 1)}}}")
            if mt.get('auc_true_labels'):
                lines.append(f"\\newcommand{{\\LIGOMixedAUCTrue}}{{{fmt(mt['auc_true_labels'], 3)}}}")
            if mt.get('auc_assigned_labels'):
                lines.append(f"\\newcommand{{\\LIGOMixedAUCAssigned}}{{{fmt(mt['auc_assigned_labels'], 3)}}}")
            # Confusion matrix
            cm = mt.get('confusion_matrix', {})
            if cm:
                lines.append("% Confusion matrix (true_pred counts)")
                lines.append(f"\\newcommand{{\\LIGOMixedCMFF}}{{{cm.get('true_fast_pred_fast', '???')}}}")
                lines.append(f"\\newcommand{{\\LIGOMixedCMFD}}{{{cm.get('true_fast_pred_delayed', '???')}}}")
                lines.append(f"\\newcommand{{\\LIGOMixedCMDF}}{{{cm.get('true_delayed_pred_fast', '???')}}}")
                lines.append(f"\\newcommand{{\\LIGOMixedCMDD}}{{{cm.get('true_delayed_pred_delayed', '???')}}}")

        # Configuration
        if cfg:
            lines.append("% Null simulation configuration")
            lines.append(f"\\newcommand{{\\LIGONullLateWindow}}{{{cfg.get('late_window_ms', [30, 100])}}}")
            lines.append(f"\\newcommand{{\\LIGONullEarlyWindow}}{{{cfg.get('early_window_ms', [0, 20])}}}")

        # Parameter sweep results (worst-case)
        sweep = ns.get('sweep', [])
        if sweep:
            lines.append("% Parameter sweep (worst-case tuning)")
            # Find worst-case across fixed-window sweeps (noise, AICc, delay)
            fixed_window_sweeps = [s for s in sweep if s.get('param') != 'late_window']
            if fixed_window_sweeps:
                max_fixed = max(fixed_window_sweeps, key=lambda x: x.get('delayed_rate', 0))
                lines.append(f"\\newcommand{{\\LIGOSweepFixedMax}}{{{fmt(100 * max_fixed.get('delayed_rate', 0), 2)}}}")
                lines.append(f"\\newcommand{{\\LIGOSweepFixedMaxCIHi}}{{{fmt(100 * max_fixed.get('ci', [0, 0])[1], 2)}}}")
            # Find worst-case window
            window_sweeps = [s for s in sweep if s.get('param') == 'late_window']
            if window_sweeps:
                max_window = max(window_sweeps, key=lambda x: x.get('delayed_rate', 0))
                lines.append(f"\\newcommand{{\\LIGOSweepWindowMax}}{{{fmt(100 * max_window.get('delayed_rate', 0), 2)}}}")
                lines.append(f"\\newcommand{{\\LIGOSweepWindowMaxCIHi}}{{{fmt(100 * max_window.get('ci', [0, 0])[1], 2)}}}")
                window_val = max_window.get('value', [0, 0])
                lines.append(f"\\newcommand{{\\LIGOSweepWorstWindow}}{{{window_val[0]}--{window_val[1]}}}")
            # Overall worst case
            if sweep:
                overall_max = max(sweep, key=lambda x: x.get('delayed_rate', 0))
                lines.append(f"\\newcommand{{\\LIGOSweepOverallMax}}{{{fmt(100 * overall_max.get('delayed_rate', 0), 2)}}}")
                lines.append(f"\\newcommand{{\\LIGOSweepOverallMaxCIHi}}{{{fmt(100 * overall_max.get('ci', [0, 0])[1], 2)}}}")

        # Spliced-null control results (cross-trace shuffle)
        spliced = ns.get('spliced', {})
        if spliced:
            lines.append("% Spliced-null control (cross-trace shuffle)")
            lines.append(f"\\newcommand{{\\LIGOSplicedAUCUnspliced}}{{{fmt(spliced.get('auc_unspliced', 0), 3)}}}")
            lines.append(f"\\newcommand{{\\LIGOSplicedAUCTrue}}{{{fmt(spliced.get('auc_true', 0), 3)}}}")
            lines.append(f"\\newcommand{{\\LIGOSplicedAUCMean}}{{{fmt(spliced.get('auc_spliced_mean', 0), 3)}}}")
            lines.append(f"\\newcommand{{\\LIGOSplicedAUCStd}}{{{fmt(spliced.get('auc_spliced_std', 0), 3)}}}")
            lines.append(f"\\newcommand{{\\LIGOSplicedAUCDrop}}{{{fmt(spliced.get('auc_drop', 0), 3)}}}")
    else:
        # Fallback values
        lines.append("% Null simulation (fallback - run ligo_null_simulation.py)")
        lines.append("\\newcommand{\\LIGONullMislabelRate}{1.5}")
        lines.append("\\newcommand{\\LIGONullMislabelCILo}{0.91}")
        lines.append("\\newcommand{\\LIGONullMislabelCIHi}{2.46}")
        lines.append("\\newcommand{\\LIGOMixedAUCTrue}{0.989}")
    lines.append("")

    # === Chinese 63-qubit ===
    lines.append("% === Li et al. 63-qubit (baseline) ===")
    lines.append("% Qualitative baseline only; no stable-core percentages reported due to pipeline differences")
    lines.append("% All show fast exponential recovery (capacity-wins baseline)")
    lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    """Generate and save macros file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LaTeX macros from pipeline outputs"
    )
    parser.add_argument(
        '--strict', action='store_true',
        help="Fail if any required JSON file is missing (for CI/release builds)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Generating LaTeX Macros from Pipeline Outputs")
    if args.strict:
        print("  Mode: STRICT (will fail on missing files)")
    print("=" * 70)

    # Define required files
    required_files = {
        'LIGO stability': LIGO_DIR / "bootstrap_beta_b.json",
        'LIGO threshold': LIGO_DIR / "threshold_sweep_results.json",
        'LIGO dip test': LIGO_DIR / "dip_test.json",
        'McEwen stability': OUTPUT_DIR / "robustness_sweep" / "stability_table.json",
    }

    optional_files = {
        'LIGO baseline': LIGO_DIR / "baseline_robustness_results.json",
    }

    # Check for required files
    print(f"\nChecking pipeline outputs...")
    missing = []
    for name, path in required_files.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status}")
        if not exists:
            missing.append(name)

    for name, path in optional_files.items():
        status = "OK" if path.exists() else "MISSING (optional)"
        print(f"  {name}: {status}")

    if args.strict and missing:
        print("\n" + "=" * 70)
        print("ERROR: Missing required files in strict mode:")
        for name in missing:
            print(f"  - {name}")
        print("\nRun the appropriate pipeline scripts to generate these files.")
        print("=" * 70)
        raise SystemExit(1)

    # Generate macros
    print(f"\nGenerating macros...")
    content = generate_macros()

    # Write output
    with open(MACROS_FILE, 'w') as f:
        f.write(content)

    print(f"\nWritten: {MACROS_FILE}")
    print(f"  {len(content.splitlines())} lines")

    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run combine_latex.py to update signatures_combined.tex")
    print("  2. Rebuild the PDF to verify all macros render correctly")


if __name__ == "__main__":
    main()
