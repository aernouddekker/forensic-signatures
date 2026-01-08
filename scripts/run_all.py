#!/usr/bin/env python3
"""
Master Reproducibility Script
=============================

Runs the complete Forensic Signatures analysis pipeline from raw data to PDF.
This is the single command a referee needs to reproduce all results.

Usage:
    python run_all.py --clean --n_events 10   # Fresh start (first run), test with 10 events
    python run_all.py --clean --n_events 500  # Fresh start, 500 events (~30GB, ~30 min)
    python run_all.py --cached                # Use cached data (~5 min)
    python run_all.py --macros                # Only regenerate macros and PDF (~30 sec)
    python run_all.py                         # Full pipeline, all 10k events (~168GB)

Modes:
    --clean   Fresh start: creates all directories and downloads all data.
              Use this for your first run or to start over completely.

    --cached  Skip data fetching if outputs exist. Use for re-running analysis
              after code changes without re-downloading data.

    --macros  Only regenerate LaTeX macros and compile PDF. Use when only
              the manuscript text has changed.

Data Downloads:
    The pipeline automatically downloads all required datasets:
    - LIGO strain data from GWOSC (smart bulk download)
    - McEwen 26-qubit data from Figshare (DOI: 10.6084/m9.figshare.16673851)
    - Li et al. 63-qubit data from Figshare (DOI: 10.6084/m9.figshare.28815434)

    Downloads are resumable - if interrupted, re-running continues from where
    it left off.

Requirements:
    pip install -r requirements.txt

Author: Aernoud Dekker
Date: December 2025
"""

import subprocess
import sys
import os
import re
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime

try:
    import requests
    from tqdm import tqdm
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
LATEX_DIR = PROJECT_DIR / "latex"
OUTPUT_DIR = SCRIPT_DIR / "output"
BULK_DATA_DIR = SCRIPT_DIR / "bulk_data"
DATA_DIR = SCRIPT_DIR / "data"

# Figshare dataset configuration
FIGSHARE_DATASETS = {
    'mcewan': {
        'article_id': '16673851',
        'name': 'McEwen 26-qubit cosmic ray data',
        'url': 'https://figshare.com/ndownloader/articles/16673851/versions/1',
        'output_dir': DATA_DIR / 'mcewan',
        'marker_file': 'MAIN_DATASET_0.csv',  # File that indicates successful extraction
    },
    'chinese': {
        'article_id': '28815434',
        'name': 'Li et al. 63-qubit cosmic ray data',
        'url': 'https://figshare.com/ndownloader/articles/28815434/versions/1',
        'output_dir': DATA_DIR / 'chinese',
        'marker_file': 'SI_Fig8',  # Directory that indicates successful extraction
    },
}


# =============================================================================
# Figshare Download Functions
# =============================================================================

def download_figshare_dataset(dataset_key, force=False):
    """
    Download and extract a Figshare dataset.

    Args:
        dataset_key: Key into FIGSHARE_DATASETS dict ('mcewan' or 'chinese')
        force: If True, re-download even if marker file exists

    Returns:
        True if successful, False otherwise
    """
    if not HAS_REQUESTS:
        print("  ERROR: 'requests' package not installed. Run: pip install requests tqdm")
        return False

    config = FIGSHARE_DATASETS[dataset_key]
    output_dir = config['output_dir']
    marker_path = output_dir / config['marker_file']

    # Check if already downloaded
    if marker_path.exists() and not force:
        print(f"  Already downloaded: {config['name']}")
        return True

    print(f"  Downloading: {config['name']}")
    print(f"  Source: Figshare article {config['article_id']}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download zip file
    zip_path = output_dir / f"figshare_{config['article_id']}.zip"

    try:
        response = requests.get(config['url'], stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True,
                         desc=f"  {dataset_key}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"  Downloaded: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

    except Exception as e:
        print(f"  ERROR downloading: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False

    # Extract zip file
    print(f"  Extracting to: {output_dir}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        print(f"  Extracted successfully")

        # Clean up zip file
        zip_path.unlink()

    except Exception as e:
        print(f"  ERROR extracting: {e}")
        return False

    # Verify marker file exists
    if not marker_path.exists():
        print(f"  WARNING: Expected marker file not found: {marker_path}")
        print(f"  Contents: {list(output_dir.iterdir())[:5]}...")
        # Don't fail - the structure might be slightly different

    return True


def ensure_figshare_data(cached=False):
    """
    Ensure all Figshare datasets are downloaded.

    Args:
        cached: If True, skip download if marker files exist

    Returns:
        dict mapping dataset_key -> success status
    """
    results = {}

    for key in FIGSHARE_DATASETS:
        config = FIGSHARE_DATASETS[key]
        marker_path = config['output_dir'] / config['marker_file']

        if cached and marker_path.exists():
            print(f"  {config['name']}: cached")
            results[key] = True
        else:
            results[key] = download_figshare_dataset(key)

    return results


# Script execution order (with descriptions)
PIPELINE_STEPS = [
    # Step 1: LIGO bulk data download (smart: only files containing events)
    # Downloads ~30 GB for 2000 events instead of 131 GB per-event fetching
    {
        'name': 'LIGO bulk data download',
        'script': 'ligo_bulk_download.py',
        'args': ['--class', 'Extremely_Loud', '--output', 'bulk_data'],
        'skip_if_cached': True,
        'outputs': ['../bulk_data/file_index.txt'],  # Created when download completes
        'uses_n_events': True,
    },
    # Step 2: LIGO analysis (uses bulk data if available)
    {
        'name': 'LIGO analysis',
        'script': 'ligo_glitch_analysis.py',
        'args': ['--classes', 'Extremely_Loud', '--bulk_data_dir', 'bulk_data'],
        'skip_if_cached': True,
        'outputs': ['ligo_envelope/ligo_envelope_Extremely_Loud_results.jsonl'],
        'uses_n_events': True,
    },
    # Step 3: LIGO stability analysis (3-window)
    {
        'name': 'LIGO stability analysis',
        'script': 'ligo_stability_figures.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': ['ligo_envelope/bootstrap_beta_b.json'],
    },
    # Step 4: LIGO curvature window sweep (robustness to fit interval)
    {
        'name': 'LIGO curvature sweep',
        'script': 'ligo_curvature_sweep.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': ['ligo_envelope/curvature_sweep_results.json'],
    },
    # Step 5: LIGO threshold sweep
    {
        'name': 'LIGO threshold sweep',
        'script': 'ligo_threshold_sweep.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': ['ligo_envelope/threshold_sweep_results.json'],
    },
    # Step 6: LIGO baseline robustness
    {
        'name': 'LIGO baseline robustness',
        'script': 'ligo_baseline_robustness.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': ['ligo_envelope/baseline_robustness_results.json'],
    },
    # Step 7: LIGO appendix figures (includes dip test)
    {
        'name': 'LIGO appendix figures',
        'script': 'ligo_appendix_figures.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': ['ligo_envelope/dip_test.json'],
    },
    # Step 8: LIGO GMM validation (unsupervised corroboration)
    {
        'name': 'LIGO GMM validation',
        'script': 'ligo_gmm_validation.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': ['ligo_envelope/unsupervised_validation.json'],
    },
    # Step 9: LIGO negative control (rejected morphology stress test)
    {
        'name': 'LIGO negative control',
        'script': 'ligo_negative_control.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': ['ligo_envelope/negative_control.json'],
    },
    # Step 10: LIGO alternative likelihood (log-domain robustness)
    {
        'name': 'LIGO alternative likelihood',
        'script': 'ligo_alt_likelihood.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': ['ligo_envelope/alternative_likelihood.json'],
    },
    # Step 11: LIGO null simulation control (window-decoupled, referee-grade)
    {
        'name': 'LIGO null simulation',
        'script': 'ligo_null_simulation.py',
        'args': ['--mode', 'both', '--n_synthetic', '1000'],
        'skip_if_cached': False,
        'outputs': ['ligo_envelope/null_simulation.json'],
    },
    # Step 12: Google/McEwan event freezing
    {
        'name': 'McEwan event extraction',
        'script': 'google_mcewan_analysis.py',
        'args': ['--freeze-only', '--data_dir', str(DATA_DIR / 'mcewan')],
        'skip_if_cached': True,
        'outputs': ['frozen_events.json'],
    },
    # Step 13: Google/McEwan robustness sweep (stability classification)
    {
        'name': 'McEwan stability analysis',
        'script': 'google_robustness_sweep.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': ['robustness_sweep/stability_table.json'],
    },
    # Step 13: Google/McEwan stability diagnostics (evidence strength)
    {
        'name': 'McEwan stability diagnostics',
        'script': 'google_stability_diagnostics.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': ['robustness_sweep/stability_diagnostics.json'],
    },
    # Step 14: McEwan tau statistics
    {
        'name': 'McEwan tau statistics',
        'script': 'mcewan_tau_stats.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': ['robustness_sweep/mcewan_tau_stats.json'],
    },
    # Step 15: Hesitation phase diagram
    {
        'name': 'Hesitation phase diagram',
        'script': 'hesitation_phase_diagram.py',
        'args': [],
        'skip_if_cached': False,
        'outputs': [],  # Generates figure only
    },
]


# =============================================================================
# Helpers
# =============================================================================

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_step(step_num, total, name):
    """Print step progress."""
    print(f"\n[{step_num}/{total}] {name}")
    print("-" * 50)


def run_script(script_name, args=None, cwd=None):
    """Run a Python script and return success status."""
    cmd = [sys.executable, script_name] + (args or [])
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or SCRIPT_DIR,
            capture_output=False,  # Show output in real-time
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script_name} failed with exit code {e.returncode}")
        return False


def check_outputs_exist(outputs):
    """Check if output files exist."""
    for output in outputs:
        path = OUTPUT_DIR / output
        if not path.exists():
            return False
    return True


def check_pdf_for_placeholders(pdf_path):
    """
    Check if the compiled LaTeX has any ??? or NA placeholders.
    Returns list of issues found.
    """
    # We can't easily check PDF, so check the .log file for undefined macros
    log_path = pdf_path.with_suffix('.log')
    issues = []

    if log_path.exists():
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read()
            # Check for undefined control sequences
            if 'Undefined control sequence' in content:
                issues.append("Undefined LaTeX macros found")
            # Check for overfull boxes (minor)
            overfull = content.count('Overfull')
            if overfull > 10:
                issues.append(f"{overfull} overfull boxes (layout warnings)")

    # Also check the aux file for ??? values
    aux_path = pdf_path.with_suffix('.aux')
    if aux_path.exists():
        with open(aux_path, 'r', errors='ignore') as f:
            if '???' in f.read():
                issues.append("Placeholder ??? values found in document")

    return issues


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(cached=False, macros_only=False, n_events=0, clean=False):
    """Run the complete analysis pipeline.

    Args:
        cached: Skip data fetching if cached outputs exist
        macros_only: Only regenerate macros and PDF
        n_events: Number of LIGO events (0 = all)
        clean: Fresh start - create directories and force re-download all data
    """
    start_time = datetime.now()

    # Determine mode
    if clean:
        mode = "clean (fresh start)"
    elif macros_only:
        mode = "macros-only"
    elif cached:
        mode = "cached"
    else:
        mode = "full"

    print_header("Forensic Signatures Reproducibility Pipeline")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {mode}")
    print(f"LIGO events: {n_events if n_events > 0 else 'all'}")

    # Create all required directories
    print("\nCreating directory structure...")
    directories = [
        OUTPUT_DIR / "ligo_envelope",
        OUTPUT_DIR / "robustness_sweep",
        OUTPUT_DIR / "chinese_frozen_events",
        DATA_DIR / "mcewan",
        DATA_DIR / "chinese",
        BULK_DATA_DIR,
        PROJECT_DIR / "figures" / "ligo",
        PROJECT_DIR / "figures" / "google",
        PROJECT_DIR / "figures" / "chinese",
        PROJECT_DIR / "figures" / "appendix",
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
    print(f"  Created {len(directories)} directories")

    failed_steps = []
    skipped_steps = []

    if not macros_only:
        # Download Figshare datasets (McEwan, Chinese)
        # In clean mode, force re-download; otherwise respect cached flag
        print_header("Downloading External Datasets")
        print("Checking Figshare datasets...")
        figshare_results = ensure_figshare_data(cached=(cached and not clean))
        if not all(figshare_results.values()):
            failed_datasets = [k for k, v in figshare_results.items() if not v]
            print(f"WARNING: Some datasets failed to download: {failed_datasets}")
            print("Continuing anyway - some analyses may fail...")

        # Run analysis pipeline
        total_steps = len(PIPELINE_STEPS)

        for i, step in enumerate(PIPELINE_STEPS, 1):
            print_step(i, total_steps, step['name'])

            # Check if we should skip
            # For data download steps: ALWAYS skip if data exists (never re-download)
            # For analysis steps: only skip if --cached flag is set
            if step['skip_if_cached']:
                if step['outputs'] and check_outputs_exist(step['outputs']):
                    if 'download' in step['name'].lower() or cached:
                        print(f"  Skipping (outputs exist)")
                        skipped_steps.append(step['name'])
                        continue

            # Build args, adding n_events for steps that use it
            args = step['args'].copy()
            if step.get('uses_n_events', False):
                args.extend(['--n_events', str(n_events)])

            # Run the script
            success = run_script(step['script'], args)

            if not success:
                failed_steps.append(step['name'])
                print(f"  FAILED")
                print(f"\nStopping pipeline due to failure.")
                print(f"Fix the error above and re-run with --cached to continue.")
                return 1
            else:
                print(f"  Done")

    # Step: Generate macros (always run)
    print_step(len(PIPELINE_STEPS) + 1 if not macros_only else 1,
               len(PIPELINE_STEPS) + 2 if not macros_only else 2,
               "Generate LaTeX macros")

    success = run_script('generate_macros.py', ['--strict'])
    if not success:
        print("  FAILED: Missing required JSON files")
        print(f"\nStopping pipeline due to failure.")
        return 1
    else:
        print("  Done")

    # Step: Compile PDF (using pre-combined signatures_submission.tex)
    print_step(len(PIPELINE_STEPS) + 2 if not macros_only else 2,
               len(PIPELINE_STEPS) + 2 if not macros_only else 2,
               "Compile PDF")

    pdf_file = LATEX_DIR / "signatures_submission.pdf"
    try:
        # Run pdflatex twice for references
        for run in [1, 2]:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', 'signatures_submission.tex'],
                cwd=LATEX_DIR,
                capture_output=True,
                text=True,
            )

        if pdf_file.exists():
            print(f"  PDF generated: {pdf_file}")

            # Check for issues
            issues = check_pdf_for_placeholders(pdf_file)
            if issues:
                print("  WARNINGS:")
                for issue in issues:
                    print(f"    - {issue}")
        else:
            failed_steps.append("pdflatex")
            print("  FAILED: PDF not generated")

    except FileNotFoundError:
        print("  WARNING: pdflatex not found, skipping PDF compilation")
        print("  Install TeX Live or run manually: pdflatex signatures_submission.tex")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print_header("Pipeline Summary")
    print(f"Duration: {duration}")
    print(f"Skipped: {len(skipped_steps)} steps (cached)")
    print(f"Failed: {len(failed_steps)} steps")

    if failed_steps:
        print("\nFailed steps:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nPipeline FAILED")
        return 1
    else:
        print("\nPipeline SUCCEEDED")
        print(f"\nOutput: {pdf_file}")
        return 0


# =============================================================================
# Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the complete Forensic Signatures analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --clean --n_events 10   # Fresh start, test with 10 events
  python run_all.py --clean --n_events 500  # Fresh start, 500 events (~30GB, ~30 min)
  python run_all.py --cached                # Use cached data (~5 min)
  python run_all.py --macros                # Only regenerate macros and PDF (~30 sec)
  python run_all.py                         # Full pipeline, all 10k events (~168GB)
        """
    )
    parser.add_argument(
        '--clean', action='store_true',
        help="Fresh start: create all directories and download all data (use for first run)"
    )
    parser.add_argument(
        '--cached', action='store_true',
        help="Skip data fetching steps if cached outputs exist"
    )
    parser.add_argument(
        '--macros', action='store_true',
        help="Only regenerate macros and compile PDF (assumes data exists)"
    )
    parser.add_argument(
        '--n_events', type=int, default=0,
        help="Number of LIGO events to fetch (default: 0 = all events)"
    )

    args = parser.parse_args()

    sys.exit(run_pipeline(
        cached=args.cached,
        macros_only=args.macros,
        n_events=args.n_events,
        clean=args.clean
    ))


if __name__ == "__main__":
    main()
