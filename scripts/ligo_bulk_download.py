#!/usr/bin/env python3
"""
LIGO Smart Bulk HDF5 Download Script
=====================================

Downloads ONLY the bulk HDF5 files that contain glitch events, not the full time range.
This is much more efficient than per-event fetching (which re-downloads the same 67MB file
for each event in the same 4096-second window).

Efficiency comparison for 10,099 Extremely_Loud events:
- Per-event fetch_open_data(): 661 GB network traffic (same file downloaded multiple times)
- Smart bulk download: 168 GB network traffic (each file downloaded once)

Usage:
    python ligo_bulk_download.py --class Extremely_Loud --output bulk_data/ --dry-run
    python ligo_bulk_download.py --class Extremely_Loud --output bulk_data/

Author: Aernoud Dekker
Date: December 2025
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Set, Dict
from collections import Counter

import pandas as pd
import requests
from tqdm import tqdm

try:
    from gwosc.locate import get_urls
except ImportError:
    print("ERROR: gwosc package not installed. Run: pip install gwosc")
    sys.exit(1)


# Default paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_CSV_PATH = DATA_DIR / "H1_O3a_glitches.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "bulk_data"

# Bulk file duration
BULK_FILE_DURATION = 4096  # seconds

# Gravity Spy CSV download info (Zenodo)
GRAVITY_SPY_ZENODO_URL = "https://zenodo.org/records/5649212/files/H1_O3a.csv?download=1"
GRAVITY_SPY_ZENODO_MD5 = "29aea278b622cd97496971f7c07f7d6a"


def download_gravity_spy_csv(csv_path: Path) -> bool:
    """
    Download Gravity Spy H1_O3a CSV from Zenodo if not present.

    Source: https://zenodo.org/records/5649212
    DOI: 10.5281/zenodo.5649211
    """
    if csv_path.exists():
        return True

    print(f"Gravity Spy CSV not found at: {csv_path}")
    print(f"Downloading from Zenodo (~90 MB)...")

    # Create parent directories
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(GRAVITY_SPY_ZENODO_URL, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        temp_path = csv_path.with_suffix('.tmp')

        with open(temp_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True,
                     desc="H1_O3a_glitches.csv") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Rename to final path
        temp_path.rename(csv_path)
        print(f"  Downloaded to: {csv_path}")
        return True

    except Exception as e:
        print(f"  ERROR downloading CSV: {e}")
        temp_path = csv_path.with_suffix('.tmp')
        if temp_path.exists():
            temp_path.unlink()
        return False


def load_gravity_spy_csv(csv_path: Path, glitch_class: str,
                         min_snr: float = 50.0, min_confidence: float = 0.8) -> pd.DataFrame:
    """Load and filter Gravity Spy glitch metadata."""
    print(f"Loading Gravity Spy CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")

    # Filter by class
    if 'label' in df.columns:
        label_col = 'label'
    elif 'ml_label' in df.columns:
        label_col = 'ml_label'
    else:
        raise ValueError("Cannot find label column in CSV")

    df_filtered = df[df[label_col] == glitch_class].copy()
    print(f"  {glitch_class}: {len(df_filtered)} rows")

    # Filter by IFO (H1 only for now)
    if 'ifo' in df.columns:
        df_filtered = df_filtered[df_filtered['ifo'] == 'H1']
        print(f"  H1 only: {len(df_filtered)} rows")

    # Filter by SNR
    if 'snr' in df.columns:
        df_filtered = df_filtered[df_filtered['snr'] >= min_snr]
        print(f"  SNR >= {min_snr}: {len(df_filtered)} rows")

    # Filter by confidence
    if 'ml_confidence' in df.columns:
        df_filtered = df_filtered[df_filtered['ml_confidence'] >= min_confidence]
        print(f"  Confidence >= {min_confidence}: {len(df_filtered)} rows")

    return df_filtered


def get_gps_times(df: pd.DataFrame) -> List[float]:
    """Extract GPS times from DataFrame."""
    if 'event_time' in df.columns:
        gps_col = 'event_time'
    elif 'GPStime' in df.columns:
        gps_col = 'GPStime'
    elif 'gps_time' in df.columns:
        gps_col = 'gps_time'
    else:
        raise ValueError("Cannot find GPS time column in CSV")

    return df[gps_col].tolist()


def get_unique_file_windows(gps_times: List[float]) -> Dict[int, List[float]]:
    """
    Group GPS times by which 4096-second bulk file they fall into.

    Returns dict mapping file_start_gps -> list of event GPS times in that file.
    """
    windows = {}
    for gps in gps_times:
        file_start = int(gps // BULK_FILE_DURATION) * BULK_FILE_DURATION
        if file_start not in windows:
            windows[file_start] = []
        windows[file_start].append(gps)
    return windows


def get_bulk_file_urls(file_windows: Dict[int, List[float]], detector: str = 'H1',
                       sample_rate: int = 4096) -> Dict[int, str]:
    """
    Get URLs for bulk files using gwosc.locate.get_urls().

    Only queries for the specific 4096-second windows that contain events.
    Returns dict mapping file_start_gps -> URL.
    """
    print(f"\nQuerying GWOSC for {len(file_windows)} bulk file URLs...")

    file_urls = {}
    failed = []

    for file_start, events in tqdm(sorted(file_windows.items()),
                                    desc="  Querying GWOSC",
                                    unit="file"):
        try:
            # Query for this specific 4096-second window
            urls = get_urls(detector, file_start, file_start + BULK_FILE_DURATION,
                           sample_rate=sample_rate, format='hdf5')
            if urls:
                file_urls[file_start] = urls[0]  # Should be exactly one file
            else:
                failed.append(file_start)
        except Exception as e:
            failed.append(file_start)

    print(f"\n  Found URLs for {len(file_urls)} files")
    if failed:
        print(f"  Failed to get URLs for {len(failed)} windows")

    return file_urls


def load_done_list(done_file: Path) -> Set[str]:
    """Load list of already-downloaded URLs."""
    if done_file.exists():
        with open(done_file) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_to_done_list(done_file: Path, url: str):
    """Append URL to done list."""
    with open(done_file, 'a') as f:
        f.write(url + '\n')


def load_url_cache(cache_file: Path) -> Dict[int, str]:
    """Load cached URL mapping (file_start_gps -> URL)."""
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
            # Convert string keys back to int
            return {int(k): v for k, v in data.items()}
    return {}


def save_url_cache(cache_file: Path, file_urls: Dict[int, str]):
    """Save URL mapping to cache."""
    with open(cache_file, 'w') as f:
        json.dump(file_urls, f, indent=2)


def download_file(url: str, output_dir: Path, chunk_size: int = 8192,
                  retry_count: int = 3, retry_delay: float = 5.0) -> bool:
    """
    Download a single file with streaming and retry support.
    Returns True if successful, False otherwise.
    """
    filename = url.split('/')[-1]
    output_path = output_dir / filename

    # Skip if already exists
    if output_path.exists():
        return True

    for attempt in range(retry_count):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            temp_path = output_path.with_suffix('.tmp')

            with open(temp_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True,
                             desc=filename[:40], leave=False) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

            temp_path.rename(output_path)
            return True

        except Exception as e:
            if attempt < retry_count - 1:
                print(f"  Retry {attempt + 1}/{retry_count} for {filename}: {e}")
                time.sleep(retry_delay)
            else:
                print(f"  FAILED: {filename}: {e}")
                temp_path = output_path.with_suffix('.tmp')
                if temp_path.exists():
                    temp_path.unlink()
                return False

    return False


def main():
    parser = argparse.ArgumentParser(
        description='Download ONLY bulk LIGO HDF5 files that contain glitch events')

    parser.add_argument('--csv', type=str, default=str(DEFAULT_CSV_PATH),
                       help='Path to Gravity Spy CSV')
    parser.add_argument('--class', dest='glitch_class', type=str,
                       default='Extremely_Loud',
                       help='Glitch class to download (default: Extremely_Loud)')
    parser.add_argument('--n_events', type=int, default=0,
                       help='Number of events to cover (0 = all, sorted by SNR)')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help='Output directory for bulk files')
    parser.add_argument('--min-snr', type=float, default=50.0,
                       help='Minimum SNR filter')
    parser.add_argument('--min-confidence', type=float, default=0.8,
                       help='Minimum ML confidence filter')
    parser.add_argument('--detector', type=str, default='H1',
                       help='Detector (H1 or L1)')
    parser.add_argument('--sample-rate', type=int, default=4096,
                       help='Sample rate (4096 or 16384)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between downloads in seconds')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be downloaded without downloading')

    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)

    print("=" * 70)
    print("LIGO Smart Bulk HDF5 Download")
    print("=" * 70)

    # Download CSV from Zenodo if not present
    if not download_gravity_spy_csv(csv_path):
        print("ERROR: Could not obtain Gravity Spy CSV")
        sys.exit(1)

    # Load and filter CSV
    df = load_gravity_spy_csv(csv_path, args.glitch_class,
                              args.min_snr, args.min_confidence)

    if len(df) == 0:
        print("ERROR: No events match filters")
        sys.exit(1)

    # Sort by SNR and limit to n_events if specified
    if 'snr' in df.columns:
        df = df.sort_values('snr', ascending=False)

    if args.n_events > 0:
        df = df.head(args.n_events)
        print(f"  Limited to top {args.n_events} events by SNR")

    # Get GPS times and group by bulk file window
    gps_times = get_gps_times(df)
    file_windows = get_unique_file_windows(gps_times)

    print(f"\nEvent distribution:")
    print(f"  Total events: {len(gps_times)}")
    print(f"  Unique bulk files needed: {len(file_windows)}")
    print(f"  Average events per file: {len(gps_times) / len(file_windows):.1f}")

    # Calculate efficiency
    naive_traffic = len(gps_times) * 67  # MB
    smart_traffic = len(file_windows) * 67  # MB
    savings = naive_traffic - smart_traffic
    print(f"\nBandwidth comparison:")
    print(f"  Per-event fetch: {naive_traffic / 1024:.0f} GB")
    print(f"  Smart bulk download: {smart_traffic / 1024:.0f} GB")
    print(f"  Savings: {savings / 1024:.0f} GB ({100 * savings / naive_traffic:.0f}%)")

    if args.dry_run:
        print("\n[DRY RUN] Would download files for these GPS windows:")
        for i, (file_start, events) in enumerate(sorted(file_windows.items())[:10]):
            print(f"  GPS {file_start}: {len(events)} events")
        if len(file_windows) > 10:
            print(f"  ... and {len(file_windows) - 10} more files")
        print(f"\nTo actually download, remove --dry-run flag")
        return

    # Get URLs for all required files (with caching)
    output_dir.mkdir(parents=True, exist_ok=True)
    url_cache_file = output_dir / 'url_cache.json'
    cached_urls = load_url_cache(url_cache_file)

    # Find which file windows are missing from cache
    missing_windows = {k: v for k, v in file_windows.items() if k not in cached_urls}

    if missing_windows:
        print(f"\nURL cache: {len(cached_urls)} cached, {len(missing_windows)} to query")
        new_urls = get_bulk_file_urls(missing_windows, args.detector, args.sample_rate)
        cached_urls.update(new_urls)
        save_url_cache(url_cache_file, cached_urls)
        print(f"  Saved {len(cached_urls)} URLs to cache")
    else:
        print(f"\nURL cache: all {len(file_windows)} URLs cached (skipping GWOSC queries)")

    # Filter to just the URLs we need for this run
    file_urls = {k: cached_urls[k] for k in file_windows.keys() if k in cached_urls}

    if not file_urls:
        print("ERROR: No bulk file URLs found")
        sys.exit(1)

    # Download files
    done_file = output_dir / 'download_done.txt'
    done_list = load_done_list(done_file)

    # Filter to pending URLs
    pending = [(start, url) for start, url in file_urls.items() if url not in done_list]

    print(f"\nDownload status:")
    print(f"  Total files: {len(file_urls)}")
    print(f"  Already done: {len(done_list)}")
    print(f"  Pending: {len(pending)}")

    if not pending:
        print("  All files already downloaded!")
    else:
        print(f"\nDownloading {len(pending)} files...")
        success_count = 0
        fail_count = 0

        for i, (file_start, url) in enumerate(pending):
            filename = url.split('/')[-1]
            n_events = len(file_windows[file_start])
            print(f"[{i+1}/{len(pending)}] {filename} ({n_events} events)")

            if download_file(url, output_dir):
                save_to_done_list(done_file, url)
                success_count += 1
            else:
                fail_count += 1

            if i < len(pending) - 1:
                time.sleep(args.delay)

        print(f"\nDownload complete: {success_count} succeeded, {fail_count} failed")

    # Save file index for later use
    index_file = output_dir / 'file_index.txt'
    with open(index_file, 'w') as f:
        for file_start, url in sorted(file_urls.items()):
            filename = url.split('/')[-1]
            n_events = len(file_windows[file_start])
            f.write(f"{file_start}\t{filename}\t{n_events}\n")
    print(f"\nFile index saved to: {index_file}")


if __name__ == "__main__":
    main()
