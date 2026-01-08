#!/usr/bin/env python3
"""
LIGO Pipeline Common Utilities
==============================

Centralized invariants for LIGO envelope analysis. All scripts import from here
to ensure mechanical identity of:
- Time reference (GPS-relative)
- Peak localization (±500ms constraint)
- Baseline estimation (tail of analysis window)
- Recovery coordinate normalization

This module is the single source of truth for these operations.

Author: Aernoud Dekker
Date: December 2025
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from typing import Tuple, Optional


# =============================================================================
# Constants
# =============================================================================

import pickle
from pathlib import Path

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

PEAK_SEARCH_WINDOW_MS = 500.0  # Search for peak within ±500ms of GPS time
DEFAULT_BASELINE_TAIL_FRACTION = 0.2  # Use last 20% of window for baseline
DEFAULT_BASELINE_WINDOW_MS = 150.0  # Default window for baseline estimation
CURVATURE_WINDOW_MS = 20.0  # Window for curvature index computation
ANALYSIS_WINDOWS_MS = [60, 100, 150]  # Standard analysis windows


# =============================================================================
# Cache Loading (handles legacy and new formats)
# =============================================================================

_legacy_fallback_warned = set()  # Track which GPS times triggered legacy warning

def load_cached_strain(gps_time: float, cache_dir: Path,
                       warn_legacy: bool = True) -> Optional[dict]:
    """
    Load cached strain data, handling both legacy and new cache formats.

    Legacy format: strain_{int(gps_time)}.pkl
    New format: strain_{gps_time:.6f}.pkl

    Args:
        gps_time: GPS time of event
        cache_dir: Path to cache directory
        warn_legacy: If True, warn once per GPS time when using legacy cache

    Returns:
        Dict with 'times', 'values', 'sample_rate' or None if not found
    """
    # Try new format first (6 decimal places - handles sub-second collisions)
    new_path = cache_dir / f"strain_{gps_time:.6f}.pkl"
    if new_path.exists():
        try:
            with open(new_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass

    # Fall back to legacy format (integer GPS time)
    legacy_path = cache_dir / f"strain_{int(gps_time)}.pkl"
    if legacy_path.exists():
        try:
            with open(legacy_path, 'rb') as f:
                data = pickle.load(f)
                # Warn once per integer second about legacy cache usage
                int_gps = int(gps_time)
                if warn_legacy and int_gps not in _legacy_fallback_warned:
                    _legacy_fallback_warned.add(int_gps)
                    import warnings
                    warnings.warn(
                        f"Using legacy cache for GPS {gps_time:.6f} "
                        f"(file: strain_{int_gps}.pkl). "
                        f"Consider re-fetching for collision safety.",
                        stacklevel=2
                    )
                return data
        except:
            pass

    return None


# =============================================================================
# Bulk HDF5 Reading (from pre-downloaded GWOSC bulk files)
# =============================================================================

BULK_FILE_DURATION = 4096  # Each bulk file covers 4096 seconds

# Cache for bulk file index to avoid repeated directory scans
_bulk_file_index = {}


def _build_bulk_file_index(bulk_dir: Path) -> dict:
    """
    Build index mapping GPS start times to bulk HDF5 file paths.
    Bulk files are named like: H-H1_GWOSC_O3a_4KHZ_R1-{GPS_START}-4096.hdf5
    """
    global _bulk_file_index

    bulk_dir_str = str(bulk_dir)
    if bulk_dir_str in _bulk_file_index:
        return _bulk_file_index[bulk_dir_str]

    index = {}
    if bulk_dir.exists():
        for f in bulk_dir.glob("*.hdf5"):
            # Parse GPS start from filename (second-to-last component before duration)
            parts = f.stem.split('-')
            if len(parts) >= 2:
                try:
                    gps_start = int(parts[-2])
                    duration = int(parts[-1])
                    index[gps_start] = {'path': f, 'duration': duration}
                except ValueError:
                    continue

    _bulk_file_index[bulk_dir_str] = index
    return index


def _find_bulk_file_for_gps(gps_time: float, bulk_dir: Path) -> Optional[Path]:
    """Find the bulk HDF5 file containing a given GPS time."""
    index = _build_bulk_file_index(bulk_dir)

    for gps_start, info in index.items():
        if gps_start <= gps_time < gps_start + info['duration']:
            return info['path']

    return None


def load_strain_from_bulk(gps_time: float, bulk_dir: Path,
                          window_before: float = 2.0,
                          window_after: float = 5.0) -> Optional[dict]:
    """
    Load strain data directly from pre-downloaded bulk HDF5 files.

    Args:
        gps_time: GPS time of the event
        bulk_dir: Directory containing bulk HDF5 files
        window_before: Seconds before GPS time (default 2)
        window_after: Seconds after GPS time (default 5)

    Returns:
        Dict with 'times', 'values', 'gps_time', 'sample_rate' or None
    """
    if not HAS_H5PY:
        return None

    bulk_dir = Path(bulk_dir)
    if not bulk_dir.exists():
        return None

    bulk_file = _find_bulk_file_for_gps(gps_time, bulk_dir)
    if bulk_file is None:
        return None

    try:
        with h5py.File(bulk_file, 'r') as f:
            strain_dataset = f['strain/Strain']
            strain_full = strain_dataset[:]

            t0 = strain_dataset.attrs['Xstart']
            dt = strain_dataset.attrs['Xspacing']
            n_samples = len(strain_full)

            # Extract window around GPS time
            t_start = gps_time - window_before
            t_end = gps_time + window_after

            idx_start = max(0, int((t_start - t0) / dt))
            idx_end = min(n_samples, int((t_end - t0) / dt) + 1)

            if idx_end <= idx_start:
                return None

            times = t0 + np.arange(idx_start, idx_end) * dt
            values = strain_full[idx_start:idx_end]

            return {
                'times': times,
                'values': values,
                'gps_time': gps_time,
                'sample_rate': 1.0 / dt
            }

    except Exception:
        return None


# =============================================================================
# Signal Processing
# =============================================================================

def bandpass_filter(data: np.ndarray, fs: float,
                    lowcut: float = 10.0, highcut: float = 500.0) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter.

    Args:
        data: Input signal
        fs: Sample rate in Hz
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz

    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)

    try:
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data)
    except Exception:
        # Fallback: just highpass if bandpass fails
        try:
            b, a = butter(4, low, btype='high')
            return filtfilt(b, a, data)
        except Exception:
            return data


def compute_hilbert_envelope(strain: np.ndarray, fs: float,
                              bandpass: bool = True) -> np.ndarray:
    """
    Compute Hilbert envelope of strain data.

    Args:
        strain: Raw strain values
        fs: Sample rate in Hz
        bandpass: Whether to apply bandpass filter first

    Returns:
        Envelope (magnitude of analytic signal)
    """
    if bandpass:
        filtered = bandpass_filter(strain, fs)
    else:
        filtered = strain
    analytic = hilbert(filtered)
    return np.abs(analytic)


# =============================================================================
# Time Reference (GPS-relative)
# =============================================================================

def compute_times_ms(times: np.ndarray, gps_time: float) -> np.ndarray:
    """
    Convert absolute times to milliseconds relative to GPS time.

    This is the canonical time reference for all LIGO analysis.
    t=0 corresponds to the reported GPS time of the event.

    Args:
        times: Absolute times in seconds
        gps_time: GPS time of the event in seconds

    Returns:
        Times in milliseconds relative to GPS time
    """
    return (times - gps_time) * 1000


def center_times_on_peak(times_ms: np.ndarray, peak_idx: int) -> np.ndarray:
    """
    Center times so peak is at t=0.

    This is the canonical time centering for all downstream analysis.
    After calling this, t_fit[0] should be ~0 if peak_idx is the first
    sample in the fit window.

    Use everywhere for consistency across main pipeline and downstream scripts.

    Args:
        times_ms: Time array in milliseconds (GPS-relative)
        peak_idx: Index of the peak in the envelope

    Returns:
        Times in milliseconds with peak at t=0
    """
    return times_ms - times_ms[peak_idx]


def extract_fit_window_indices(
    times_ms: np.ndarray,
    peak_idx: int,
    window_ms: float
) -> np.ndarray:
    """
    Extract indices for fit window using searchsorted (exact match to mask rule).

    This replaces the error-prone ceil()+1 logic. Uses searchsorted to find
    exactly which samples satisfy: 0 <= t <= window_ms (inclusive both ends).

    Equivalent to: mask = (times_ms >= peak_time) & (times_ms <= peak_time + window_ms)

    IMPORTANT: times_ms should be peak-centered (times_ms[peak_idx] == 0) for
    correct behavior. If not, center first with center_times_on_peak().

    Args:
        times_ms: Time array in ms (should be peak-centered so peak is at t=0)
        peak_idx: Index of peak in envelope
        window_ms: Window duration in ms

    Returns:
        Array of indices into envelope for the fit window
    """
    start = int(peak_idx)

    # Find all samples with t <= window_ms (inclusive, matches mask <=)
    # Using side="right" means: find insertion point after all values <= window_ms
    rel_times = times_ms[start:]
    n = int(np.searchsorted(rel_times, window_ms, side="right"))
    stop = start + n

    return np.arange(start, stop)


# =============================================================================
# Peak Localization (±500ms constraint)
# =============================================================================

def find_constrained_peak(envelope: np.ndarray, times_ms: np.ndarray,
                          search_window_ms: float = PEAK_SEARCH_WINDOW_MS) -> int:
    """
    Find peak index constrained to ±search_window_ms around GPS time (t=0).

    This ensures all scripts identify the same peak for a given event,
    even if there are louder features elsewhere in the fetched segment.

    Args:
        envelope: Hilbert envelope array
        times_ms: Time array in ms (GPS-relative, so t=0 is GPS time)
        search_window_ms: Search within ±this many ms of t=0

    Returns:
        Index of the peak in envelope array
    """
    search_mask = np.abs(times_ms) < search_window_ms
    search_envelope = envelope.copy()
    search_envelope[~search_mask] = 0
    return int(np.argmax(search_envelope))


def validate_peak_within_window(times_ms: np.ndarray, peak_idx: int,
                                 search_window_ms: float = PEAK_SEARCH_WINDOW_MS,
                                 warn_threshold_ms: float = 50.0) -> bool:
    """
    Validate that peak is within search window. Return True if valid.

    Real glitches don't reliably peak exactly at the catalog GPS timestamp.
    This validates the peak is within the constraint window, and warns if
    it's more than warn_threshold_ms from GPS time.

    Args:
        times_ms: Time array in ms (GPS-relative)
        peak_idx: Index of identified peak
        search_window_ms: Maximum allowed deviation (default 500ms)
        warn_threshold_ms: Threshold for warning (default 50ms)

    Returns:
        True if peak is within search window, False otherwise
    """
    peak_time = times_ms[peak_idx]
    if abs(peak_time) >= search_window_ms:
        return False
    if abs(peak_time) > warn_threshold_ms:
        import warnings
        warnings.warn(f"Peak at {peak_time:.1f} ms from GPS time (> {warn_threshold_ms} ms)")
    return True


# =============================================================================
# Baseline Estimation (tail of analysis window)
# =============================================================================

def baseline_from_postpeak_window(envelope: np.ndarray, times_ms: np.ndarray,
                                   peak_idx: int, window_ms: float,
                                   tail_fraction: float = DEFAULT_BASELINE_TAIL_FRACTION) -> float:
    """
    Compute baseline from the tail of the post-peak analysis window.

    This matches ligo_glitch_analysis.py which uses:
        baseline = np.median(env_fit[-len(env_fit)//5:])
    i.e., the last 20% of the analysis window.

    Args:
        envelope: Full envelope array
        times_ms: Time array in ms (GPS-relative)
        peak_idx: Index of peak in envelope
        window_ms: Analysis window duration (e.g., 100 or 150 ms)
        tail_fraction: Fraction of window to use for baseline (default 0.2)

    Returns:
        Baseline value (median of tail region)
    """
    peak_time = times_ms[peak_idx]
    end_time = peak_time + window_ms
    tail_start = peak_time + window_ms * (1 - tail_fraction)

    # Primary: samples in tail region
    tail_mask = (times_ms >= tail_start) & (times_ms <= end_time)
    if np.sum(tail_mask) >= 10:
        return float(np.median(envelope[tail_mask]))

    # Fallback: last tail_fraction of samples in window
    win_mask = (times_ms >= peak_time) & (times_ms <= end_time)
    idx = np.where(win_mask)[0]
    if len(idx) >= 10:
        tail_idx = idx[int((1 - tail_fraction) * len(idx)):]
        return float(np.median(envelope[tail_idx]))

    # Last resort: end of segment
    return float(np.median(envelope[-100:]))


# =============================================================================
# Recovery Coordinate Normalization
# =============================================================================

def compute_recovery_z(envelope: np.ndarray, peak_val: float,
                       baseline: float) -> np.ndarray:
    """
    Compute normalized recovery coordinate z(t).

    z = 1 - (E - baseline) / (peak - baseline)

    z starts at ~0 at peak, rises toward ~1 as envelope decays to baseline.

    Args:
        envelope: Envelope values (can be subset of full array)
        peak_val: Peak envelope value
        baseline: Baseline envelope value

    Returns:
        Recovery coordinate z (same shape as envelope)
    """
    if peak_val <= baseline:
        return np.zeros_like(envelope)
    return 1 - (envelope - baseline) / (peak_val - baseline)


def extract_fit_window(envelope: np.ndarray, times_ms: np.ndarray,
                       peak_idx: int, window_ms: float,
                       baseline_window_ms: Optional[float] = None
                       ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Extract post-peak window for model fitting with consistent normalization.

    Args:
        envelope: Full envelope array
        times_ms: Time array in ms (GPS-relative)
        peak_idx: Index of peak
        window_ms: Fit window duration in ms
        baseline_window_ms: Window for baseline estimation (default: same as window_ms)

    Returns:
        (t_fit, z_fit, peak_val, baseline)
        - t_fit: Time relative to peak, starting at 0
        - z_fit: Recovery coordinate in fit window
        - peak_val: Peak envelope value
        - baseline: Baseline value used for normalization
    """
    if baseline_window_ms is None:
        baseline_window_ms = window_ms

    peak_time = times_ms[peak_idx]
    peak_val = envelope[peak_idx]
    baseline = baseline_from_postpeak_window(envelope, times_ms, peak_idx, baseline_window_ms)

    # Extract window
    mask = (times_ms >= peak_time) & (times_ms <= peak_time + window_ms)
    t_fit = times_ms[mask] - peak_time
    env_fit = envelope[mask]

    # Compute recovery coordinate
    z_fit = compute_recovery_z(env_fit, peak_val, baseline)
    z_fit = np.clip(z_fit, 0, 1.5)  # Allow slight overshoot

    return t_fit, z_fit, peak_val, baseline


# =============================================================================
# Sample-rate-independent peak finding
# =============================================================================

def distance_samples_from_times(times_ms: np.ndarray,
                                 min_sep_ms: float = 2.5) -> int:
    """
    Convert minimum peak separation (in ms) to samples.

    This ensures find_peaks distance is sample-rate independent.
    At 4096 Hz, min_sep_ms=2.5 gives distance~10.
    At 16384 Hz, it gives distance~41.

    Args:
        times_ms: Time array in milliseconds
        min_sep_ms: Minimum separation between peaks in ms

    Returns:
        Distance in samples for scipy.signal.find_peaks
    """
    dt = np.median(np.diff(times_ms))
    if not np.isfinite(dt) or dt <= 0:
        return 10  # fallback
    return max(1, int(np.ceil(min_sep_ms / dt)))


# =============================================================================
# Curvature Index (Canonical Implementation)
# =============================================================================

def compute_curvature_index(envelope: np.ndarray, times_ms: np.ndarray,
                            peak_idx: int,
                            curvature_window_ms: float = CURVATURE_WINDOW_MS,
                            baseline_window_ms: float = DEFAULT_BASELINE_WINDOW_MS
                            ) -> Optional[float]:
    """
    Compute early-time curvature index b using CANONICAL functions.

    Uses the same pipeline as classification:
    - Center times on peak (peak at t=0)
    - Extract 150ms window by indices (same as classification)
    - Compute baseline from n//5 tail of 150ms window
    - Apply canonical z-transform with epsilon and clip
    - Polyfit on first 20ms of z(t)

    This ensures curvature is computed identically across all scripts.

    Args:
        envelope: Full envelope array
        times_ms: Time array in ms (GPS-relative, NOT peak-centered)
        peak_idx: Index of peak
        curvature_window_ms: Window for quadratic fit (default 20 ms)
        baseline_window_ms: Window for baseline estimation (default 150 ms)

    Returns:
        Curvature index b (quadratic coefficient), or None if computation fails
    """
    # Import canonical baseline function (no circular dependency)
    from iof_metrics import baseline_tail_median

    peak_val = envelope[peak_idx]

    # Step 1: Center times on peak (canonical approach)
    times_centered = center_times_on_peak(times_ms, peak_idx)

    # Step 2: Extract 150ms window by indices (same as classification)
    env_fit_150_idx = extract_fit_window_indices(times_centered, peak_idx, baseline_window_ms)

    if len(env_fit_150_idx) < 50:
        return None

    env_fit_150 = envelope[env_fit_150_idx]

    # Step 3: Canonical baseline from n//5 tail of 150ms window
    baseline_150 = baseline_tail_median(env_fit_150)

    if peak_val <= baseline_150:
        return None

    # Step 4: Extract 20ms curvature window by indices
    env_fit_20_idx = extract_fit_window_indices(times_centered, peak_idx, curvature_window_ms)

    if len(env_fit_20_idx) < 10:
        return None

    t_20 = times_centered[env_fit_20_idx]
    env_20 = envelope[env_fit_20_idx]

    # Step 5: Canonical z-transform with epsilon and clip (same as classification)
    z_20 = 1 - (env_20 - baseline_150) / (peak_val - baseline_150 + 1e-30)
    z_20 = np.clip(z_20, 0, 1.5)

    # Step 6: Polyfit and return quadratic coefficient
    try:
        coeffs = np.polyfit(t_20, z_20, 2)
        return float(coeffs[0])  # Quadratic coefficient b
    except:
        return None
