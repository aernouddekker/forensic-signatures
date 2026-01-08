#!/usr/bin/env python3
"""
IOF Metrics Core Module
=======================

Unified measurement functions for IOF forensic analysis across all datasets:
- McEwen cosmic ray data (superconducting qubits)
- LIGO glitch data (gravitational wave detectors)
- Chinese 63-qubit cosmic ray data

This module provides consistent methodology for:
1. Event detection with robust statistics
2. Recovery window extraction
3. t_peak computation (time to steepest recovery slope)
4. Competing model fits with AICc
5. Event classification (Standard vs IOF)
6. Parameter robustness sweeps

Author: Aernoud Dekker
Date: December 2025
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any, Callable
import json
import hashlib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# Data Classes for Structured Results
# =============================================================================

@dataclass
class EventDetectionParams:
    """Parameters for event detection."""
    threshold_sigma: float = 5.0      # Detection threshold in MAD sigmas
    min_drop_depth: float = 10.0      # Minimum absolute drop/rise
    min_separation_ms: float = 500.0  # Minimum time between events
    detection_direction: str = 'drop' # 'drop' or 'rise'

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        """Return hash of parameters for reproducibility tracking."""
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:8]


@dataclass
class SmoothingParams:
    """Parameters for smoothing operations."""
    method: str = 'moving_avg'  # 'none', 'moving_avg', 'savgol'
    window_ms: float = 1.0      # Window size in milliseconds
    savgol_order: int = 3       # Polynomial order for Savitzky-Golay

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DerivativeParams:
    """Parameters for derivative computation."""
    method: str = 'gradient'    # 'gradient', 'central', 'savgol'
    smooth_first: bool = True   # Apply smoothing before derivative
    smoothing: SmoothingParams = field(default_factory=SmoothingParams)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['smoothing'] = self.smoothing.to_dict()
        return d


@dataclass
class FitParams:
    """Parameters for model fitting."""
    fix_baseline: bool = True           # Fix baseline from late-time data
    baseline_fraction: float = 0.2      # Fraction of late-time data for baseline
    early_window_ms: float = 20.0       # Early-time window for separate fit
    full_window_ms: float = 100.0       # Full window for fitting
    maxfev: int = 10000                 # Max function evaluations

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassificationParams:
    """Parameters for IOF classification.

    Note: legacy_t_peak_threshold_ms is unused in model-based classification.
    Classification is based on winning model geometry (fast vs delayed),
    not a threshold on derivative-based t_peak. Kept for backward compatibility.
    """
    legacy_t_peak_threshold_ms: float = 5.0  # UNUSED - kept for compatibility
    aic_threshold: float = 10.0              # AIC difference threshold

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Event:
    """Detected event with metadata."""
    index: int                          # Index in original array
    time_ms: float                      # Time in milliseconds
    value: float                        # Value at event (min or max)
    baseline: float                     # Baseline value
    amplitude: float                    # Amplitude (baseline - value) or (value - baseline)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitResult:
    """Result from a single model fit."""
    model_name: str
    params: Dict[str, float]
    fit_values: np.ndarray
    residuals: np.ndarray
    ss_res: float
    r2: float
    aic: float
    aicc: float
    n_params: int
    success: bool = True
    error_msg: str = ""
    hit_bounds: bool = False           # True if any param hit its bound
    hit_bounds_params: List[str] = field(default_factory=list)  # Which params hit bounds

    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'params': self.params,
            'ss_res': self.ss_res,
            'r2': self.r2,
            'aic': self.aic,
            'aicc': self.aicc,
            'n_params': self.n_params,
            'success': self.success,
            'error_msg': self.error_msg,
            'hit_bounds': self.hit_bounds,
            'hit_bounds_params': self.hit_bounds_params
        }


@dataclass
class EventAnalysis:
    """Complete analysis result for a single event."""
    event: Event
    # Model-based classification (primary - methodologically correct)
    classification: str                  # 'standard', 'iof', or 'uncertain'
    classification_reason: str
    winning_model: str                   # Best model by AICc
    model_t_peak_ms: float              # t_peak from winning model (analytical)
    delta_aicc: float                    # AICc gap to second-best model
    # Derivative-based t_peak (diagnostic only - kept for comparison)
    deriv_t_peak_ms: float              # t_peak from smoothed derivative
    deriv_t_peak_ci: Tuple[float, float] # Bootstrap CI for derivative t_peak
    # Fit results
    fits: Dict[str, FitResult]
    normalized_curve: np.ndarray
    time_grid_ms: np.ndarray
    params_hash: str                     # Hash of all parameters used

    def to_dict(self) -> dict:
        return {
            'event': self.event.to_dict(),
            'classification': self.classification,
            'classification_reason': self.classification_reason,
            'winning_model': self.winning_model,
            'model_t_peak_ms': self.model_t_peak_ms,
            'delta_aicc': self.delta_aicc,
            'deriv_t_peak_ms': self.deriv_t_peak_ms,
            'deriv_t_peak_ci': list(self.deriv_t_peak_ci),
            'fits': {k: v.to_dict() for k, v in self.fits.items()},
            'params_hash': self.params_hash
        }


# =============================================================================
# Model Functions
# =============================================================================

def exponential_recovery(t: np.ndarray, A: float, tau: float, baseline: float) -> np.ndarray:
    """
    Standard Physics: Exponential recovery to baseline.

    y(t) = baseline - A * exp(-t/tau)

    At t=0: y = baseline - A (minimum)
    As t->inf: y -> baseline
    """
    return baseline - A * np.exp(-t / tau)


def exponential_recovery_fixed(baseline: float) -> Callable:
    """Return exponential recovery with fixed baseline."""
    def model(t: np.ndarray, A: float, tau: float) -> np.ndarray:
        return baseline - A * np.exp(-t / tau)
    return model


def power_law_recovery(t: np.ndarray, A: float, tau: float, baseline: float) -> np.ndarray:
    """
    Standard Physics: Power-law recovery (1/t behavior).

    y(t) = baseline - A / (1 + t/tau)
    """
    return baseline - A / (1 + t / tau)


def power_law_recovery_fixed(baseline: float) -> Callable:
    """Return power-law recovery with fixed baseline."""
    def model(t: np.ndarray, A: float, tau: float) -> np.ndarray:
        return baseline - A / (1 + t / tau)
    return model


def logistic_sigmoid(t: np.ndarray, A: float, k: float, t0: float, minimum: float) -> np.ndarray:
    """
    IOF Physics: Logistic sigmoid recovery.

    y(t) = minimum + A / (1 + exp(-k*(t - t0)))

    Shows initial plateau before rapid recovery.
    Inflection point (steepest slope) at t = t0.
    """
    return minimum + A / (1 + np.exp(-k * (t - t0)))


# =============================================================================
# Model-Based t_peak Computation
# =============================================================================

def compute_model_t_peak(model_name: str, params: Dict[str, float]) -> float:
    """
    Compute t_peak analytically from fitted model parameters.

    This is the methodologically correct approach - t_peak is derived from
    the model geometry, not from a smoothed numerical derivative.

    Args:
        model_name: Name of the fitted model
        params: Dictionary of fitted parameter values

    Returns:
        t_peak in milliseconds (time of steepest recovery slope)

    Model-specific formulas:
        - exponential, exponential_fixed, power_law: t_peak = 0
          (steepest slope is at t=0 for monotonic concave-down recovery)
        - sigmoid: t_peak = t0 (inflection point of logistic)
        - delayed: t_peak = delay (onset of recovery after plateau)
    """
    if model_name in ('exponential', 'exponential_fixed', 'power_law'):
        # For pure exponential/power-law recovery, steepest slope is at t=0
        return 0.0

    elif model_name == 'sigmoid':
        # Logistic sigmoid: inflection at t = t0
        return params.get('t0', 0.0)

    elif model_name == 'delayed':
        # Delayed exponential: steepest slope at onset of recovery
        return params.get('delay', 0.0)

    else:
        # Unknown model - return 0 as fallback
        return 0.0


def get_model_geometry(model_name: str) -> str:
    """
    Classify model into geometry type.

    Returns:
        'fast': Immediate recovery (exponential, power-law)
        'delayed': Hesitation/plateau before recovery (sigmoid, delayed)
    """
    if model_name in ('exponential', 'exponential_fixed', 'power_law'):
        return 'fast'
    elif model_name in ('sigmoid', 'delayed'):
        return 'delayed'
    else:
        return 'unknown'


def delayed_exponential(t: np.ndarray, A: float, tau: float, delay: float, minimum: float) -> np.ndarray:
    """
    IOF Physics: Delayed exponential recovery.

    Exponential recovery that only begins after an initial plateau.
    """
    baseline = minimum + A
    return np.where(
        t < delay,
        minimum,
        baseline - A * np.exp(-(t - delay) / tau)
    )


# =============================================================================
# LIGO Canonical Fitting Functions (Science Contract)
# =============================================================================
# These functions define the canonical statistics for LIGO envelope analysis.
# They are the single source of truth - all scripts must use these functions
# to ensure reproducibility and prevent implementation drift.
#
# Science Contract (frozen):
# - baseline_tail_median(): n//5 rule, no parameters
# - check_fit_sanity(): returns (bool, str) tuple
# - fit_competing_models_z(): z-domain fitter, NO baseline logic
# - fit_envelope_with_baseline(): explicit baseline, no recompute
# =============================================================================

def check_bound_hits(
    params: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    eps_frac: float = 0.01,
    eps_abs: Optional[float] = None,
    eps_abs_params: Optional[set] = None
) -> Tuple[bool, List[str]]:
    """
    Check if any fitted parameters are within eps of their bounds.

    This catches the "optimizer parks at boundary" pathology (e.g., t0 at grid max).

    Args:
        params: Fitted parameter values
        bounds: Parameter bounds as {name: (lower, upper)}
        eps_frac: Fraction of range to consider "at boundary" (default 1%)
        eps_abs: Absolute epsilon floor (e.g., 0.5 ms for time params).
        eps_abs_params: Set of param names to apply eps_abs to (e.g., {'tau', 't0', 'delay'}).
                        If None and eps_abs is set, applies to all params.

    Returns:
        (hit_any, hit_params): Whether any param hit bounds, and which ones
    """
    # Default time-like params if eps_abs_params not specified but eps_abs is
    if eps_abs is not None and eps_abs_params is None:
        eps_abs_params = {'tau', 't0', 'delay'}

    hit_params = []
    for name, value in params.items():
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        rng = hi - lo
        if rng <= 0:
            # Degenerate bounds - skip this param
            continue
        eps = eps_frac * rng
        # Apply eps_abs only to specified params (time-like by default)
        if eps_abs is not None and eps_abs_params and name in eps_abs_params:
            eps = max(eps, eps_abs)
        if value <= lo + eps or value >= hi - eps:
            hit_params.append(name)
    return len(hit_params) > 0, hit_params


def baseline_tail_median(env_fit: np.ndarray) -> float:
    """
    Compute baseline from tail of fit window.

    EXACT match to main pipeline: np.median(env_fit[-len(env_fit)//5:])
    No parameters — this is a science invariant.

    Args:
        env_fit: Post-peak envelope samples (must start at peak)

    Returns:
        Baseline value (median of last n//5 samples)
    """
    n = len(env_fit)
    if n <= 0:
        return float("nan")
    k = max(1, n // 5)  # EXACT integer division, matches main pipeline
    return float(np.median(env_fit[-k:]))


def check_fit_sanity(fit: FitResult) -> Tuple[bool, str]:
    """
    Canonical sanity filter for model fits.

    Returns (bool, str) tuple - NEVER just bool.
    This prevents the truthy-tuple bug where tuples are always True.

    Args:
        fit: FitResult from model fitting

    Returns:
        (is_sane, reason): Tuple of (pass/fail, explanation string)
    """
    if not fit.success:
        return False, "fit_failed"

    if fit.r2 < 0.3:
        return False, f"low_r2_{fit.r2:.2f}"

    p = fit.params
    if 'tau' in p and (p['tau'] < 0.5 or p['tau'] > 180):
        return False, f"tau_out_of_range_{p['tau']:.1f}"

    if 'delay' in p and (p['delay'] < 0 or p['delay'] > 80):
        return False, f"delay_out_of_range_{p['delay']:.1f}"

    if 't0' in p and (p['t0'] < 1 or p['t0'] > 80):
        return False, f"t0_out_of_range_{p['t0']:.1f}"

    return True, "ok"


def fit_competing_models_z(
    t_ms: np.ndarray,
    z: np.ndarray,
    model_set: Optional[set] = None,
    maxfev: int = 10000
) -> Dict[str, FitResult]:
    """
    Canonical fitter: fits z-domain data directly. NO baseline logic.

    This is the core fitting function. Caller is responsible for:
    1. Computing baseline via baseline_tail_median()
    2. Transforming envelope to z via: z = 1 - (env - baseline) / (peak - baseline + 1e-30)
    3. Clipping z to [0, 1.5]

    Models fitted:
    - exponential: z(t) = 1 - exp(-t/tau)
    - exponential_fixed: same, 2 params (for compatibility)
    - power_law: z(t) = t / (t + tau)
    - sigmoid: z(t) = 1 / (1 + exp(-k*(t - t0)))
    - delayed: z(t) = 0 for t < delay, else 1 - exp(-(t-delay)/tau)

    Args:
        t_ms: Time array in ms (must start at 0, i.e., peak-centered)
        z: Recovery coordinate array (0 at peak, 1 at baseline)
        model_set: Set of model names to fit (default: all 5)
        maxfev: Max function evaluations per fit

    Returns:
        Dictionary of model_name -> FitResult
    """
    if model_set is None:
        model_set = {'exponential', 'exponential_fixed', 'power_law', 'sigmoid', 'delayed'}

    results = {}

    if len(z) < 20:
        return results

    # z starts near 0 at peak, rises toward 1 at baseline
    min_z = z[0]
    amplitude_z = 1.0 - min_z  # Should be ~1 for well-normalized data

    if amplitude_z < 0.1:
        return results  # Insufficient amplitude in z-space

    # --- Model functions in z-space ---

    def z_exponential(t, tau):
        """z(t) = 1 - exp(-t/tau)"""
        return 1 - np.exp(-t / tau)

    def z_power_law(t, tau):
        """z(t) = t / (t + tau)"""
        return t / (t + tau)

    def z_sigmoid(t, k, t0):
        """z(t) = 1 / (1 + exp(-k*(t - t0)))"""
        return 1 / (1 + np.exp(-k * (t - t0)))

    def z_delayed(t, tau, delay):
        """z(t) = 0 for t < delay, else 1 - exp(-(t-delay)/tau)"""
        return np.where(t < delay, 0.0, 1 - np.exp(-(t - delay) / tau))

    # --- Fit each model ---

    if 'exponential' in model_set or 'exponential_fixed' in model_set:
        # Both use same function, just different naming
        try:
            popt, _ = curve_fit(z_exponential, t_ms, z, p0=[15.0],
                               bounds=([0.5], [180]), maxfev=maxfev)
            y_fit = z_exponential(t_ms, *popt)
            residuals = z - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((z - np.mean(z))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            n = len(z)
            n_params = 1
            aicc = compute_aicc(ss_res, n, n_params)

            fit_result = FitResult(
                model_name='exponential',
                params={'tau': popt[0]},
                fit_values=y_fit,
                residuals=residuals,
                ss_res=ss_res,
                r2=r2,
                aic=compute_aic(ss_res, n, n_params),
                aicc=aicc,
                n_params=n_params,
                success=True
            )
            if 'exponential' in model_set:
                results['exponential'] = fit_result
            if 'exponential_fixed' in model_set:
                # Clone for exponential_fixed (same fit in z-space)
                results['exponential_fixed'] = FitResult(
                    model_name='exponential_fixed',
                    params={'tau': popt[0]},
                    fit_values=y_fit,
                    residuals=residuals,
                    ss_res=ss_res,
                    r2=r2,
                    aic=compute_aic(ss_res, n, n_params),
                    aicc=aicc,
                    n_params=n_params,
                    success=True
                )
        except Exception as e:
            fail = FitResult(
                model_name='exponential',
                params={},
                fit_values=np.zeros_like(z),
                residuals=z.copy(),
                ss_res=np.sum(z**2),
                r2=0,
                aic=np.inf,
                aicc=np.inf,
                n_params=1,
                success=False,
                error_msg=str(e)
            )
            if 'exponential' in model_set:
                results['exponential'] = fail
            if 'exponential_fixed' in model_set:
                results['exponential_fixed'] = FitResult(
                    model_name='exponential_fixed', params={},
                    fit_values=np.zeros_like(z), residuals=z.copy(),
                    ss_res=np.sum(z**2), r2=0, aic=np.inf, aicc=np.inf,
                    n_params=1, success=False, error_msg=str(e)
                )

    if 'power_law' in model_set:
        try:
            # tau bound [0.5, 180] aligned with sanity filter
            popt, _ = curve_fit(z_power_law, t_ms, z, p0=[10.0],
                               bounds=([0.5], [180]), maxfev=maxfev)
            y_fit = z_power_law(t_ms, *popt)
            residuals = z - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((z - np.mean(z))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            n = len(z)
            n_params = 1

            results['power_law'] = FitResult(
                model_name='power_law',
                params={'tau': popt[0]},
                fit_values=y_fit,
                residuals=residuals,
                ss_res=ss_res,
                r2=r2,
                aic=compute_aic(ss_res, n, n_params),
                aicc=compute_aicc(ss_res, n, n_params),
                n_params=n_params,
                success=True
            )
        except Exception as e:
            results['power_law'] = FitResult(
                model_name='power_law', params={},
                fit_values=np.zeros_like(z), residuals=z.copy(),
                ss_res=np.sum(z**2), r2=0, aic=np.inf, aicc=np.inf,
                n_params=1, success=False, error_msg=str(e)
            )

    if 'sigmoid' in model_set:
        try:
            # Bounds for sigmoid: k in [0.01, 2], t0 in [1, 80]
            sigmoid_bounds = {'k': (0.01, 2), 't0': (1, 80)}
            popt, _ = curve_fit(z_sigmoid, t_ms, z, p0=[0.2, 15.0],
                               bounds=([0.01, 1], [2, 80]), maxfev=maxfev)
            y_fit = z_sigmoid(t_ms, *popt)
            residuals = z - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((z - np.mean(z))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            n = len(z)
            n_params = 2

            params = {'k': popt[0], 't0': popt[1]}
            # Use eps_frac=0.0 + eps_abs=0.5 ms for consistent time-param boundary detection
            hit_bounds, hit_params = check_bound_hits(params, sigmoid_bounds, eps_frac=0.0, eps_abs=0.5)

            results['sigmoid'] = FitResult(
                model_name='sigmoid',
                params=params,
                fit_values=y_fit,
                residuals=residuals,
                ss_res=ss_res,
                r2=r2,
                aic=compute_aic(ss_res, n, n_params),
                aicc=compute_aicc(ss_res, n, n_params),
                n_params=n_params,
                success=True,
                hit_bounds=hit_bounds,
                hit_bounds_params=hit_params
            )
        except Exception as e:
            results['sigmoid'] = FitResult(
                model_name='sigmoid', params={},
                fit_values=np.zeros_like(z), residuals=z.copy(),
                ss_res=np.sum(z**2), r2=0, aic=np.inf, aicc=np.inf,
                n_params=2, success=False, error_msg=str(e)
            )

    if 'delayed' in model_set:
        try:
            # Bounds for delayed: tau in [0.5, 180], delay in [0, 80]
            delayed_bounds = {'tau': (0.5, 180), 'delay': (0, 80)}
            popt, _ = curve_fit(z_delayed, t_ms, z, p0=[15.0, 5.0],
                               bounds=([0.5, 0], [180, 80]), maxfev=maxfev)
            y_fit = z_delayed(t_ms, *popt)
            residuals = z - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((z - np.mean(z))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            n = len(z)
            n_params = 2

            params = {'tau': popt[0], 'delay': popt[1]}
            # Use eps_frac=0.0 + eps_abs=0.5 ms for consistent time-param boundary detection
            hit_bounds, hit_params = check_bound_hits(params, delayed_bounds, eps_frac=0.0, eps_abs=0.5)

            results['delayed'] = FitResult(
                model_name='delayed',
                params=params,
                fit_values=y_fit,
                residuals=residuals,
                ss_res=ss_res,
                r2=r2,
                aic=compute_aic(ss_res, n, n_params),
                aicc=compute_aicc(ss_res, n, n_params),
                n_params=n_params,
                success=True,
                hit_bounds=hit_bounds,
                hit_bounds_params=hit_params
            )
        except Exception as e:
            results['delayed'] = FitResult(
                model_name='delayed', params={},
                fit_values=np.zeros_like(z), residuals=z.copy(),
                ss_res=np.sum(z**2), r2=0, aic=np.inf, aicc=np.inf,
                n_params=2, success=False, error_msg=str(e)
            )

    return results


def fit_envelope_with_baseline(
    t_ms: np.ndarray,
    env_fit: np.ndarray,
    baseline: float,
    model_set: Optional[set] = None,
    maxfev: int = 10000
) -> Dict[str, FitResult]:
    """
    Convenience wrapper: transforms envelope to z, then calls canonical fitter.

    Baseline is AUTHORITATIVE — never recomputed internally.
    This ensures no flag can cause baseline drift.

    Args:
        t_ms: Time array in ms (must start at 0, peak-centered)
        env_fit: Envelope values in fit window (must start at peak)
        baseline: Baseline value from baseline_tail_median()
        model_set: Set of models to fit (default: all)
        maxfev: Max function evaluations

    Returns:
        Dictionary of model_name -> FitResult
    """
    max_value = env_fit[0]  # Peak is first sample
    amplitude = max_value - baseline

    # Amplitude gate: reject if amplitude < 1% of peak
    if amplitude < 0.01 * max_value:
        return {}

    # Canonical z-transform with epsilon
    z = 1 - (env_fit - baseline) / (max_value - baseline + 1e-30)
    z = np.clip(z, 0, 1.5)

    return fit_competing_models_z(t_ms, z, model_set, maxfev)


# =============================================================================
# Core Functions
# =============================================================================

def compute_robust_baseline(data: np.ndarray) -> Tuple[float, float]:
    """
    Compute robust baseline using median and MAD.

    Returns:
        baseline: Median of data
        sigma: Robust standard deviation (1.4826 * MAD)
    """
    baseline = np.median(data)
    mad = np.median(np.abs(data - baseline))
    sigma = 1.4826 * mad  # Scale to standard deviation equivalent
    return baseline, sigma


def detect_events(
    data: np.ndarray,
    time_ms: np.ndarray,
    params: EventDetectionParams
) -> List[Event]:
    """
    Detect events (drops or rises) in time series data.

    Uses robust statistics (median + MAD) for threshold computation.

    Args:
        data: Signal values
        time_ms: Time array in milliseconds
        params: Detection parameters

    Returns:
        List of detected Event objects
    """
    baseline, sigma = compute_robust_baseline(data)

    # Compute threshold based on direction
    if params.detection_direction == 'drop':
        threshold = baseline - params.threshold_sigma * sigma
        threshold = min(threshold, baseline - params.min_drop_depth)
        below_threshold = data < threshold
    else:  # 'rise'
        threshold = baseline + params.threshold_sigma * sigma
        threshold = max(threshold, baseline + params.min_drop_depth)
        below_threshold = data > threshold

    # Estimate sampling interval
    if len(time_ms) > 1:
        dt_ms = np.median(np.diff(time_ms))
    else:
        dt_ms = 1.0

    min_separation_samples = int(params.min_separation_ms / dt_ms)

    # Find contiguous regions
    events = []
    in_event = False
    event_start = 0
    last_event_end = -min_separation_samples

    for i in range(len(data)):
        if below_threshold[i] and not in_event:
            if i - last_event_end >= min_separation_samples:
                in_event = True
                event_start = i
        elif not below_threshold[i] and in_event:
            in_event = False
            last_event_end = i

            # Find extremum within this event
            event_region = data[event_start:i]
            if params.detection_direction == 'drop':
                extremum_idx = event_start + np.argmin(event_region)
            else:
                extremum_idx = event_start + np.argmax(event_region)

            extremum_val = data[extremum_idx]
            amplitude = abs(baseline - extremum_val)

            events.append(Event(
                index=extremum_idx,
                time_ms=time_ms[extremum_idx],
                value=extremum_val,
                baseline=baseline,
                amplitude=amplitude
            ))

    # Handle EOF: if trace ends while in_event, close the event
    if in_event:
        event_region = data[event_start:]
        if len(event_region) > 0:
            if params.detection_direction == 'drop':
                extremum_idx = event_start + np.argmin(event_region)
            else:
                extremum_idx = event_start + np.argmax(event_region)

            extremum_val = data[extremum_idx]
            amplitude = abs(baseline - extremum_val)

            events.append(Event(
                index=extremum_idx,
                time_ms=time_ms[extremum_idx],
                value=extremum_val,
                baseline=baseline,
                amplitude=amplitude
            ))

    return events


def extract_window(
    data: np.ndarray,
    time_ms: np.ndarray,
    event: Event,
    window_ms: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract recovery window starting from event.

    Args:
        data: Full signal array
        time_ms: Full time array in milliseconds
        event: Event object with index
        window_ms: Window duration in milliseconds

    Returns:
        t_window: Time array starting from 0
        y_window: Data array for window
    """
    # Estimate sampling interval
    if len(time_ms) > 1:
        dt_ms = np.median(np.diff(time_ms))
    else:
        dt_ms = 1.0

    window_samples = int(window_ms / dt_ms)
    end_idx = min(event.index + window_samples, len(data))

    t_window = time_ms[event.index:end_idx] - time_ms[event.index]
    y_window = data[event.index:end_idx]

    return t_window, y_window


def apply_smoothing(
    data: np.ndarray,
    time_ms: np.ndarray,
    params: SmoothingParams
) -> np.ndarray:
    """
    Apply smoothing to data.

    Args:
        data: Signal values
        time_ms: Time array in milliseconds
        params: Smoothing parameters

    Returns:
        Smoothed data array
    """
    if params.method == 'none':
        return data.copy()

    # Compute window size in samples
    if len(time_ms) > 1:
        dt_ms = np.median(np.diff(time_ms))
    else:
        dt_ms = 1.0

    window_samples = max(3, int(params.window_ms / dt_ms))
    # Ensure odd window for savgol
    if window_samples % 2 == 0:
        window_samples += 1

    if params.method == 'moving_avg':
        kernel = np.ones(window_samples) / window_samples
        # Pad to maintain length
        smoothed = np.convolve(data, kernel, mode='same')
        return smoothed

    elif params.method == 'savgol':
        if len(data) < window_samples:
            return data.copy()
        order = min(params.savgol_order, window_samples - 1)
        return savgol_filter(data, window_samples, order)

    return data.copy()


def compute_derivative(
    data: np.ndarray,
    time_ms: np.ndarray,
    params: DerivativeParams
) -> np.ndarray:
    """
    Compute derivative of data.

    Args:
        data: Signal values
        time_ms: Time array in milliseconds
        params: Derivative parameters

    Returns:
        Derivative array (same length as input)
    """
    # Optionally smooth first
    if params.smooth_first:
        data = apply_smoothing(data, time_ms, params.smoothing)

    if params.method == 'gradient':
        return np.gradient(data, time_ms)

    elif params.method == 'central':
        deriv = np.zeros_like(data)
        deriv[1:-1] = (data[2:] - data[:-2]) / (time_ms[2:] - time_ms[:-2])
        deriv[0] = deriv[1]
        deriv[-1] = deriv[-2]
        return deriv

    elif params.method == 'savgol':
        # Compute window size in samples
        if len(time_ms) > 1:
            dt_ms = np.median(np.diff(time_ms))
        else:
            dt_ms = 1.0
        window_samples = max(5, int(params.smoothing.window_ms / dt_ms))
        if window_samples % 2 == 0:
            window_samples += 1
        if len(data) < window_samples:
            return np.gradient(data, time_ms)
        order = min(params.smoothing.savgol_order, window_samples - 1)
        return savgol_filter(data, window_samples, order, deriv=1, delta=dt_ms)

    return np.gradient(data, time_ms)


def compute_t_peak(
    t_ms: np.ndarray,
    y: np.ndarray,
    params: DerivativeParams,
    direction: str = 'recovery'
) -> float:
    """
    Compute time of steepest slope (t_peak).

    For recovery (upward): finds maximum positive derivative
    For decay (downward): finds minimum (most negative) derivative

    Args:
        t_ms: Time array in milliseconds
        y: Data array
        params: Derivative parameters
        direction: 'recovery' (upward) or 'decay' (downward)

    Returns:
        t_peak in milliseconds
    """
    deriv = compute_derivative(y, t_ms, params)

    # Limit search to early time (first 30ms typically)
    early_mask = t_ms <= 30
    if not np.any(early_mask):
        early_mask = np.ones(len(t_ms), dtype=bool)

    if direction == 'recovery':
        # Steepest upward slope = maximum derivative
        peak_idx = np.argmax(deriv[early_mask])
    else:
        # Steepest downward slope = minimum derivative
        peak_idx = np.argmin(deriv[early_mask])

    return t_ms[early_mask][peak_idx]


def compute_t_peak_bootstrap(
    t_ms: np.ndarray,
    y: np.ndarray,
    params: DerivativeParams,
    direction: str = 'recovery',
    n_bootstrap: int = 100,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute t_peak with bootstrap confidence interval.

    Uses residual resampling: fits smooth curve, resamples residuals.

    Args:
        t_ms: Time array in milliseconds
        y: Data array
        params: Derivative parameters
        direction: 'recovery' or 'decay'
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (0-1)
        seed: Random seed for reproducibility

    Returns:
        t_peak: Point estimate
        ci: (lower, upper) confidence interval
    """
    np.random.seed(seed)

    # Get point estimate
    t_peak = compute_t_peak(t_ms, y, params, direction)

    # Smooth curve for residuals
    smoothed = apply_smoothing(y, t_ms, params.smoothing)
    residuals = y - smoothed

    # Bootstrap
    t_peaks = []
    for _ in range(n_bootstrap):
        # Resample residuals
        resampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
        y_boot = smoothed + resampled_residuals

        t_peak_boot = compute_t_peak(t_ms, y_boot, params, direction)
        t_peaks.append(t_peak_boot)

    # Compute confidence interval
    alpha = 1 - confidence
    lower = np.percentile(t_peaks, 100 * alpha / 2)
    upper = np.percentile(t_peaks, 100 * (1 - alpha / 2))

    return t_peak, (lower, upper)


def compute_aic(ss_res: float, n: int, k: int) -> float:
    """Compute AIC.

    Note: Uses epsilon floor for ss_res to handle near-perfect fits.
    Returning inf for ss_res <= 0 would make the best fit rank worst.
    """
    if n <= 0:
        return np.inf
    ss = max(float(ss_res), 1e-30)  # epsilon floor for near-perfect fits
    return n * np.log(ss / n) + 2 * k


def compute_aicc(ss_res: float, n: int, k: int) -> float:
    """Compute corrected AIC (AICc) for small samples."""
    aic = compute_aic(ss_res, n, k)
    if n - k - 1 <= 0:
        return np.inf
    correction = (2 * k * (k + 1)) / (n - k - 1)
    return aic + correction


def fit_model(
    t_ms: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    p0: List[float],
    bounds: Tuple[List[float], List[float]],
    model_name: str,
    n_params: int,
    maxfev: int = 10000
) -> FitResult:
    """
    Fit a single model to data.

    Args:
        t_ms: Time array
        y: Data array
        model_func: Model function
        p0: Initial parameters
        bounds: Parameter bounds (lower, upper)
        model_name: Name for reporting
        n_params: Number of parameters
        maxfev: Max function evaluations

    Returns:
        FitResult object
    """
    try:
        popt, _ = curve_fit(model_func, t_ms, y, p0=p0, bounds=bounds, maxfev=maxfev)

        y_fit = model_func(t_ms, *popt)
        residuals = y - y_fit
        ss_res = np.sum(residuals**2)

        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        n = len(y)
        aic = compute_aic(ss_res, n, n_params)
        aicc = compute_aicc(ss_res, n, n_params)

        # Extract parameter names from bounds structure
        param_names = [f'p{i}' for i in range(len(popt))]
        if model_name == 'exponential':
            param_names = ['A', 'tau', 'baseline'][:len(popt)]
        elif model_name == 'exponential_fixed':
            param_names = ['A', 'tau']
        elif model_name == 'power_law':
            param_names = ['A', 'tau', 'baseline'][:len(popt)]
        elif model_name == 'power_law_fixed':
            param_names = ['A', 'tau']
        elif model_name == 'sigmoid':
            param_names = ['A', 'k', 't0', 'minimum']
        elif model_name == 'delayed':
            param_names = ['A', 'tau', 'delay', 'minimum']

        params = {name: val for name, val in zip(param_names, popt)}

        # Build bounds dict for bound-hit detection
        # bounds is tuple: ([lo1, lo2, ...], [hi1, hi2, ...])
        bounds_dict = {}
        lower_bounds, upper_bounds = bounds
        for i, name in enumerate(param_names):
            if i < len(lower_bounds) and i < len(upper_bounds):
                bounds_dict[name] = (lower_bounds[i], upper_bounds[i])

        # Use eps_frac=0.0 + eps_abs=0.5 for consistent boundary detection across all fitters
        hit_bounds, hit_bounds_params = check_bound_hits(params, bounds_dict, eps_frac=0.0, eps_abs=0.5)

        return FitResult(
            model_name=model_name,
            params=params,
            fit_values=y_fit,
            residuals=residuals,
            ss_res=ss_res,
            r2=r2,
            aic=aic,
            aicc=aicc,
            n_params=n_params,
            success=True,
            hit_bounds=hit_bounds,
            hit_bounds_params=hit_bounds_params
        )

    except Exception as e:
        return FitResult(
            model_name=model_name,
            params={},
            fit_values=np.zeros_like(y),
            residuals=y.copy(),
            ss_res=np.sum(y**2),
            r2=0,
            aic=np.inf,
            aicc=np.inf,
            n_params=n_params,
            success=False,
            error_msg=str(e)
        )


def fit_competing_models(
    t_ms: np.ndarray,
    y: np.ndarray,
    baseline_estimate: float,
    params: FitParams
) -> Dict[str, FitResult]:
    """
    Fit all competing models to recovery data.

    Models:
    - exponential: Standard physics (free baseline)
    - exponential_fixed: Standard physics (fixed baseline)
    - power_law: Standard physics alternative
    - sigmoid: IOF physics
    - delayed: IOF physics alternative

    Args:
        t_ms: Time array in milliseconds
        y: Data array
        baseline_estimate: Estimated baseline value
        params: Fitting parameters

    Returns:
        Dictionary of model_name -> FitResult
    """
    results = {}

    min_value = y[0]
    amplitude = abs(baseline_estimate - min_value)

    # Relative amplitude gate: amplitude must be meaningful relative to baseline
    # Use 1% of |baseline| as threshold, with absolute floor of 0.01 for near-zero baselines
    min_amplitude = max(0.01 * abs(baseline_estimate), 0.01)
    if amplitude < min_amplitude:
        return results

    # Compute fixed baseline from late-time data if requested
    if params.fix_baseline:
        late_start = int(len(y) * (1 - params.baseline_fraction))
        fixed_baseline = np.median(y[late_start:])
    else:
        fixed_baseline = baseline_estimate

    # Compute safe bounds that work for negative signals too
    # Bracket baseline around estimate, not around zero
    bl0 = baseline_estimate
    bl_lo = bl0 - 2 * amplitude
    bl_hi = bl0 + 2 * amplitude

    # For minimum param (sigmoid/delayed): bracket around min_value
    min_lo = min(min_value, np.min(y)) - amplitude
    min_hi = max(baseline_estimate, np.max(y)) + amplitude

    # --- Exponential (free baseline) ---
    # Note: tau bound [0.5, 180] aligned with z-domain fitter and sanity filter
    results['exponential'] = fit_model(
        t_ms, y,
        exponential_recovery,
        p0=[amplitude, 10.0, baseline_estimate],
        bounds=([0, 0.5, bl_lo], [amplitude * 2, 180, bl_hi]),
        model_name='exponential',
        n_params=3,
        maxfev=params.maxfev
    )

    # --- Exponential (fixed baseline) ---
    results['exponential_fixed'] = fit_model(
        t_ms, y,
        exponential_recovery_fixed(fixed_baseline),
        p0=[amplitude, 10.0],
        bounds=([0, 0.5], [amplitude * 2, 180]),
        model_name='exponential_fixed',
        n_params=2,
        maxfev=params.maxfev
    )

    # --- Power Law (free baseline) ---
    results['power_law'] = fit_model(
        t_ms, y,
        power_law_recovery,
        p0=[amplitude, 5.0, baseline_estimate],
        bounds=([0, 0.5, bl_lo], [amplitude * 2, 180, bl_hi]),
        model_name='power_law',
        n_params=3,
        maxfev=params.maxfev
    )

    # --- Sigmoid (IOF) ---
    results['sigmoid'] = fit_model(
        t_ms, y,
        logistic_sigmoid,
        p0=[amplitude, 0.2, 15.0, min_value],
        bounds=([0, 0.01, 1, min_lo], [amplitude * 2, 2, 80, min_hi]),
        model_name='sigmoid',
        n_params=4,
        maxfev=params.maxfev
    )

    # --- Delayed Exponential (IOF) ---
    # Note: tau=[0.5, 180], delay=[0, 80] aligned with z-domain fitter and sanity filter
    results['delayed'] = fit_model(
        t_ms, y,
        delayed_exponential,
        p0=[amplitude, 15.0, 5.0, min_value],
        bounds=([0, 0.5, 0, min_lo], [amplitude * 2, 180, 80, min_hi]),
        model_name='delayed',
        n_params=4,
        maxfev=params.maxfev
    )

    return results


def classify_event_model_based(
    fits: Dict[str, FitResult],
    params: ClassificationParams
) -> Tuple[str, str, str, float, float]:
    """
    Classify event using model-based approach (Option A).

    This is the methodologically correct approach:
    1. Select winning model by AICc
    2. Compute t_peak analytically from winning model parameters
    3. Classify by model geometry (fast vs delayed)
    4. Handle uncertainty when models are comparable (ΔAICc < 2)

    Args:
        fits: Dictionary of fit results
        params: Classification parameters

    Returns:
        classification: 'standard', 'iof', or 'uncertain'
        reason: Explanation string
        winning_model: Name of best-fit model
        model_t_peak: t_peak from winning model (ms)
        delta_aicc: AICc difference between best and second-best
    """
    # Get successful AND sane fits only
    # check_fit_sanity() filters garbage geometry (low R², absurd tau, etc.)
    valid_fits = {}
    for k, v in fits.items():
        if not v.success:
            continue
        ok, why = check_fit_sanity(v)
        if ok:
            valid_fits[k] = v

    if not valid_fits:
        return 'uncertain', 'no successful fits or all failed sanity', 'none', 0.0, 0.0

    # Find best model by AICc
    sorted_fits = sorted(valid_fits.items(), key=lambda x: x[1].aicc)
    best_name, best_fit = sorted_fits[0]

    # Compute delta AICc to second-best (if available)
    if len(sorted_fits) >= 2:
        second_name, second_fit = sorted_fits[1]
        delta_aicc = second_fit.aicc - best_fit.aicc
    else:
        delta_aicc = float('inf')

    # Compute model-based t_peak from winning model
    model_t_peak = compute_model_t_peak(best_name, best_fit.params)

    # Get model geometry
    geometry = get_model_geometry(best_name)

    # Classification logic
    reasons = []

    # Check for bound-hit pathology (optimizer parking at boundary)
    # This catches the SI_Fig12a-style failure where t0/delay pegs at grid max
    if best_fit.hit_bounds and geometry == 'delayed':
        critical_params = ['t0', 'delay']
        critical_hits = [p for p in best_fit.hit_bounds_params if p in critical_params]
        if critical_hits:
            # Delay parameter hit bounds - suspect unless overwhelming evidence
            if delta_aicc < 10.0:
                reasons.append(f"bound-hit on {critical_hits} (ΔAICc={delta_aicc:.1f}<10)")
                reasons.append("non-identifiable delay - forcing uncertain")
                return 'uncertain', "; ".join(reasons), best_name, model_t_peak, delta_aicc
            else:
                reasons.append(f"WARNING: bound-hit on {critical_hits} but ΔAICc={delta_aicc:.1f}≥10")

    # Check if model selection is confident (ΔAICc >= 2)
    if delta_aicc < 2.0:
        # Models are comparable - check if they agree on geometry
        geometries = [get_model_geometry(name) for name, _ in sorted_fits[:2]]
        if geometries[0] == geometries[1]:
            # Both top models have same geometry - confident classification
            reasons.append(f"top models agree on geometry ({geometry})")
        else:
            # Models disagree on geometry - uncertain
            reasons.append(f"model selection ambiguous (ΔAICc={delta_aicc:.1f}<2)")
            reasons.append(f"best={best_name} ({geometry}), second={second_name} ({geometries[1]})")
            return 'uncertain', "; ".join(reasons), best_name, model_t_peak, delta_aicc

    # Classify based on geometry
    if geometry == 'fast':
        classification = 'standard'
        reasons.append(f"fast geometry ({best_name})")
        reasons.append(f"model t_peak={model_t_peak:.1f}ms")
    elif geometry == 'delayed':
        classification = 'iof'
        reasons.append(f"delayed geometry ({best_name})")
        reasons.append(f"model t_peak={model_t_peak:.1f}ms")
    else:
        classification = 'uncertain'
        reasons.append(f"unknown geometry ({best_name})")

    reasons.append(f"ΔAICc={delta_aicc:.1f}")

    return classification, "; ".join(reasons), best_name, model_t_peak, delta_aicc


def classify_event(
    t_peak_ms: float,
    fits: Dict[str, FitResult],
    params: ClassificationParams
) -> Tuple[str, str]:
    """
    Classify event as 'standard' or 'iof' (legacy derivative-based).

    DEPRECATED: Use classify_event_model_based() instead.
    This function is kept for backward compatibility and diagnostics.

    Classification criteria:
    1. t_peak > threshold -> IOF (derivative-based, sensitive to smoothing)
    2. Sigmoid AICc significantly better than exponential -> IOF

    Args:
        t_peak_ms: Time of steepest slope (from derivative)
        fits: Dictionary of fit results
        params: Classification parameters

    Returns:
        classification: 'standard' or 'iof'
        reason: Explanation string
    """
    reasons = []
    iof_votes = 0

    # Criterion 1: t_peak (derivative-based - diagnostic only)
    if t_peak_ms > params.legacy_t_peak_threshold_ms:
        iof_votes += 1
        reasons.append(f"deriv_t_peak={t_peak_ms:.1f}ms > {params.legacy_t_peak_threshold_ms}ms")
    else:
        reasons.append(f"deriv_t_peak={t_peak_ms:.1f}ms <= {params.legacy_t_peak_threshold_ms}ms")

    # Criterion 2: AICc comparison (use fixed baseline exponential for fair comparison)
    exp_key = 'exponential_fixed' if 'exponential_fixed' in fits else 'exponential'
    if exp_key in fits and 'sigmoid' in fits:
        exp_fit = fits[exp_key]
        sig_fit = fits['sigmoid']

        if exp_fit.success and sig_fit.success:
            delta_aicc = exp_fit.aicc - sig_fit.aicc
            if delta_aicc > params.aic_threshold:
                iof_votes += 1
                reasons.append(f"sigmoid preferred (dAICc={delta_aicc:.1f})")
            elif delta_aicc < -params.aic_threshold:
                reasons.append(f"exponential preferred (dAICc={delta_aicc:.1f})")
            else:
                reasons.append(f"models comparable (dAICc={delta_aicc:.1f})")

    classification = 'iof' if iof_votes >= 1 else 'standard'
    reason = "; ".join(reasons)

    return classification, reason


def analyze_event(
    data: np.ndarray,
    time_ms: np.ndarray,
    event: Event,
    detection_params: EventDetectionParams,
    derivative_params: DerivativeParams,
    fit_params: FitParams,
    classification_params: ClassificationParams,
    window_ms: float = 100.0,
    n_bootstrap: int = 100,
    seed: int = 42
) -> EventAnalysis:
    """
    Complete analysis of a single event.

    Uses model-based classification (Option A) as the primary method:
    - Fits competing models (exponential, sigmoid, delayed)
    - Selects winner by AICc
    - Computes t_peak analytically from winning model
    - Classifies by model geometry (fast vs delayed)

    Derivative-based t_peak is computed as a diagnostic only.

    Args:
        data: Full signal array
        time_ms: Full time array
        event: Event to analyze
        detection_params: Event detection parameters
        derivative_params: Derivative computation parameters
        fit_params: Model fitting parameters
        classification_params: Classification parameters
        window_ms: Analysis window duration
        n_bootstrap: Bootstrap iterations for derivative t_peak CI
        seed: Random seed

    Returns:
        EventAnalysis object with all results
    """
    # Extract window
    t_window, y_window = extract_window(data, time_ms, event, window_ms)

    if len(t_window) < 10:
        raise ValueError("Insufficient data in window")

    # Fit competing models (this is the primary analysis)
    fits = fit_competing_models(t_window, y_window, event.baseline, fit_params)

    # Model-based classification (primary - methodologically correct)
    classification, reason, winning_model, model_t_peak, delta_aicc = \
        classify_event_model_based(fits, classification_params)

    # Derivative-based t_peak (diagnostic only - kept for comparison)
    deriv_t_peak, deriv_t_peak_ci = compute_t_peak_bootstrap(
        t_window, y_window, derivative_params,
        direction='recovery', n_bootstrap=n_bootstrap, seed=seed
    )

    # Normalize curve for aggregation
    y_norm = (y_window - event.value) / (event.baseline - event.value + 1e-10)

    # Interpolate to common grid
    common_t = np.linspace(0, window_ms, 1000)
    y_interp = np.interp(common_t, t_window, y_norm)

    # Compute parameter hash
    all_params = {
        'detection': detection_params.to_dict(),
        'derivative': derivative_params.to_dict(),
        'fit': fit_params.to_dict(),
        'classification': classification_params.to_dict()
    }
    params_hash = hashlib.md5(json.dumps(all_params, sort_keys=True).encode()).hexdigest()[:12]

    return EventAnalysis(
        event=event,
        classification=classification,
        classification_reason=reason,
        winning_model=winning_model,
        model_t_peak_ms=model_t_peak,
        delta_aicc=delta_aicc,
        deriv_t_peak_ms=deriv_t_peak,
        deriv_t_peak_ci=deriv_t_peak_ci,
        fits=fits,
        normalized_curve=y_interp,
        time_grid_ms=common_t,
        params_hash=params_hash
    )


# =============================================================================
# Robustness Sweep
# =============================================================================

@dataclass
class SweepConfig:
    """Configuration for parameter sweep.

    Note: t_peak_thresholds_ms removed - model-based classification uses
    winning model geometry, not a threshold on derivative-based t_peak.
    """
    threshold_sigmas: List[float] = field(default_factory=lambda: [4.0, 5.0, 6.0, 7.0, 8.0])
    smoothing_methods: List[str] = field(default_factory=lambda: ['none', 'moving_avg', 'savgol'])
    smoothing_windows_ms: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0])
    derivative_methods: List[str] = field(default_factory=lambda: ['gradient', 'central', 'savgol'])
    window_lengths_ms: List[float] = field(default_factory=lambda: [50.0, 100.0, 150.0, 200.0])
    fix_baseline_options: List[bool] = field(default_factory=lambda: [True, False])


@dataclass
class SweepResult:
    """Result from a single sweep configuration."""
    config: Dict[str, Any]
    n_events: int
    n_iof: int
    iof_fraction: float
    mean_t_peak_ms: float
    std_t_peak_ms: float


def run_robustness_sweep(
    data: np.ndarray,
    time_ms: np.ndarray,
    sweep_config: SweepConfig,
    base_detection_params: EventDetectionParams,
    seed: int = 42
) -> List[SweepResult]:
    """
    Run parameter robustness sweep.

    Tests IOF classification stability across parameter variations.

    Args:
        data: Signal array
        time_ms: Time array
        sweep_config: Sweep configuration
        base_detection_params: Base detection parameters
        seed: Random seed

    Returns:
        List of SweepResult objects
    """
    results = []

    # Generate all parameter combinations
    # Note: t_peak_threshold_ms removed - model-based classification doesn't use it
    configs = []
    for thresh in sweep_config.threshold_sigmas:
        for smooth_method in sweep_config.smoothing_methods:
            for smooth_window in sweep_config.smoothing_windows_ms:
                for deriv_method in sweep_config.derivative_methods:
                    for fix_baseline in sweep_config.fix_baseline_options:
                        for window_ms in sweep_config.window_lengths_ms:
                            configs.append({
                                'threshold_sigma': thresh,
                                'smoothing_method': smooth_method,
                                'smoothing_window_ms': smooth_window,
                                'derivative_method': deriv_method,
                                'fix_baseline': fix_baseline,
                                'window_ms': window_ms
                            })

    print(f"Running sweep over {len(configs)} configurations...")

    for i, config in enumerate(configs):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(configs)}")

        try:
            # Build parameter objects
            detection_params = EventDetectionParams(
                threshold_sigma=config['threshold_sigma'],
                min_drop_depth=base_detection_params.min_drop_depth,
                min_separation_ms=base_detection_params.min_separation_ms,
                detection_direction=base_detection_params.detection_direction
            )

            smoothing_params = SmoothingParams(
                method=config['smoothing_method'],
                window_ms=config['smoothing_window_ms']
            )

            derivative_params = DerivativeParams(
                method=config['derivative_method'],
                smooth_first=True,
                smoothing=smoothing_params
            )

            fit_params = FitParams(
                fix_baseline=config['fix_baseline']
            )

            # Model-based classification uses default params
            # (t_peak_threshold_ms only affects deprecated derivative-based classification)
            classification_params = ClassificationParams()

            # Detect events
            events = detect_events(data, time_ms, detection_params)

            if len(events) == 0:
                continue

            # Analyze each event
            t_peaks = []
            n_iof = 0

            for event in events:
                try:
                    analysis = analyze_event(
                        data, time_ms, event,
                        detection_params, derivative_params,
                        fit_params, classification_params,
                        window_ms=config['window_ms'],
                        n_bootstrap=20,  # Reduced for speed
                        seed=seed
                    )
                    # Use model_t_peak_ms (from winning model, methodologically correct)
                    t_peaks.append(analysis.model_t_peak_ms)
                    if analysis.classification == 'iof':
                        n_iof += 1
                except Exception:
                    continue

            if len(t_peaks) > 0:
                results.append(SweepResult(
                    config=config,
                    n_events=len(t_peaks),
                    n_iof=n_iof,
                    iof_fraction=n_iof / len(t_peaks),
                    mean_t_peak_ms=np.mean(t_peaks),
                    std_t_peak_ms=np.std(t_peaks)
                ))

        except Exception:
            continue

    return results


# =============================================================================
# Output Utilities
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results_jsonl(
    analyses: List[EventAnalysis],
    output_path: Path,
    dataset_name: str,
    dataset_hash: str
):
    """
    Save analysis results to JSON Lines format.

    One line per event, machine-readable.
    """
    with open(output_path, 'w') as f:
        for analysis in analyses:
            record = {
                'dataset': dataset_name,
                'dataset_hash': dataset_hash,
                **analysis.to_dict()
            }
            f.write(json.dumps(record, cls=NumpyEncoder) + '\n')


def compute_dataset_hash(data: np.ndarray) -> str:
    """Compute hash of dataset for reproducibility tracking."""
    return hashlib.md5(data.tobytes()).hexdigest()[:12]


def generate_sweep_summary(results: List[SweepResult]) -> Dict[str, Any]:
    """
    Generate summary statistics from sweep results.

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}

    iof_fractions = [r.iof_fraction for r in results]

    return {
        'n_configurations': len(results),
        'iof_fraction_mean': np.mean(iof_fractions),
        'iof_fraction_std': np.std(iof_fractions),
        'iof_fraction_min': np.min(iof_fractions),
        'iof_fraction_max': np.max(iof_fractions),
        'iof_fraction_median': np.median(iof_fractions),
        'iof_fraction_q25': np.percentile(iof_fractions, 25),
        'iof_fraction_q75': np.percentile(iof_fractions, 75)
    }
