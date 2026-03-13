"""
Statistical analysis utilities for Yang-Mills Monte Carlo simulations.

Provides bootstrap/jackknife error estimation and autocorrelation analysis
needed to produce statistically valid results from Markov chain Monte Carlo.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional


def bootstrap_error(
    samples: np.ndarray,
    func: Callable = np.mean,
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """
    Estimate (mean, std_error) of func(samples) via bootstrap resampling.

    Parameters
    ----------
    samples    : 1-D array of measurements
    func       : statistic to estimate (default: mean)
    n_bootstrap: number of bootstrap resamples
    rng        : random generator for reproducibility

    Returns
    -------
    (central_value, bootstrap_std)
    """
    samples = np.asarray(samples, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    n = len(samples)
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(samples, size=n, replace=True)
        boot_stats[i] = func(resample)
    return float(func(samples)), float(np.std(boot_stats, ddof=1))


def jackknife_error(
    samples: np.ndarray,
    func: Callable = np.mean,
) -> tuple[float, float]:
    """
    Estimate (mean, std_error) of func(samples) via leave-one-out jackknife.

    More efficient than bootstrap for small samples; also gives bias correction.

    Returns
    -------
    (jackknife_mean, jackknife_std)
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    jack_stats = np.empty(n)
    for i in range(n):
        leave_one_out = np.concatenate([samples[:i], samples[i + 1:]])
        jack_stats[i] = func(leave_one_out)
    jack_mean = np.mean(jack_stats)
    jack_var = (n - 1) / n * np.sum((jack_stats - jack_mean) ** 2)
    return float(jack_mean), float(np.sqrt(jack_var))


def integrated_autocorrelation_time(
    series: np.ndarray,
    c: float = 5.0,
) -> float:
    """
    Estimate the integrated autocorrelation time τ_int of a time series.

    Uses the Madras-Sokal automatic windowing procedure: sum the normalized
    autocorrelation function Γ(t)/Γ(0) up to a window W where W < c * τ_int.

    Parameters
    ----------
    series : 1-D array of sequential measurements
    c      : safety factor for window (default 5, as in Madras & Sokal 1988)

    Returns
    -------
    tau_int : integrated autocorrelation time (>= 0.5 for white noise)
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 4:
        return 0.5

    x = series - np.mean(series)
    # Normalized autocorrelation via FFT for efficiency
    f = np.fft.rfft(x, n=2 * n)
    acf_raw = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf_raw / acf_raw[0]  # normalize so acf[0] = 1

    tau = 0.5
    for t in range(1, n):
        tau += acf[t]
        # Madras-Sokal window: stop when window exceeds c * tau
        if t >= c * tau:
            break

    return max(0.5, float(tau))


def effective_sample_size(series: np.ndarray, c: float = 5.0) -> int:
    """
    Compute the effective number of statistically independent samples.

    N_eff = N / (1 + 2 * tau_int)

    A Markov chain with high autocorrelation gives far fewer independent
    samples than the raw count suggests.

    Returns
    -------
    n_eff : effective sample size (>= 1)
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    tau = integrated_autocorrelation_time(series, c=c)
    n_eff = n / (1.0 + 2.0 * tau)
    return max(1, int(n_eff))
