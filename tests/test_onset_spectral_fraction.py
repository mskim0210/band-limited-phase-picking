"""Tests for scripts/onset_spectral_fraction.py (per-trace onset energy above a cutoff).

One definition for both datasets (STEAD test set and the Korean borehole
windows): Z component, window [P, P + win_sec], Welch PSD, fraction of PSD
power above the cutoff frequency.
"""

from __future__ import annotations

import numpy as np

from scripts.onset_spectral_fraction import fraction_above, summarize_fractions

FS = 100


def _tone(freq_hz: float, n: int = 6000) -> np.ndarray:
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def test_high_frequency_tone_is_almost_entirely_above_cutoff() -> None:
    frac = fraction_above(_tone(30.0), p=1000, fs=FS, win_sec=3.0, fc_hz=12.5)
    assert frac > 0.95


def test_low_frequency_tone_is_almost_entirely_below_cutoff() -> None:
    frac = fraction_above(_tone(5.0), p=1000, fs=FS, win_sec=3.0, fc_hz=12.5)
    assert frac < 0.05


def test_window_outside_trace_returns_nan() -> None:
    assert np.isnan(fraction_above(_tone(5.0), p=5900, fs=FS, win_sec=3.0, fc_hz=12.5))
    assert np.isnan(fraction_above(_tone(5.0), p=-1, fs=FS, win_sec=3.0, fc_hz=12.5))


def test_summary_reports_median_mean_and_majority_share() -> None:
    s = summarize_fractions(np.array([0.2, 0.6, 0.7, np.nan]))
    assert s["n"] == 3
    assert abs(s["median"] - 0.6) < 1e-9
    assert abs(s["mean"] - 0.5) < 1e-9
    assert abs(s["majority_above"] - 2 / 3) < 1e-9
