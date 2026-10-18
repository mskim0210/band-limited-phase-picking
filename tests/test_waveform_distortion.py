"""Unit tests for scripts/waveform_distortion.py (R1-8 waveform-fidelity metrics)."""

from __future__ import annotations

import numpy as np

from scripts.waveform_distortion import (
    band_limited_reconstruct,
    frequency_response,
    noise_transfer_function,
    nrmse,
    onset_lag_samples,
)


def _signal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 2 Hz sits well inside the CR=4 pass band (db4 roll-off starts below the
    # nominal 12.5 Hz cutoff: a 5 Hz tone already loses ~5 %, see
    # test_frequency_response_is_monotone), 30 Hz is fully in the removed band.
    t = np.arange(6000) / 100.0
    low = np.sin(2 * np.pi * 2 * t)
    high = np.sin(2 * np.pi * 30 * t) * (t > 30)
    x = (low + high)[None, None, :].astype(np.float32)  # (1, 1, 6000)
    return x, low, high


def test_reconstruct_preserves_length_and_low_band() -> None:
    x, low, _ = _signal()
    xr = band_limited_reconstruct(x, cr=4)
    assert xr.shape == x.shape
    assert nrmse(xr[0, 0], low) < 0.05


def test_reconstruct_removes_high_band() -> None:
    x, low, high = _signal()
    xr = band_limited_reconstruct(x, cr=4)
    residual_energy = np.sum((xr[0, 0] - low) ** 2)
    assert residual_energy < 0.1 * np.sum(high ** 2)


def test_nrmse_identity_and_scale() -> None:
    a = np.random.RandomState(0).randn(1000)
    assert nrmse(a, a) == 0.0
    assert abs(nrmse(np.zeros_like(a), a) - 1.0) < 1e-12


def test_frequency_response_is_monotone() -> None:
    resp = frequency_response(cr=4, freqs_hz=(2.0, 5.0, 10.0, 20.0))
    assert list(resp.keys()) == [2.0, 5.0, 10.0, 20.0]
    vals = list(resp.values())
    assert vals == sorted(vals)            # attenuation grows with frequency
    assert vals[0] < 0.01 and vals[-1] > 0.9  # 2 Hz passes, 20 Hz is removed


def test_noise_transfer_function_pass_and_stop_bands() -> None:
    # Pure tones at the dyadic band edge (12.5 Hz at CR=4) alias coherently and
    # look artificially preserved; a white-noise PSD ratio gives the smooth,
    # edge-free transfer function used in the manuscript table.
    f, h = noise_transfer_function(cr=4, seed=0)
    assert f.shape == h.shape and f[0] >= 0.0
    assert h[(f > 1.0) & (f < 3.0)].mean() > 0.95
    assert h[(f > 18.0) & (f < 22.0)].mean() < 0.05
    assert np.all(h >= -1e-6) and np.all(h <= 1.0 + 1e-6)


def test_onset_lag_detects_known_shift() -> None:
    t = np.arange(6000) / 100.0
    x = np.exp(-((t - 30) ** 2) / 0.02) * np.sin(2 * np.pi * 8 * t)
    shifted = np.roll(x, 7)
    assert onset_lag_samples(x, shifted, onset=3000, half_window=100, max_lag=50) == 7
    assert onset_lag_samples(x, x, onset=3000, half_window=100, max_lag=50) == 0
