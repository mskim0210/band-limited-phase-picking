"""Tests for scripts/band_energy_retention.py.

Eq. (2) defines E_r as the fraction of wavelet-coefficient energy carried by
the retained bands.  The script now also reports the time-domain ratio
||x~||^2 / ||x||^2 of the actual encoder output so that the two can be
compared instead of assumed equal (symmetric boundary extension makes the
decomposition only approximately orthonormal).
"""

from __future__ import annotations

import numpy as np
import pywt

from cs_pipeline.sensing_matrix import create_sensing_matrix
from scripts.band_energy_retention import DCT_M, component_retention, diff_stats

FS = 100.0
N = 6000


def test_dct_energy_uses_the_coefficient_count_the_encoder_keeps() -> None:
    """The DCT encoder keeps N // CR = 1500 coefficients and zero-pads to an
    output length of 1506; the energy fraction must count the kept
    coefficients, not the output length."""
    enc = create_sensing_matrix("dct", N=N, CR=4)
    assert enc.M == 1506 and enc._dct_keep == 1500
    assert DCT_M[4] == enc._dct_keep


def test_low_frequency_tone_is_fully_retained_in_both_measures() -> None:
    t = np.arange(N) / FS
    x = np.sin(2 * np.pi * 1.0 * t)
    coef, tdom = component_retention(x, keep=5, m=1506)
    assert 0.98 < coef <= 1.0 + 1e-9
    assert 0.98 < tdom <= 1.02


def test_periodization_mode_makes_both_measures_identical() -> None:
    rng = np.random.RandomState(0)
    x = rng.randn(6144)  # multiple of 2^6: no odd-length padding at any level
    coef, tdom = component_retention(x, keep=5, m=1536, mode="periodization")
    assert 0.18 < coef < 0.32  # 0–12.5 Hz of a 0–50 Hz white spectrum, db4 roll-off
    assert abs(coef - tdom) < 1e-9  # exactly orthonormal without boundary extension


def test_symmetric_extension_can_inflate_the_coefficient_fraction() -> None:
    """Boundary coefficients of the symmetric extension carry extra energy at
    coarse levels, so the coefficient fraction may exceed the energy ratio of
    the encoder output; the latter stays close to the orthonormal value.
    This is why the script reports both and their difference."""
    rng = np.random.RandomState(0)
    x = rng.randn(N)
    coef_s, tdom_s = component_retention(x, keep=5, m=1506)
    coef_p, _ = component_retention(x, keep=5, m=1500, mode="periodization")
    assert coef_s - tdom_s > 0.03
    assert abs(tdom_s - coef_p) < 0.02


def test_time_domain_ratio_uses_encoder_output_length() -> None:
    rng = np.random.RandomState(1)
    x = rng.randn(N)
    coeffs = pywt.wavedec(x, "db4", level=6, mode="symmetric")
    xt = pywt.waverec(coeffs[:5], "db4", mode="symmetric")[:1506]
    _, tdom = component_retention(x, keep=5, m=1506)
    assert np.isclose(tdom, (xt ** 2).sum() / (x ** 2).sum())


def test_diff_stats_reports_points_and_coverage() -> None:
    coef = np.array([0.70, 0.60, 0.50, 0.40])
    tdom = np.array([0.70, 0.601, 0.49, 0.40])
    s = diff_stats(coef, tdom, within_pts=0.3)
    assert s["mean_pts"] == -0.225  # mean of (0, +0.1, -1.0, 0)
    assert s["median_pts"] == 0.0
    assert s["within_frac"] == 0.75
    assert s["p5_pts"] <= s["median_pts"] <= s["p95_pts"]
    assert s["max_abs_pts"] == 1.0
