"""Unit tests for cs_pipeline.sparsity_analysis module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cs_pipeline.sparsity_analysis import (
    analyze_sparsity,
    dct_coefficients,
    energy_retention,
    plot_sparsity,
    wavelet_coefficients,
)


# ---------------------------------------------------------------------------
# wavelet_coefficients
# ---------------------------------------------------------------------------


class TestWaveletCoefficients:
    def test_output_sorted_descending(self) -> None:
        """Returned magnitudes should be in descending order."""
        trace = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 6000))
        mags = wavelet_coefficients(trace)
        assert np.all(mags[:-1] >= mags[1:])

    def test_output_length(self) -> None:
        """Wavelet coefficient count should be close to input length."""
        trace = np.random.randn(6000)
        mags = wavelet_coefficients(trace, wavelet="db4", level=6)
        # pywt may produce slightly more coefficients due to padding
        assert abs(len(mags) - 6000) < 100

    def test_known_sparse_signal(self) -> None:
        """Delta function should concentrate energy in few coefficients."""
        trace = np.zeros(6000)
        trace[3000] = 1.0
        mags = wavelet_coefficients(trace)
        ret = energy_retention(mags, [0.05])
        assert ret[0] > 90.0  # top-5% should retain > 90% energy


# ---------------------------------------------------------------------------
# dct_coefficients
# ---------------------------------------------------------------------------


class TestDCTCoefficients:
    def test_output_sorted_descending(self) -> None:
        """Returned magnitudes should be in descending order."""
        trace = np.random.randn(6000)
        mags = dct_coefficients(trace)
        assert np.all(mags[:-1] >= mags[1:])

    def test_output_length_matches_input(self) -> None:
        """DCT output length should equal input length."""
        trace = np.random.randn(6000)
        mags = dct_coefficients(trace)
        assert len(mags) == 6000

    def test_pure_cosine(self) -> None:
        """Pure cosine should concentrate energy in ~1 coefficient."""
        n = 6000
        t = np.arange(n) / n
        trace = np.cos(2 * np.pi * 50 * t)
        mags = dct_coefficients(trace)
        ret = energy_retention(mags, [0.01])
        # 1% of 6000 = 60 coefficients — should capture nearly all energy
        assert ret[0] > 99.0


# ---------------------------------------------------------------------------
# energy_retention
# ---------------------------------------------------------------------------


class TestEnergyRetention:
    def test_full_retention(self) -> None:
        """kept_ratio=1.0 should give 100% energy."""
        mags = np.array([10.0, 5.0, 3.0, 1.0])
        ret = energy_retention(mags, [1.0])
        assert abs(ret[0] - 100.0) < 0.01

    def test_monotonic_increase(self) -> None:
        """More coefficients should give higher retention."""
        mags = np.array([10.0, 5.0, 3.0, 1.0, 0.5, 0.1])
        ratios = [0.1, 0.2, 0.5, 0.8, 1.0]
        ret = energy_retention(mags, ratios)
        for i in range(len(ret) - 1):
            assert ret[i] <= ret[i + 1]

    def test_known_values(self) -> None:
        """Verify exact energy calculation with known input."""
        # mags = [10, 5, 3, 1] → energies = [100, 25, 9, 1] = 135 total
        mags = np.array([10.0, 5.0, 3.0, 1.0])
        # kept_ratio=0.25 → 1 coeff → 100/135 = 74.07%
        # kept_ratio=0.50 → 2 coeffs → 125/135 = 92.59%
        ret = energy_retention(mags, [0.25, 0.50])
        assert abs(ret[0] - 74.074) < 0.1
        assert abs(ret[1] - 92.593) < 0.1

    def test_zero_energy_trace(self) -> None:
        """Zero-energy input should return 0% retention."""
        mags = np.zeros(100)
        ret = energy_retention(mags, [0.5])
        assert ret[0] == 0.0


# ---------------------------------------------------------------------------
# analyze_sparsity
# ---------------------------------------------------------------------------


class TestAnalyzeSparsity:
    @pytest.fixture
    def synthetic_waveforms(self) -> np.ndarray:
        """Create small synthetic waveforms for testing."""
        rng = np.random.RandomState(42)
        # 50 traces × 3 channels × 256 samples (small for speed)
        waveforms = rng.randn(50, 3, 256).astype(np.float32)
        # Add some sparse structure: sine waves with noise
        t = np.linspace(0, 1, 256)
        for i in range(50):
            freq = rng.uniform(5, 50)
            waveforms[i, 0] += 5 * np.sin(2 * np.pi * freq * t)
            waveforms[i, 1] += 3 * np.sin(2 * np.pi * freq * 1.5 * t)
            waveforms[i, 2] += 4 * np.sin(2 * np.pi * freq * 0.8 * t)
        return waveforms

    def test_result_schema(self, synthetic_waveforms: np.ndarray) -> None:
        """Result should contain all required top-level keys."""
        results = analyze_sparsity(
            synthetic_waveforms, n_traces=20, wavelet_level=3
        )
        assert "metadata" in results
        assert "decay_curves" in results
        assert "energy_retention" in results
        assert "cr_retention" in results
        assert "key_findings" in results

        # Check key_findings subkeys
        kf = results["key_findings"]
        assert "wavelet_20pct_energy" in kf
        assert "dct_20pct_energy" in kf
        assert "cs_justified" in kf
        assert "best_transform" in kf
        assert "cr8_wavelet_energy" in kf

    def test_cs_justified_flag(self) -> None:
        """Very sparse synthetic signals should be CS-justified."""
        # Create highly sparse signals: pure sine waves (no noise)
        rng = np.random.RandomState(42)
        n = 50
        waveforms = np.zeros((n, 3, 1024), dtype=np.float32)
        t = np.linspace(0, 1, 1024)
        for i in range(n):
            freq = rng.uniform(5, 30)
            for c in range(3):
                waveforms[i, c] = 10 * np.sin(2 * np.pi * freq * t)
        results = analyze_sparsity(waveforms, n_traces=20, wavelet_level=4)
        assert results["key_findings"]["cs_justified"] is True

    def test_reproducibility(self, synthetic_waveforms: np.ndarray) -> None:
        """Same seed should give identical results."""
        r1 = analyze_sparsity(synthetic_waveforms, n_traces=20, seed=42, wavelet_level=3)
        r2 = analyze_sparsity(synthetic_waveforms, n_traces=20, seed=42, wavelet_level=3)
        assert r1["key_findings"] == r2["key_findings"]
        np.testing.assert_array_almost_equal(
            r1["energy_retention"]["wavelet"]["mean"],
            r2["energy_retention"]["wavelet"]["mean"],
        )


# ---------------------------------------------------------------------------
# plot_sparsity
# ---------------------------------------------------------------------------


class TestPlotSparsity:
    def test_creates_two_figures(self, tmp_path: Path) -> None:
        """Should create fig_decay.png and fig_energy_retention.png."""
        rng = np.random.RandomState(42)
        waveforms = rng.randn(30, 3, 256).astype(np.float32)
        results = analyze_sparsity(waveforms, n_traces=10, wavelet_level=3)

        paths = plot_sparsity(results, tmp_path)
        assert len(paths) == 2
        assert (tmp_path / "fig_decay.png").exists()
        assert (tmp_path / "fig_energy_retention.png").exists()

    def test_figure_file_sizes(self, tmp_path: Path) -> None:
        """Generated figures should have non-zero file size."""
        rng = np.random.RandomState(42)
        waveforms = rng.randn(30, 3, 256).astype(np.float32)
        results = analyze_sparsity(waveforms, n_traces=10, wavelet_level=3)

        paths = plot_sparsity(results, tmp_path)
        for p in paths:
            assert p.stat().st_size > 1000  # at least 1KB

    def test_figure_dimensions(self, tmp_path: Path) -> None:
        """Figures should be IEEE 2-column size (3.5 x 2.5 inch)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.image import imread

        rng = np.random.RandomState(42)
        waveforms = rng.randn(30, 3, 256).astype(np.float32)
        results = analyze_sparsity(waveforms, n_traces=10, wavelet_level=3)
        paths = plot_sparsity(results, tmp_path)

        for p in paths:
            img = imread(str(p))
            # At 300 dpi, 3.5 x 2.5 inch -> ~1050 x 750 pixels (approximate)
            h, w = img.shape[:2]
            assert 900 < w < 1400, f"Width {w} not in expected range for 3.5in @ 300dpi"
            assert 600 < h < 1100, f"Height {h} not in expected range for 2.5in @ 300dpi"


class TestBandEnergyRetention:
    """Retained-band energy behaves physically for known signals."""

    def _band_retention(self, x, keep):
        import pywt
        coeffs = pywt.wavedec(x, "db4", level=6, mode="symmetric")
        sq = [float((c ** 2).sum()) for c in coeffs]
        return sum(sq[:keep]) / sum(sq) * 100.0

    def test_low_frequency_tone_fully_retained(self):
        t = np.arange(6000) / 100.0
        x = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz, well below the CR=4 bands
        assert self._band_retention(x, keep=5) > 99.0

    def test_high_frequency_tone_mostly_removed(self):
        t = np.arange(6000) / 100.0
        x = np.sin(2 * np.pi * 40.0 * t)  # 40 Hz, inside removed cD1
        assert self._band_retention(x, keep=5) < 10.0

    def test_retention_decreases_with_cr(self):
        rng = np.random.RandomState(0)
        x = rng.randn(6000)
        r = [self._band_retention(x, keep=k) for k in (6, 5, 4, 3)]
        assert r[0] > r[1] > r[2] > r[3]
