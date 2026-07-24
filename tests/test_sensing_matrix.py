"""Unit tests for cs_pipeline.sensing_matrix module."""

from __future__ import annotations

import torch
import pytest

from cs_pipeline.sensing_matrix import (
    GaussianSensingMatrix,
    DCTSubsampling,
    ButterworthLowpassNoDownsample,
    create_sensing_matrix,
)


# ---------------------------------------------------------------------------
# GaussianSensingMatrix
# ---------------------------------------------------------------------------


class TestGaussianSensingMatrix:
    def test_output_shape(self) -> None:
        """encode() should produce (B, C, M) from (B, C, N)."""
        matrix = GaussianSensingMatrix(N=6000, CR=8, seed=42)
        x = torch.randn(4, 3, 6000)
        y = matrix.encode(x)
        assert y.shape == (4, 3, 750)

    def test_output_shape_cr4(self) -> None:
        """CR=4 should give M=1500."""
        matrix = GaussianSensingMatrix(N=6000, CR=4, seed=42)
        x = torch.randn(2, 3, 6000)
        y = matrix.encode(x)
        assert y.shape == (2, 3, 1500)

    def test_output_shape_cr16(self) -> None:
        """CR=16 should give M=375."""
        matrix = GaussianSensingMatrix(N=6000, CR=16, seed=42)
        x = torch.randn(2, 3, 6000)
        y = matrix.encode(x)
        assert y.shape == (2, 3, 375)

    def test_reproducibility(self) -> None:
        """Same seed should produce identical matrices and outputs."""
        m1 = GaussianSensingMatrix(N=6000, CR=8, seed=42)
        m2 = GaussianSensingMatrix(N=6000, CR=8, seed=42)
        assert torch.allclose(m1.phi, m2.phi)

        x = torch.randn(2, 3, 6000)
        assert torch.allclose(m1.encode(x), m2.encode(x))

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different matrices."""
        m1 = GaussianSensingMatrix(N=6000, CR=8, seed=42)
        m2 = GaussianSensingMatrix(N=6000, CR=8, seed=123)
        assert not torch.allclose(m1.phi, m2.phi)

    def test_row_normalization(self) -> None:
        """Rows should have approximately unit norm / sqrt(M) scaling."""
        matrix = GaussianSensingMatrix(N=6000, CR=8, seed=42)
        # Each row has N elements drawn from N(0, 1/M)
        # Expected row norm ≈ sqrt(N/M) = sqrt(CR)
        row_norms = torch.norm(matrix.phi, dim=1)
        expected_norm = (6000 / 750) ** 0.5  # sqrt(8) ≈ 2.83
        mean_norm = row_norms.mean().item()
        assert abs(mean_norm - expected_norm) < 0.5

    def test_rip_approximate(self) -> None:
        """RIP approximate: distances should be approximately preserved."""
        matrix = GaussianSensingMatrix(N=600, CR=4, seed=42)
        # Create two sparse signals
        x1 = torch.zeros(1, 1, 600)
        x2 = torch.zeros(1, 1, 600)
        x1[0, 0, 10] = 1.0
        x2[0, 0, 20] = 1.0

        y1 = matrix.encode(x1)
        y2 = matrix.encode(x2)

        dist_orig = torch.norm(x1 - x2).item()
        dist_comp = torch.norm(y1 - y2).item()

        # RIP: (1-δ)||x||² ≤ ||Φx||² ≤ (1+δ)||x||²
        # With M=150, δ should be moderate. Check ratio is reasonable.
        ratio = dist_comp / dist_orig
        assert 0.1 < ratio < 10.0  # Very loose check

    def test_invalid_cr(self) -> None:
        """CR too large should raise ValueError."""
        with pytest.raises(ValueError):
            GaussianSensingMatrix(N=100, CR=200)


# ---------------------------------------------------------------------------
# DCTSubsampling
# ---------------------------------------------------------------------------


class TestDCTSubsampling:
    def test_output_shape(self) -> None:
        """encode() should produce (B, C, M) time-domain output."""
        matrix = DCTSubsampling(N=6000, CR=8)
        x = torch.randn(4, 3, 6000)
        y = matrix.encode(x)
        assert y.shape == (4, 3, matrix.M)

    def test_output_is_time_domain(self) -> None:
        """Output should be a smoothed time-domain waveform, not frequency coefficients."""
        matrix = DCTSubsampling(N=6000, CR=4, target_M=1506)
        t = torch.linspace(0, 1, 6000)
        x = torch.sin(2 * 3.14159 * 5 * t).unsqueeze(0).unsqueeze(0)
        y = matrix.encode(x)
        # Time-domain sine should still look sinusoidal — check it oscillates
        y_np = y[0, 0].numpy()
        zero_crossings = ((y_np[:-1] * y_np[1:]) < 0).sum()
        assert zero_crossings >= 6, "Output doesn't look like time-domain signal"

    def test_decode_shape(self) -> None:
        """decode() should produce (B, C, N)."""
        matrix = DCTSubsampling(N=6000, CR=8)
        y = torch.randn(4, 3, matrix.M)
        x_hat = matrix.decode(y)
        assert x_hat.shape == (4, 3, 6000)

    def test_reconstruction_error_cr4(self) -> None:
        """encode→decode at CR=4 should retain >90% energy."""
        matrix = DCTSubsampling(N=6000, CR=4, target_M=1506)
        t = torch.linspace(0, 1, 6000)
        x = torch.sin(2 * 3.14159 * 10 * t).unsqueeze(0).unsqueeze(0)
        x = x.expand(1, 3, 6000).clone()

        y = matrix.encode(x)
        x_hat = matrix.decode(y)

        original_energy = (x ** 2).sum().item()
        error_energy = ((x - x_hat) ** 2).sum().item()
        error_ratio = error_energy / original_energy
        assert error_ratio < 0.10, f"Reconstruction error {error_ratio:.4f} > 10%"

    def test_deterministic(self) -> None:
        """DCT is deterministic — same input gives same output."""
        matrix = DCTSubsampling(N=6000, CR=8)
        x = torch.randn(2, 3, 6000)
        y1 = matrix.encode(x)
        y2 = matrix.encode(x)
        assert torch.allclose(y1, y2)

    def test_target_m_matches_wavelet(self) -> None:
        """When target_M is set, output length should match wavelet."""
        matrix = DCTSubsampling(N=6000, CR=4, target_M=1506)
        x = torch.randn(2, 3, 6000)
        y = matrix.encode(x)
        assert y.shape == (2, 3, 1506)


# ---------------------------------------------------------------------------
# ButterworthLowpassNoDownsample (LP-only: denoising vs dimensionality reduction)
# ---------------------------------------------------------------------------


class TestButterworthLowpassNoDownsample:
    def test_preserves_length(self) -> None:
        """LP-only must keep the full N samples (no downsampling)."""
        m = ButterworthLowpassNoDownsample(N=6000, cutoff_cr=4)
        x = torch.randn(2, 3, 6000)
        y = m.encode(x)
        assert y.shape == (2, 3, 6000)
        assert m.M == 6000 and m.CR == 1

    def test_cutoff_matches_cr4_band(self) -> None:
        """cutoff_cr=4 @ fs=100 → 12.5 Hz, matching the CR=4 denoising band."""
        m = ButterworthLowpassNoDownsample(N=6000, cutoff_cr=4, fs=100.0)
        assert abs(m._cutoff - 12.5) < 1e-6

    def test_suppresses_high_frequency(self) -> None:
        """A 30 Hz tone (above the 12.5 Hz cutoff) must be strongly attenuated."""
        fs, n = 100.0, 6000
        t = torch.arange(n) / fs
        high = torch.sin(2 * torch.pi * 30.0 * t).reshape(1, 1, n)
        m = ButterworthLowpassNoDownsample(N=n, cutoff_cr=4, fs=fs)
        y = m.encode(high)
        # Output energy should be a small fraction of the input energy.
        assert y.pow(2).mean().item() < 0.05 * high.pow(2).mean().item()

    def test_preserves_low_frequency(self) -> None:
        """A 2 Hz tone (below cutoff) must pass through largely intact."""
        fs, n = 100.0, 6000
        t = torch.arange(n) / fs
        low = torch.sin(2 * torch.pi * 2.0 * t).reshape(1, 1, n)
        m = ButterworthLowpassNoDownsample(N=n, cutoff_cr=4, fs=fs)
        y = m.encode(low)
        assert y.pow(2).mean().item() > 0.8 * low.pow(2).mean().item()

    def test_decode_is_identity_length(self) -> None:
        m = ButterworthLowpassNoDownsample(N=6000)
        y = torch.randn(1, 3, 6000)
        assert m.decode(y).shape == (1, 3, 6000)


class TestCreateSensingMatrix:
    def test_gaussian(self) -> None:
        m = create_sensing_matrix("gaussian", N=6000, CR=8)
        assert isinstance(m, GaussianSensingMatrix)

    def test_lowpass(self) -> None:
        m = create_sensing_matrix("lowpass", N=6000, CR=1)
        assert isinstance(m, ButterworthLowpassNoDownsample)
        assert m.M == 6000

    def test_dct(self) -> None:
        m = create_sensing_matrix("dct", N=6000, CR=8)
        assert isinstance(m, DCTSubsampling)

    def test_butterworth(self) -> None:
        m = create_sensing_matrix("butterworth", N=6000, CR=4)
        assert m.M == 1506

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            create_sensing_matrix("curvelet", N=6000, CR=8)

    def test_all_have_encode(self) -> None:
        """All matrix types should have .encode() method with correct M."""
        x = torch.randn(2, 3, 6000)
        for mt in ["wavelet", "gaussian", "dct", "butterworth", "lowpass"]:
            m = create_sensing_matrix(mt, N=6000, CR=4)
            y = m.encode(x)
            assert y.shape == (2, 3, m.M), f"{mt} encode shape mismatch: {y.shape} vs M={m.M}"
