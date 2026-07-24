"""Waveform encoders for compressed-domain training.

Encoder types:
    1. WaveletSensingMatrix — band-level truncation (db4; primary method)
    2. GaussianSensingMatrix — random Gaussian projection (structure-destroying reference)
    3. DCTSubsampling — low-frequency DCT truncation, reconstructed to the time domain
    4. ButterworthLowpass — zero-phase low-pass + resampling
    5. ButterworthLowpassNoDownsample — LP-only control (length-preserving)

All classes share a common ``.encode(x)`` interface:
    input:  (B, C, N) where N=6000 (samples per channel)
    output: (B, C, M) where M=N/CR
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pywt
import torch
import torch.nn as nn
from scipy.fft import dct as scipy_dct
from scipy.fft import idct as scipy_idct

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wavelet Sensing Matrix (PRIMARY — best energy retention per sparsity analysis)
# ---------------------------------------------------------------------------


class WaveletSensingMatrix:
    """Wavelet band-level truncation for time-structure-preserving compression.

    Applies pywt.wavedec and reconstructs with fewer detail levels,
    producing a smoothed waveform that preserves temporal structure.

    Band-level truncation (db4, level=6, N=6000):
        CR≈2:  keep 5 levels (remove cD1)        → M=3004
        CR≈4:  keep 4 levels (remove cD1,cD2)    → M=1506
        CR≈8:  keep 3 levels (remove cD1-cD3)    → M=756
        CR≈16: keep 2 levels (remove cD1-cD4)    → M=382

    NOTE: Previous top-M magnitude selection destroyed time structure
    and caused F1=0.0 training failure. This band-level approach
    preserves temporal ordering for Conv1d learning.

    Args:
        N: Input dimension (samples per channel). Default: 6000.
        CR: Compression ratio (2, 4, 8, or 16).
        wavelet: Wavelet name for pywt. Default: 'db4'.
        level: Decomposition level. Default: 6.

    Example::

        matrix = WaveletSensingMatrix(N=6000, CR=4)
        y = matrix.encode(x)  # (B, 3, 6000) → (B, 3, 1506)
    """

    # CR → number of wavelet levels to keep in partial reconstruction
    _CR_TO_KEEP_LEVELS = {2: 5, 4: 4, 8: 3, 16: 2}

    def __init__(
        self,
        N: int = 6000,
        CR: int = 8,
        wavelet: str = "db4",
        level: int = 6,
        **kwargs: object,
    ) -> None:
        self.N = N
        self.CR = CR
        self.wavelet = wavelet
        self.level = level

        if CR not in self._CR_TO_KEEP_LEVELS:
            raise ValueError(
                f"CR={CR} not supported for wavelet. Use one of {list(self._CR_TO_KEEP_LEVELS.keys())}"
            )
        self._keep_levels = self._CR_TO_KEEP_LEVELS[CR]

        # Pre-compute output length M via dummy signal
        dummy = np.zeros(N)
        coeffs = pywt.wavedec(dummy, wavelet, level=level, mode="symmetric")
        partial = coeffs[: self._keep_levels + 1]
        rec = pywt.waverec(partial, wavelet, mode="symmetric")
        self.M = len(rec)

        logger.info(
            "WaveletSensingMatrix: N=%d → M=%d (CR=%.2f), keep=%d/%d levels, "
            "wavelet=%s",
            self.N, self.M, self.N / self.M, self._keep_levels, self.level,
            self.wavelet,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Wavelet band-level truncation: remove high-freq detail, reconstruct.

        Args:
            x: (B, C, N) input tensor.

        Returns:
            (B, C, M) time-structure-preserving compressed waveform.
        """
        x_np = x.detach().cpu().numpy()
        B, C, _ = x_np.shape
        result = np.zeros((B, C, self.M), dtype=np.float32)

        for b in range(B):
            for c in range(C):
                coeffs = pywt.wavedec(
                    x_np[b, c], self.wavelet, level=self.level, mode="symmetric"
                )
                # Keep only first (keep_levels + 1) coefficient arrays
                # coeffs[0]=cA6, coeffs[1]=cD6, ..., coeffs[keep_levels]=cD(7-keep_levels)
                partial = coeffs[: self._keep_levels + 1]
                rec = pywt.waverec(partial, self.wavelet, mode="symmetric")
                result[b, c] = rec[: self.M].astype(np.float32)

        return torch.from_numpy(result).to(x.device, dtype=x.dtype)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Upsample compressed waveform back to original length via interpolation.

        This is approximate — the removed high-frequency detail cannot be recovered.

        Args:
            y: (B, C, M) compressed waveform.

        Returns:
            (B, C, N) upsampled waveform.
        """
        import torch.nn.functional as F

        # Simple linear interpolation to original length
        return F.interpolate(
            y.float(), size=self.N, mode="linear", align_corners=False
        ).to(y.dtype)


# ---------------------------------------------------------------------------
# Gaussian Sensing Matrix
# ---------------------------------------------------------------------------


class GaussianSensingMatrix:
    """Random Gaussian sensing matrix Φ ∈ R^(M×N).

    Rows are normalized by 1/√M to satisfy RIP conditions in expectation.
    The matrix is generated once at init and reused for all encode() calls.

    Args:
        N: Input dimension (samples per channel). Default: 6000.
        CR: Compression ratio. M = N // CR.
        seed: Random seed for reproducibility. Default: 42.
        device: Torch device. Default: 'cpu'.

    Example::

        matrix = GaussianSensingMatrix(N=6000, CR=8, seed=42)
        y = matrix.encode(x)  # (B, 3, 6000) → (B, 3, 750)
    """

    def __init__(
        self,
        N: int = 6000,
        CR: int = 8,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ) -> None:
        self.N = N
        self.M = N // CR
        self.CR = CR
        self.seed = seed

        if self.M < 1:
            raise ValueError(f"CR={CR} too large for N={N}: M={self.M}")

        gen = torch.Generator(device="cpu").manual_seed(seed)
        phi = torch.randn(self.M, self.N, generator=gen) / math.sqrt(self.M)
        self.phi = phi.to(device)  # (M, N)

        logger.info(
            "GaussianSensingMatrix: N=%d, M=%d, CR=%d, seed=%d",
            self.N, self.M, self.CR, self.seed,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress waveform via matrix multiplication.

        Args:
            x: (B, C, N) input tensor.

        Returns:
            (B, C, M) compressed tensor.
        """
        # x: (B, C, N), phi: (M, N) → x @ phi.T → (B, C, M)
        phi = self.phi.to(x.device)
        return torch.matmul(x, phi.T)

    def to(self, device: str | torch.device) -> "GaussianSensingMatrix":
        """Move matrix to device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.phi = self.phi.to(device)
        return self


# ---------------------------------------------------------------------------
# DCT Subsampling
# ---------------------------------------------------------------------------


class DCTSubsampling:
    """DCT low-pass filtering with time-domain output.

    Applies DCT, keeps first K low-frequency coefficients, reconstructs
    via IDCT to time domain, then uniformly downsamples to M samples.

    This produces a time-domain smoothed waveform comparable to wavelet
    band-level truncation — both remove high-frequency detail and output
    a shorter time-domain signal that Conv1d can learn from.

    When target_M is set (e.g. to match wavelet's M=1506), output length
    is forced to that value so that all encoders share the same output length.

    Args:
        N: Input dimension. Default: 6000.
        CR: Compression ratio. Controls DCT truncation point.
        target_M: Force output length (to match the wavelet output length).
            If None, uses N // CR.

    Example::

        matrix = DCTSubsampling(N=6000, CR=4, target_M=1506)
        y = matrix.encode(x)  # (B, 3, 6000) → (B, 3, 1506) time-domain
    """

    def __init__(
        self,
        N: int = 6000,
        CR: int = 8,
        target_M: int | None = None,
        **kwargs: object,
    ) -> None:
        self.N = N
        self.CR = CR
        self._dct_keep = N // CR  # number of DCT coefficients to keep

        if self._dct_keep < 1:
            raise ValueError(f"CR={CR} too large for N={N}")

        # Output length: match wavelet if target_M given, else N // CR
        self.M = target_M if target_M is not None else self._dct_keep

        logger.info(
            "DCTSubsampling: N=%d, keep=%d DCT coeffs, output M=%d (CR≈%.1f)",
            self.N, self._dct_keep, self.M, self.N / self.M,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """DCT low-pass → IDCT reconstruct → downsample to M.

        Args:
            x: (B, C, N) input tensor.

        Returns:
            (B, C, M) time-domain smoothed waveform.
        """
        x_np = x.detach().cpu().numpy()
        B, C, N = x_np.shape

        # DCT → truncate → zero-pad → IDCT (back to time domain, length N)
        coeffs = scipy_dct(x_np, type=2, norm="ortho", axis=-1)
        coeffs[..., self._dct_keep:] = 0.0
        reconstructed = scipy_idct(coeffs, type=2, norm="ortho", axis=-1)  # (B, C, N)

        # Uniform downsample to M
        indices = np.linspace(0, N - 1, self.M, dtype=int)
        result = reconstructed[..., indices].astype(np.float32)

        return torch.from_numpy(result).to(x.device, dtype=x.dtype)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Upsample compressed time-domain waveform back to N via interpolation.

        Args:
            y: (B, C, M) compressed waveform.

        Returns:
            (B, C, N) upsampled waveform.
        """
        import torch.nn.functional as F

        return F.interpolate(
            y.float(), size=self.N, mode="linear", align_corners=False
        ).to(y.dtype)


# ---------------------------------------------------------------------------
# Butterworth Low-pass Baseline
# ---------------------------------------------------------------------------


class ButterworthLowpass:
    """Butterworth low-pass filter + uniform downsampling baseline.

    Applies a zero-phase Butterworth low-pass filter and then resamples
    to M samples. This serves as a control experiment to test whether
    the P-F1 improvement from wavelet band-truncation is due to the
    wavelet basis specifically, or simply from low-pass smoothing.

    At CR=4 with fs=100Hz:
        Nyquist of target rate = fs / (2 * CR) = 12.5 Hz
        Butterworth order=4, cutoff=12.5 Hz, zero-phase (filtfilt)

    Args:
        N: Input dimension. Default: 6000.
        CR: Compression ratio.
        order: Butterworth filter order. Default: 4.
        fs: Sampling rate in Hz. Default: 100.
        target_M: Force output length (to match wavelet M). If None, uses N//CR.

    Example::

        matrix = ButterworthLowpass(N=6000, CR=4)
        y = matrix.encode(x)  # (B, 3, 6000) → (B, 3, 1506)
    """

    def __init__(
        self,
        N: int = 6000,
        CR: int = 8,
        order: int = 4,
        fs: float = 100.0,
        target_M: int | None = None,
        **kwargs: object,
    ) -> None:
        from scipy.signal import butter

        self.N = N
        self.CR = CR
        self.fs = fs
        self.order = order

        # Output length matches wavelet if target_M given
        self.M = target_M if target_M is not None else N // CR

        # Cutoff = Nyquist of target sample rate
        self._cutoff = fs / (2.0 * CR)
        nyquist = fs / 2.0
        normalized_cutoff = self._cutoff / nyquist

        # Clamp to valid range (0, 1) exclusive
        normalized_cutoff = min(max(normalized_cutoff, 0.001), 0.999)

        self._b, self._a = butter(order, normalized_cutoff, btype="low")

        logger.info(
            "ButterworthLowpass: N=%d → M=%d (CR≈%.1f), cutoff=%.1fHz, order=%d",
            self.N, self.M, self.N / self.M, self._cutoff, self.order,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Low-pass filter + resample to M samples.

        Args:
            x: (B, C, N) input tensor.

        Returns:
            (B, C, M) filtered and downsampled time-domain waveform.
        """
        from scipy.signal import filtfilt, resample

        x_np = x.detach().cpu().numpy()
        B, C, N = x_np.shape
        result = np.zeros((B, C, self.M), dtype=np.float32)

        for b in range(B):
            for c in range(C):
                # Zero-phase Butterworth low-pass
                filtered = filtfilt(self._b, self._a, x_np[b, c]).astype(np.float32)
                # Resample to target M
                result[b, c] = resample(filtered, self.M)

        return torch.from_numpy(result).to(x.device, dtype=x.dtype)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Upsample back to N via interpolation."""
        import torch.nn.functional as F

        return F.interpolate(
            y.float(), size=self.N, mode="linear", align_corners=False
        ).to(y.dtype)


# ---------------------------------------------------------------------------
# Butterworth low-pass WITHOUT downsampling (LP-only: denoising vs dim-reduction)
# ---------------------------------------------------------------------------


class ButterworthLowpassNoDownsample:
    """Zero-phase Butterworth low-pass that keeps the full length (no resample).

    Applies the *same* low-pass denoising as the CR=4 Butterworth baseline
    (4th-order, 12.5 Hz cutoff, zero-phase ``filtfilt``) but preserves all
    ``N`` samples. This isolates the implicit-denoising effect from the
    dimensionality-reduction effect:

        Baseline      : no denoise, no dim-reduction, M=6000
        LP-only (this): denoise,    no dim-reduction, M=6000
        Butterworth   : denoise,    dim-reduction,    M=1500
        Wavelet CR=4  : denoise,    dim-reduction,    M=1506

    Comparing LP-only against the baseline and against the downsampled
    Butterworth/Wavelet pins down which effect drives the P-F1 gain.

    Args:
        N: Input/output dimension (samples per channel). Default: 6000.
        cutoff_cr: Effective CR used only to set the cutoff frequency
            (cutoff = fs / (2 * cutoff_cr)). Default: 4 → 12.5 Hz @ fs=100,
            matching the CR=4 Butterworth/Wavelet denoising band.
        order: Butterworth filter order. Default: 4.
        fs: Sampling rate in Hz. Default: 100.

    Example::

        matrix = ButterworthLowpassNoDownsample(N=6000, cutoff_cr=4)
        y = matrix.encode(x)  # (B, 3, 6000) → (B, 3, 6000)
    """

    def __init__(
        self,
        N: int = 6000,
        cutoff_cr: int = 4,
        order: int = 4,
        fs: float = 100.0,
        **kwargs: object,
    ) -> None:
        from scipy.signal import butter

        self.N = N
        self.M = N  # no dimensionality reduction
        self.CR = 1  # output length unchanged
        self.fs = fs
        self.order = order
        self.cutoff_cr = cutoff_cr

        self._cutoff = fs / (2.0 * cutoff_cr)
        normalized_cutoff = min(max(self._cutoff / (fs / 2.0), 0.001), 0.999)
        self._b, self._a = butter(order, normalized_cutoff, btype="low")

        logger.info(
            "ButterworthLowpassNoDownsample: N=%d → M=%d (no downsample), "
            "cutoff=%.1fHz, order=%d",
            self.N, self.M, self._cutoff, self.order,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-phase low-pass filter, keeping all N samples.

        Args:
            x: (B, C, N) input tensor.

        Returns:
            (B, C, N) low-pass-filtered time-domain waveform.
        """
        from scipy.signal import filtfilt

        x_np = x.detach().cpu().numpy()
        B, C, _ = x_np.shape
        result = np.zeros((B, C, self.M), dtype=np.float32)

        for b in range(B):
            for c in range(C):
                result[b, c] = filtfilt(self._b, self._a, x_np[b, c]).astype(np.float32)

        return torch.from_numpy(result).to(x.device, dtype=x.dtype)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Identity: output is already at original length N."""
        return y


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_sensing_matrix(
    matrix_type: str,
    N: int = 6000,
    CR: int = 8,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> object:
    """Factory function to create a sensing matrix by type name.

    Args:
        matrix_type: One of 'wavelet', 'gaussian', 'dct', 'butterworth',
            'lowpass'.
        N: Input dimension.
        CR: Compression ratio.
        seed: Random seed.
        device: Torch device (for Gaussian).

    Returns:
        Sensing matrix instance with .encode() method.

    Raises:
        ValueError: If matrix_type is unknown.
    """
    if matrix_type == "wavelet":
        return WaveletSensingMatrix(N=N, CR=CR)
    elif matrix_type == "gaussian":
        return GaussianSensingMatrix(N=N, CR=CR, seed=seed, device=device)
    elif matrix_type == "dct":
        # Match the wavelet output length so all encoders are comparable
        wav = WaveletSensingMatrix(N=N, CR=CR)
        return DCTSubsampling(N=N, CR=CR, target_M=wav.M)
    elif matrix_type == "butterworth":
        wav = WaveletSensingMatrix(N=N, CR=CR)
        return ButterworthLowpass(N=N, CR=CR, target_M=wav.M)
    elif matrix_type == "lowpass":
        # Denoising-only control. Cutoff fixed to the CR=4 band
        # (12.5 Hz); CR arg is ignored since output length stays N.
        return ButterworthLowpassNoDownsample(N=N, cutoff_cr=4)
    else:
        raise ValueError(
            f"Unknown matrix_type: {matrix_type!r}. Choose from: "
            "'wavelet', 'gaussian', 'dct', 'butterworth', 'lowpass'."
        )
