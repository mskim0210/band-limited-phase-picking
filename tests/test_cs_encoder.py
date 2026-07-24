"""Unit tests for cs_pipeline.cs_encoder module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cs_pipeline.cs_encoder import encode_dataset, encode_trace
from cs_pipeline.sensing_matrix import (
    DCTSubsampling,
    GaussianSensingMatrix,
    create_sensing_matrix,
)


# ---------------------------------------------------------------------------
# encode_trace
# ---------------------------------------------------------------------------


class TestEncodeTrace:
    def test_shape_gaussian_cr8(self) -> None:
        """Gaussian CR=8: (3, 6000) → (3, 750)."""
        matrix = GaussianSensingMatrix(N=6000, CR=8, seed=42)
        waveform = np.random.randn(3, 6000).astype(np.float32)
        result = encode_trace(waveform, matrix)
        assert result.shape == (3, 750)
        assert result.dtype == np.float32

    def test_shape_dct_cr4(self) -> None:
        """DCT CR=4: (3, 6000) → (3, 1500)."""
        matrix = DCTSubsampling(N=6000, CR=4)
        waveform = np.random.randn(3, 6000).astype(np.float32)
        result = encode_trace(waveform, matrix)
        assert result.shape == (3, 1500)

    def test_shape_cr16(self) -> None:
        """CR=16: (3, 6000) → (3, 375)."""
        matrix = GaussianSensingMatrix(N=6000, CR=16, seed=42)
        waveform = np.random.randn(3, 6000).astype(np.float32)
        result = encode_trace(waveform, matrix)
        assert result.shape == (3, 375)

    def test_gaussian_reproducible(self) -> None:
        """Same seed and input → same output."""
        m1 = GaussianSensingMatrix(N=6000, CR=8, seed=42)
        m2 = GaussianSensingMatrix(N=6000, CR=8, seed=42)
        waveform = np.random.randn(3, 6000).astype(np.float32)

        r1 = encode_trace(waveform, m1)
        r2 = encode_trace(waveform, m2)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_dct_output_is_time_domain(self) -> None:
        """DCT encode should produce time-domain output (not freq coefficients)."""
        matrix = DCTSubsampling(N=600, CR=4, target_M=156)
        t = np.linspace(0, 1, 600)
        waveform = np.stack([np.sin(2 * np.pi * 5 * t)] * 3).astype(np.float32)
        result = encode_trace(waveform, matrix)
        assert result.shape == (3, 156)
        # Should still look sinusoidal
        zero_crossings = ((result[0, :-1] * result[0, 1:]) < 0).sum()
        assert zero_crossings >= 6


# ---------------------------------------------------------------------------
# encode_dataset
# ---------------------------------------------------------------------------


class TestEncodeDataset:
    @pytest.fixture
    def mock_memmap_dir(self, tmp_path: Path) -> Path:
        """Create a minimal mock memmap directory."""
        n_train, n_val = 100, 20
        N = 600  # Small N for test speed

        # Waveforms
        for split, n in [("train", n_train), ("val", n_val)]:
            waveforms = np.random.randn(n, 3, N).astype(np.float32)
            np.save(str(tmp_path / f"{split}_waveforms.npy"), waveforms)

            # Arrivals
            p_samples = np.random.randint(100, 500, size=n).astype(np.int32)
            s_samples = np.random.randint(200, 550, size=n).astype(np.int32)
            np.savez(
                str(tmp_path / f"{split}_arrivals.npz"),
                p_samples=p_samples,
                s_samples=s_samples,
            )

        return tmp_path

    def test_creates_files(self, mock_memmap_dir: Path, tmp_path: Path) -> None:
        """Should create waveform memmaps, arrivals, and metadata."""
        output_dir = tmp_path / "output"
        encode_dataset(
            memmap_dir=mock_memmap_dir,
            output_dir=output_dir,
            matrix_type="gaussian",
            CR=4,
            seed=42,
            num_workers=1,
            splits=["train", "val"],
        )

        assert (output_dir / "train_waveforms.npy").exists()
        assert (output_dir / "val_waveforms.npy").exists()
        assert (output_dir / "train_arrivals.npz").exists()
        assert (output_dir / "val_arrivals.npz").exists()
        assert (output_dir / "metadata.json").exists()

    def test_output_shape(self, mock_memmap_dir: Path, tmp_path: Path) -> None:
        """Output memmap should have correct shape."""
        output_dir = tmp_path / "output"
        encode_dataset(
            memmap_dir=mock_memmap_dir,
            output_dir=output_dir,
            matrix_type="gaussian",
            CR=4,
            seed=42,
            num_workers=1,
            splits=["train"],
        )

        result = np.load(str(output_dir / "train_waveforms.npy"), mmap_mode="r")
        assert result.shape == (100, 3, 150)  # N=600, CR=4, Gaussian M=N//CR=150
        assert result.dtype == np.float32

    def test_labels_copied(self, mock_memmap_dir: Path, tmp_path: Path) -> None:
        """Arrival labels should be identical to source."""
        output_dir = tmp_path / "output"
        encode_dataset(
            memmap_dir=mock_memmap_dir,
            output_dir=output_dir,
            matrix_type="gaussian",
            CR=4,
            seed=42,
            num_workers=1,
            splits=["train"],
        )

        orig = np.load(str(mock_memmap_dir / "train_arrivals.npz"))
        copied = np.load(str(output_dir / "train_arrivals.npz"))
        np.testing.assert_array_equal(orig["p_samples"], copied["p_samples"])
        np.testing.assert_array_equal(orig["s_samples"], copied["s_samples"])

    def test_metadata_json_schema(
        self, mock_memmap_dir: Path, tmp_path: Path
    ) -> None:
        """metadata.json should contain all required keys."""
        output_dir = tmp_path / "output"
        encode_dataset(
            memmap_dir=mock_memmap_dir,
            output_dir=output_dir,
            matrix_type="gaussian",
            CR=4,
            seed=42,
            num_workers=1,
            splits=["train"],
        )

        with open(output_dir / "metadata.json") as f:
            meta = json.load(f)

        assert meta["matrix_type"] == "gaussian"
        assert meta["CR"] == 4
        assert meta["N"] == 600  # mock data uses N=600
        assert meta["M"] == 150  # 600 // 4
        assert meta["seed"] == 42
        assert "train" in meta["splits"]
        assert "n_traces" in meta["splits"]["train"]
        assert "encoding_time_sec" in meta["splits"]["train"]
        assert "created" in meta

    def test_dct_encoding(self, mock_memmap_dir: Path, tmp_path: Path) -> None:
        """DCT encoding should produce valid time-domain compressed data."""
        output_dir = tmp_path / "output"
        encode_dataset(
            memmap_dir=mock_memmap_dir,
            output_dir=output_dir,
            matrix_type="dct",
            CR=4,
            seed=42,
            num_workers=1,
            splits=["val"],
        )

        result = np.load(str(output_dir / "val_waveforms.npy"), mmap_mode="r")
        # DCT with target_M matching wavelet — M depends on N=600
        assert result.shape[0] == 20
        assert result.shape[1] == 3
        assert result.shape[2] > 0  # M varies by wavelet structure
        assert np.abs(result).sum() > 0
