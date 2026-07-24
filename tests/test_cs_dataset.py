"""Unit tests for cs_pipeline.cs_dataset module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cs_pipeline.cs_dataset import CompressedSTEADDataset


@pytest.fixture
def mock_compressed_dir(tmp_path: Path) -> Path:
    """Create a mock compressed dataset directory."""
    n_traces = 50
    M = 750  # CR=8
    N = 6000

    # Waveforms
    waveforms = np.random.randn(n_traces, 3, M).astype(np.float32)
    np.save(str(tmp_path / "train_waveforms.npy"), waveforms)

    # Arrivals
    p_samples = np.random.randint(500, 4000, size=n_traces).astype(np.int32)
    s_samples = np.random.randint(1000, 5500, size=n_traces).astype(np.int32)
    np.savez(
        str(tmp_path / "train_arrivals.npz"),
        p_samples=p_samples,
        s_samples=s_samples,
    )

    # Metadata
    metadata = {
        "matrix_type": "gaussian",
        "CR": 8,
        "N": N,
        "M": M,
        "seed": 42,
        "source_dir": "/path/to/stead_memmap",
        "splits": {"train": {"n_traces": n_traces, "encoding_time_sec": 1.0}},
        "created": "2026-03-27T00:00:00+00:00",
    }
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return tmp_path


class TestCompressedSTEADDataset:
    def test_len(self, mock_compressed_dir: Path) -> None:
        """Dataset length should match number of traces."""
        ds = CompressedSTEADDataset(mock_compressed_dir, split="train")
        assert len(ds) == 50

    def test_getitem_shapes(self, mock_compressed_dir: Path) -> None:
        """waveform should be (3, M), label should be (3, 6000)."""
        ds = CompressedSTEADDataset(mock_compressed_dir, split="train")
        waveform, label = ds[0]

        assert waveform.shape == (3, 750)
        assert label.shape == (3, 6000)
        assert waveform.dtype == torch.float32
        assert label.dtype == torch.float32

    def test_waveform_normalized(self, mock_compressed_dir: Path) -> None:
        """Waveform should be std-normalized (~1.0 std)."""
        ds = CompressedSTEADDataset(mock_compressed_dir, split="train")
        waveform, _ = ds[0]
        std = waveform.std().item()
        # After normalization, std should be ~1.0
        assert 0.5 < std < 2.0

    def test_label_sums_to_one(self, mock_compressed_dir: Path) -> None:
        """Label channels should approximately sum to 1.0 at each time step."""
        ds = CompressedSTEADDataset(mock_compressed_dir, split="train")
        _, label = ds[0]
        channel_sum = label.sum(dim=0)
        assert torch.allclose(channel_sum, torch.ones(6000), atol=0.01)

    def test_metadata_loaded(self, mock_compressed_dir: Path) -> None:
        """M, CR, N should be loaded from metadata.json."""
        ds = CompressedSTEADDataset(mock_compressed_dir, split="train")
        assert ds.M == 750
        assert ds.CR == 8
        assert ds.N == 6000

    def test_missing_waveforms_raises(self, tmp_path: Path) -> None:
        """Missing waveform file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CompressedSTEADDataset(tmp_path, split="train")

    def test_dataloader_compatible(self, mock_compressed_dir: Path) -> None:
        """Should work with PyTorch DataLoader."""
        ds = CompressedSTEADDataset(mock_compressed_dir, split="train")
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (8, 3, 750)
        assert batch_y.shape == (8, 3, 6000)

    def test_no_metadata_fallback(self, tmp_path: Path) -> None:
        """Should work without metadata.json, inferring M from data."""
        n, M = 10, 500
        np.save(str(tmp_path / "train_waveforms.npy"),
                np.random.randn(n, 3, M).astype(np.float32))
        np.savez(str(tmp_path / "train_arrivals.npz"),
                 p_samples=np.full(n, 2000, dtype=np.int32),
                 s_samples=np.full(n, 3500, dtype=np.int32))

        ds = CompressedSTEADDataset(tmp_path, split="train")
        assert ds.M == 500
        assert len(ds) == 10
