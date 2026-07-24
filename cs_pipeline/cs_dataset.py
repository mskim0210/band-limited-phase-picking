"""Compressed STEAD dataset for CS pre-encoded memmap data.

Loads (3, M) compressed waveforms from pre-encoded memmap files
alongside original-length (3, 6000) Gaussian labels.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CompressedSTEADDataset(Dataset):
    """Memory-mapped dataset for CS pre-encoded STEAD waveforms.

    Loads compressed (3, M) waveforms from pre-encoded memmap files.
    Labels remain at original length (3, 6000) with Gaussian encoding.

    Args:
        compressed_dir: Directory with {split}_waveforms.npy,
            {split}_arrivals.npz, and metadata.json.
        split: One of "train", "val", "test".
        sigma: Gaussian label width in samples. Default: 10.0.

    Example::

        dataset = CompressedSTEADDataset("data/cs_encoded/gaussian_cr8", "train")
        waveform, label = dataset[0]
        # waveform.shape: (3, 750)  — compressed
        # label.shape:    (3, 6000) — original length
    """

    def __init__(
        self,
        compressed_dir: str | Path,
        split: str = "train",
        sigma: float = 10.0,
    ) -> None:
        compressed_dir = Path(compressed_dir)
        self._sigma = sigma

        # Read metadata
        meta_path = compressed_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)
            self.M = self._metadata["M"]
            self.CR = self._metadata["CR"]
            self.N = self._metadata["N"]
        else:
            self._metadata = {}
            self.M = None
            self.CR = None
            self.N = 6000

        # Load compressed waveforms (memmap)
        waveform_path = compressed_dir / f"{split}_waveforms.npy"
        if not waveform_path.exists():
            raise FileNotFoundError(
                f"Compressed waveforms not found: {waveform_path}. "
                "Run: python scripts/preprocess_cs.py --config configs/stead_experiment.yaml"
            )

        self.waveforms = np.load(str(waveform_path), mmap_mode="r")

        if self.M is None:
            self.M = self.waveforms.shape[2] if self.waveforms.ndim == 3 else None

        # Load arrivals (small, fully in RAM)
        arrivals_path = compressed_dir / f"{split}_arrivals.npz"
        if not arrivals_path.exists():
            raise FileNotFoundError(
                f"Arrivals not found: {arrivals_path}. "
                "Ensure preprocess_cs.py copied the arrival files."
            )

        arrivals = np.load(str(arrivals_path))
        self.p_samples = arrivals["p_samples"]
        self.s_samples = arrivals["s_samples"]

        logger.info(
            "CompressedSTEADDataset(%s): %d traces, M=%s, CR=%s (memmap)",
            split,
            len(self.p_samples),
            self.M,
            self.CR,
        )

    def __len__(self) -> int:
        return len(self.p_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a compressed trace and its original-length label.

        Returns:
            waveform: (3, M) float32 tensor, std-normalized in compressed domain.
            label: (3, 6000) float32 tensor — [P(Noise), P(P), P(S)].
        """
        # Compressed waveform
        waveform = self.waveforms[idx].copy().astype(np.float32)  # (3, M)

        # Std normalization in compressed domain
        std = waveform.std()
        if std > 1e-8:
            waveform /= std

        # Gaussian label at original length (6000)
        p = self.p_samples[idx]
        s = self.s_samples[idx]
        label = self._make_gaussian_label(
            int(p) if p >= 0 else None,
            int(s) if s >= 0 else None,
        )

        return torch.from_numpy(waveform), torch.from_numpy(label)

    def _make_gaussian_label(
        self,
        p_sample: int | None,
        s_sample: int | None,
        n_samples: int = 6000,
    ) -> np.ndarray:
        """Generate (3, 6000) Gaussian label array."""
        label = np.zeros((3, n_samples), dtype=np.float32)
        x = np.arange(n_samples, dtype=np.float32)

        if p_sample is not None and 0 <= p_sample < n_samples:
            label[1] = np.exp(-((x - p_sample) ** 2) / (2 * self._sigma**2))
        if s_sample is not None and 0 <= s_sample < n_samples:
            label[2] = np.exp(-((x - s_sample) ** 2) / (2 * self._sigma**2))

        label[0] = np.clip(1.0 - label[1] - label[2], 0.0, 1.0)
        return label
