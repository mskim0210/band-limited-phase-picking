"""Event-level (source_id-grouped) re-split of the fixed STEAD memmap files.

The memmap files were cut from a seed-42 random permutation of the STEAD CSV
(trace-level: ``preprocess_stead_memmap.py``).  This module builds an
alternative 85/5/10 split in which all traces of one earthquake (same
``source_id``) fall on the same side, and exposes it as index arrays
``[file_id, row]`` into the existing memmap / CS-encoded files, so no data
are copied.  Noise traces (``source_id`` NaN) are singleton groups.  The
CS-encoded directories preserve the memmap row order per split (verified:
``test_arrivals.npz`` identical), so the same index applies to both.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

FILE_IDS = {"train": 0, "val": 1, "test": 2}


def _offsets(n_total: int, split_ratio: tuple[float, float, float]) -> tuple[int, int]:
    """Return (n_train, n_val) exactly as the memmap preprocessing computed them."""
    n_train = int(split_ratio[0] * n_total)
    n_val = int(split_ratio[1] * n_total)
    return n_train, n_val


def memmap_row_to_csv_row(file_id: int, row: int, n_total: int, seed: int,
                          split_ratio: tuple[float, float, float]) -> int:
    """Map (memmap file, row) back to the original CSV row index.

    Args:
        file_id: 0=train, 1=val, 2=test memmap file.
        row: Row within that file.
        n_total: Number of CSV rows.
        seed: Permutation seed used by the memmap preprocessing.
        split_ratio: (train, val, test) fractions used by the preprocessing.
    """
    perm = np.random.RandomState(seed).permutation(n_total)
    n_train, n_val = _offsets(n_total, split_ratio)
    offset = (0, n_train, n_train + n_val)[file_id]
    return int(perm[offset + row])


def build_event_split(csv_path: str | Path, seed: int,
                      split_ratio: tuple[float, float, float]) -> dict[str, np.ndarray]:
    """Group-shuffle split by ``source_id``.

    Args:
        csv_path: STEAD merge.csv (needs the ``source_id`` column).
        seed: Memmap permutation seed; the group shuffle uses ``seed + 1``.
        split_ratio: Target (train, val, test) trace fractions.

    Returns:
        ``{"train"|"val"|"test": (K, 2) int64 array of [file_id, row]}``.
    """
    meta = pd.read_csv(csv_path, usecols=["source_id"], low_memory=False)
    n_total = len(meta)
    perm = np.random.RandomState(seed).permutation(n_total)
    n_train, n_val = _offsets(n_total, split_ratio)

    # csv_row -> (file_id, row) of the existing memmap layout
    file_of = np.empty(n_total, dtype=np.int8)
    row_of = np.empty(n_total, dtype=np.int64)
    for lo, hi, fid in ((0, n_train, 0), (n_train, n_train + n_val, 1),
                        (n_train + n_val, n_total, 2)):
        file_of[perm[lo:hi]] = fid
        row_of[perm[lo:hi]] = np.arange(hi - lo)

    sid = meta["source_id"].astype("string")
    noise_ids = pd.Series([f"_noise_{i}" for i in range(n_total)], dtype="string")
    groups = sid.fillna(noise_ids)
    codes, uniques = pd.factorize(groups)
    sizes = np.bincount(codes, minlength=len(uniques))

    order = np.random.RandomState(seed + 1).permutation(len(uniques))
    cum = np.cumsum(sizes[order]) / n_total
    side_in_order = np.where(cum <= split_ratio[0], 0,
                             np.where(cum <= split_ratio[0] + split_ratio[1], 1, 2))
    side = np.empty(len(uniques), dtype=np.int8)
    side[order] = side_in_order
    trace_side = side[codes]

    out: dict[str, np.ndarray] = {}
    for name, fid in FILE_IDS.items():
        rows = np.where(trace_side == fid)[0]
        out[name] = np.stack([file_of[rows], row_of[rows]], axis=1).astype(np.int64)
        logger.info("event split %s: %d traces (%.2f%%)",
                    name, len(rows), 100.0 * len(rows) / n_total)
    return out


def load_or_build_event_split(csv_path: str | Path, cache: str | Path, seed: int,
                              split_ratio: tuple[float, float, float]) -> dict[str, np.ndarray]:
    """Load cached split arrays (npz) or build and cache them."""
    cache = Path(cache)
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in FILE_IDS}
    splits = build_event_split(csv_path, seed, split_ratio)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **splits)
    logger.info("event split cached at %s", cache)
    return splits


class IndexedSplitDataset(Dataset):
    """Delegates to the three per-file datasets through a [file_id, row] index.

    Args:
        base_datasets: ``{0: train_ds, 1: val_ds, 2: test_ds}`` built on the
            original memmap (or CS-encoded) files.
        index: (K, 2) array of [file_id, row].
    """

    def __init__(self, base_datasets: dict[int, Dataset], index: np.ndarray) -> None:
        self.base = base_datasets
        self.index = np.asarray(index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        fid, row = self.index[i]
        return self.base[int(fid)][int(row)]
