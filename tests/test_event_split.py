"""Unit tests for cs_pipeline.event_split (source_id-grouped re-split)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cs_pipeline.event_split import (
    IndexedSplitDataset,
    build_event_split,
    memmap_row_to_csv_row,
)

RATIO = (0.85, 0.05, 0.10)


def _toy_csv(tmp_path: Path, n_events: int = 200, per_event: int = 3,
             n_noise: int = 100) -> Path:
    sid = np.repeat(np.arange(n_events).astype(float), per_event).tolist() + [np.nan] * n_noise
    cat = ["earthquake_local"] * (n_events * per_event) + ["noise"] * n_noise
    df = pd.DataFrame({"source_id": sid, "trace_category": cat})
    p = tmp_path / "meta.csv"
    df.to_csv(p, index=False)
    return p


def test_row_mapping_inverts_permutation() -> None:
    n = 100
    perm = np.random.RandomState(42).permutation(n)
    n_tr, n_va = int(0.85 * n), int(0.05 * n)
    assert memmap_row_to_csv_row(0, 3, n, 42, RATIO) == perm[3]
    assert memmap_row_to_csv_row(1, 2, n, 42, RATIO) == perm[n_tr + 2]
    assert memmap_row_to_csv_row(2, 0, n, 42, RATIO) == perm[n_tr + n_va]


def test_no_event_overlap_and_ratio(tmp_path: Path) -> None:
    csv = _toy_csv(tmp_path)
    splits = build_event_split(csv, seed=42, split_ratio=RATIO)
    meta = pd.read_csv(csv)
    n = len(meta)
    seen: dict[float, str] = {}
    for name, arr in splits.items():
        assert arr.ndim == 2 and arr.shape[1] == 2
        rows = [memmap_row_to_csv_row(int(f), int(r), n, 42, RATIO) for f, r in arr]
        for sid in meta.iloc[rows]["source_id"].dropna():
            assert seen.setdefault(sid, name) == name, f"event {sid} leaks into {name}"
    total = sum(len(a) for a in splits.values())
    assert total == n
    assert abs(len(splits["train"]) / n - 0.85) < 0.03
    assert abs(len(splits["test"]) / n - 0.10) < 0.03


def test_every_memmap_row_used_exactly_once(tmp_path: Path) -> None:
    csv = _toy_csv(tmp_path)
    splits = build_event_split(csv, seed=42, split_ratio=RATIO)
    n = len(pd.read_csv(csv))
    n_tr, n_va = int(0.85 * n), int(0.05 * n)
    sizes = {0: n_tr, 1: n_va, 2: n - n_tr - n_va}
    allrows = np.concatenate(list(splits.values()))
    for fid, size in sizes.items():
        rows = np.sort(allrows[allrows[:, 0] == fid, 1])
        assert np.array_equal(rows, np.arange(size)), f"file {fid} rows not a permutation"


class _Const(torch.utils.data.Dataset):
    def __init__(self, tag: int, n: int) -> None:
        self.tag, self.n = tag, n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([self.tag, i]), torch.tensor(0)


def test_indexed_dataset_delegates() -> None:
    base = {0: _Const(0, 5), 1: _Const(1, 5), 2: _Const(2, 5)}
    index = np.array([[2, 4], [0, 1], [1, 3]])
    ds = IndexedSplitDataset(base, index)
    assert len(ds) == 3
    assert ds[0][0].tolist() == [2, 4]
    assert ds[2][0].tolist() == [1, 3]
