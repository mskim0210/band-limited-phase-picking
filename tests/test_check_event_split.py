"""Tests for scripts/check_event_split.py (event-level split audit)."""

from __future__ import annotations

import numpy as np

from scripts.check_event_split import coverage_report, overlap_report


def test_overlap_report_counts_shared_source_ids_pairwise() -> None:
    sids = {"train": np.array(["a", "a", "b", None], dtype=object),
            "val": np.array(["c", "b"], dtype=object),
            "test": np.array(["d"], dtype=object)}
    r = overlap_report(sids)
    assert r["unique_events"] == {"train": 2, "val": 2, "test": 1}
    assert r["pairwise_intersection"] == {"train∩val": 1, "train∩test": 0, "val∩test": 0}
    assert r["disjoint"] is False
    assert r["noise_traces"] == {"train": 1, "val": 0, "test": 0}


def test_coverage_report_checks_every_memmap_row_used_once() -> None:
    idx = {"train": np.array([[0, 0], [0, 1], [1, 0]]),
           "val": np.array([[0, 2]]),
           "test": np.array([[2, 0], [2, 1]])}
    r = coverage_report(idx, n_rows={0: 3, 1: 1, 2: 2})
    assert r["rows_total"] == 6 and r["rows_used_once"] is True
    bad = coverage_report({"train": np.array([[0, 0], [0, 0]])}, n_rows={0: 2})
    assert bad["rows_used_once"] is False
