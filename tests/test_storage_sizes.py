"""Tests for scripts/storage_sizes.py (Table 5 storage column, text percentages).

Storage is the float32 memmap payload: traces × 3 components × M × 4 bytes.
GB means 10⁹ bytes throughout, and the saving is computed from bytes, never
from rounded gigabytes.
"""

from __future__ import annotations

from scripts.storage_sizes import payload_bytes, saving_pct, storage_table

N_TRACES = 1_265_657


def test_payload_bytes_is_float32_times_three_components() -> None:
    assert payload_bytes(N_TRACES, 6000) == 91_127_304_000
    assert payload_bytes(N_TRACES, 1506) == 22_872_953_304


def test_saving_is_computed_from_bytes_not_rounded_gigabytes() -> None:
    assert saving_pct(3004) == 49.9
    assert saving_pct(1506) == 74.9
    assert saving_pct(382) == 93.6


def test_storage_table_rows_carry_gb_and_percent_strings() -> None:
    rows = storage_table(N_TRACES, {1: 6000, 2: 3004, 4: 1506, 8: 756, 16: 382})
    assert [r["cr"] for r in rows] == [1, 2, 4, 8, 16]
    assert rows[0]["gb_str"] == "91.1 GB" and rows[0]["saving_pct"] == 0.0
    assert rows[1]["gb_str"] == "45.6 GB" and rows[1]["saving_str"] == "50%"
    assert rows[2]["gb_str"] == "22.9 GB" and rows[2]["saving_str"] == "75%"
    assert rows[4]["gb_str"] == "5.8 GB"
