"""Tests for the fixed-recipe distillation script (Section 3.7, Tables 16–17).

The script must report, from the same run that produces the Table 17
fixed-recipe row, the matching-tolerance sensitivity (Table 16) and the
adaptation cost (teacher soft labels, student distillation, encoding) so that
the manuscript quotes one protocol only.
"""

from __future__ import annotations

import numpy as np

from scripts.distill_fixed_recipe import TOLS, summarize_runs, tolerance_rows


def _probs_with_peaks(n: int, p_at: int, s_at: int) -> np.ndarray:
    probs = np.zeros((n, 3, 6000), np.float32)
    probs[:, 0] = 1.0
    for ch, at in ((1, p_at), (2, s_at)):
        probs[:, ch, at] = 0.9
        probs[:, 0, at] = 0.1
    return probs


def test_tolerance_rows_recall_grows_with_tolerance() -> None:
    probs = _probs_with_peaks(4, p_at=3015, s_at=3540)  # P 0.15 s late, S 0.40 s late
    P, S = np.full(4, 3000), np.full(4, 3500)
    rows = tolerance_rows(probs, P, S, thr=0.3)
    assert TOLS == (0.1, 0.2, 0.5)
    assert [r["tol_sec"] for r in rows] == [0.1, 0.2, 0.5]
    assert [r["P_recall"] for r in rows] == [0.0, 1.0, 1.0]
    assert [r["S_recall"] for r in rows] == [0.0, 0.0, 1.0]
    assert rows[2]["P_precision"] == 1.0 and rows[2]["S_precision"] == 1.0


def _seed(seed: int, train: float, teacher: float, n: int, p_rec01: float) -> dict:
    test = {"P": {"f1": 0.6, "recall": 0.5, "precision": 0.8, "res_mae": 0.07},
            "S": {"f1": 0.8, "recall": 0.7, "precision": 0.9, "res_mae": 0.04},
            "ft_per_hour_pooled": 11.0}
    return {"seed": seed, "train_sec": train, "teacher_sec": teacher, "n_adapt_windows": n,
            "test": test,
            "student_tolerance": [{"tol_sec": 0.1, "P_recall": p_rec01, "S_recall": 0.6,
                                   "P_precision": 0.8, "S_precision": 0.9},
                                  {"tol_sec": 0.5, "P_recall": 0.5, "S_recall": 0.7,
                                   "P_precision": 0.8, "S_precision": 0.9}],
            "teacher_tolerance": [{"tol_sec": 0.1, "P_recall": 0.6, "S_recall": 0.6,
                                   "P_precision": 0.8, "S_precision": 0.9},
                                  {"tol_sec": 0.5, "P_recall": 0.7, "S_recall": 0.65,
                                   "P_precision": 0.8, "S_precision": 0.9}]}


def test_summarize_runs_reports_timing_and_tolerance_means() -> None:
    per_seed = [_seed(42, 27.0, 0.3, 1400, 0.30), _seed(123, 23.0, 0.3, 1500, 0.40)]
    s = summarize_runs(per_seed)
    assert s["P"]["f1"] == (0.6, 0.0) and s["S"]["res_mae"] == (0.04, 0.0)
    assert s["ft_per_hour_pooled"] == 11.0
    assert s["timing"] == {"train_sec_mean": 25.0, "teacher_sec_mean": 0.3,
                           "n_adapt_windows_mean": 1450.0}
    assert s["tolerance_sweep"]["student"]["0.1"]["P_recall"] == 0.35
    assert s["tolerance_sweep"]["student"]["0.5"]["S_recall"] == 0.7
    assert s["tolerance_sweep"]["teacher_zero_shot"]["0.5"]["P_recall"] == 0.7
