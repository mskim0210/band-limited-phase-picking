"""Tests for scripts/paired_bootstrap.py (paired trace-bootstrap CIs of ΔF1).

Two models evaluated on the same test traces are compared by resampling the
same traces for both (and for every seed), so the CI describes the paired
difference of the seed-mean F1 rather than two independent intervals.
"""

from __future__ import annotations

import numpy as np

from scripts.paired_bootstrap import counts_array, f1_from_counts, paired_delta_ci


def test_f1_from_counts_matches_definition() -> None:
    f1 = f1_from_counts(np.array([3]), np.array([1]), np.array([1]))
    assert abs(f1[0] - 0.75) < 1e-12  # 2*3 / (2*3 + 1 + 1)
    assert f1_from_counts(np.array([0]), np.array([0]), np.array([0]))[0] == 0.0


def test_counts_array_from_per_trace_dicts() -> None:
    rows = [{"p_tp": 1, "p_fp": 0, "p_fn": 0, "s_tp": 0, "s_fp": 1, "s_fn": 1},
            {"p_tp": 0, "p_fp": 2, "p_fn": 1, "s_tp": 1, "s_fp": 0, "s_fn": 0}]
    c = counts_array(rows)
    assert c.shape == (2, 6) and c.tolist() == [[1, 0, 0, 0, 1, 1], [0, 2, 1, 1, 0, 0]]


def _counts(tp: np.ndarray) -> np.ndarray:
    fp = np.zeros_like(tp)
    return np.stack([tp, fp, 1 - tp, tp, fp, 1 - tp], axis=1)


def test_identical_models_give_zero_delta_and_degenerate_ci() -> None:
    tp = np.random.RandomState(0).binomial(1, 0.8, 500)
    r = paired_delta_ci([_counts(tp)], [_counts(tp)], B=100, seed=1)
    assert r["P"]["delta_pts"] == 0.0
    assert r["P"]["ci_low_pts"] == 0.0 and r["P"]["ci_high_pts"] == 0.0


def test_better_model_gives_positive_delta_with_ci_above_zero_and_per_seed_entries() -> None:
    rng = np.random.RandomState(0)
    tp_a = rng.binomial(1, 0.7, 2000)
    tp_b = np.maximum(tp_a, rng.binomial(1, 0.5, 2000))  # model b finds a superset
    r = paired_delta_ci([_counts(tp_a), _counts(tp_a)], [_counts(tp_b), _counts(tp_b)], B=200, seed=1)
    assert r["P"]["delta_pts"] > 0 and r["P"]["ci_low_pts"] > 0
    assert len(r["P"]["per_seed"]) == 2 and r["S"]["delta_pts"] == r["P"]["delta_pts"]
    assert r["B"] == 200 and r["n_traces"] == 2000


def test_singleton_clusters_reproduce_trace_resampling() -> None:
    rng = np.random.RandomState(0)
    tp_a = rng.binomial(1, 0.7, 500)
    tp_b = np.maximum(tp_a, rng.binomial(1, 0.3, 500))
    base = paired_delta_ci([_counts(tp_a)], [_counts(tp_b)], B=50, seed=3)
    same = paired_delta_ci([_counts(tp_a)], [_counts(tp_b)], B=50, seed=3, clusters=np.arange(500))
    assert same["P"] == base["P"] and same["resampling"] == "event_cluster" and same["n_clusters"] == 500
    assert base["resampling"] == "trace"


def test_single_cluster_gives_degenerate_interval_at_point_estimate() -> None:
    rng = np.random.RandomState(1)
    tp_a = rng.binomial(1, 0.6, 300)
    tp_b = np.maximum(tp_a, rng.binomial(1, 0.3, 300))
    r = paired_delta_ci([_counts(tp_a)], [_counts(tp_b)], B=20, seed=0, clusters=np.zeros(300, dtype=int))
    assert r["P"]["ci_low_pts"] == r["P"]["delta_pts"] == r["P"]["ci_high_pts"]
