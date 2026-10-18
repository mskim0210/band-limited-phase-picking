"""Unit tests for scripts/summarize_revision.py (result aggregation for the C&G revision)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.summarize_revision import aggregate_runs, fmt_pm


def _fake_run(root: Path, name: str, seed: int, p_f1: float, s_f1: float,
              wall_min: float, epochs: int, p_mae: float = 0.02) -> None:
    ck = root / "checkpoints" / f"{name}_seed{seed}"
    ck.mkdir(parents=True)
    lines = ["epoch,wall_clock_sec,train_loss,val_loss,val_p_f1,val_s_f1"]
    for e in range(1, epochs + 1):
        lines.append(f"{e},{wall_min * 60 * e / epochs:.1f},0.01,0.01,0.9,0.8")
    (ck / "epoch_log.csv").write_text("\n".join(lines) + "\n")
    res = root / "results"
    res.mkdir(exist_ok=True)
    (res / f"{name}_seed{seed}_test.json").write_text(json.dumps({
        "tolerance_0.1s": {"P": {"f1": p_f1, "precision": 0.9, "recall": 0.9},
                           "S": {"f1": s_f1, "precision": 0.8, "recall": 0.8}},
        "timing": {"P": {"mae_sec": p_mae, "mean_residual_sec": 0.001},
                   "S": {"mae_sec": 0.06, "mean_residual_sec": 0.0}},
    }))


def test_aggregate_runs_mean_std_and_wallclock(tmp_path: Path) -> None:
    for seed, p in zip((42, 123, 7), (0.90, 0.92, 0.91)):
        _fake_run(tmp_path, "fake_base", seed, p, 0.80, wall_min=100.0, epochs=10)
    agg = aggregate_runs("fake_base", seeds=(42, 123, 7), root=tmp_path)
    assert agg["n_seeds"] == 3
    assert abs(agg["p_f1_mean"] - 0.91) < 1e-9
    assert abs(agg["p_f1_std"] - np.std([0.90, 0.92, 0.91])) < 1e-9
    assert abs(agg["wall_min_mean"] - 100.0) < 1e-6
    assert agg["epochs_mean"] == 10
    assert abs(agg["p_mae_ms_mean"] - 20.0) < 1e-6


def test_aggregate_runs_missing_returns_none(tmp_path: Path) -> None:
    assert aggregate_runs("nope", seeds=(42,), root=tmp_path) is None


def test_fmt_pm() -> None:
    assert fmt_pm(0.97553, 0.00041) == "0.9755 ± 0.0004"
    assert fmt_pm(0.97553, 0.00041, digits=3) == "0.976 ± 0.000"
