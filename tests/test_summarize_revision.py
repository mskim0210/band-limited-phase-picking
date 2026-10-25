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


def test_aggregate_runs_falls_back_to_the_consolidated_summary(tmp_path: Path) -> None:
    """The CR-sweep runs exist only inside test_evaluation_summary.json (per_run)."""
    res = tmp_path / "results"
    res.mkdir()
    per_run = {}
    for seed, p in zip((42, 123, 7), (0.90, 0.92, 0.91)):
        per_run[f"fake_base_seed{seed}"] = {
            "tolerance_0.1s": {"P": {"f1": p, "precision": 0.9, "recall": 0.9},
                               "S": {"f1": 0.80, "precision": 0.8, "recall": 0.8}},
            "timing": {"P": {"mae_sec": 0.02, "mean_residual_sec": 0.001},
                       "S": {"mae_sec": 0.06, "mean_residual_sec": 0.0}}}
    (res / "test_evaluation_summary.json").write_text(json.dumps({"per_run": per_run, "summary": {}}))
    agg = aggregate_runs("fake_base", seeds=(42, 123, 7), root=tmp_path)
    assert agg is not None and agg["n_seeds"] == 3
    assert abs(agg["p_f1_mean"] - 0.91) < 1e-9 and abs(agg["s_f1_mean"] - 0.80) < 1e-9
    assert abs(agg["p_mae_ms_mean"] - 20.0) < 1e-6


def test_table_d_is_the_fixed_recipe_table_without_ceiling_language(tmp_path: Path) -> None:
    from scripts.summarize_revision import fixed_recipe_rows, table_d_markdown
    res = tmp_path / "results"
    res.mkdir()
    (res / "manual2023_distill_fixed_cr2.json").write_text(json.dumps({
        "meta": {"student_cr": 2}, "per_seed": [{"train_sec": 30.0}, {"train_sec": 32.0}],
        "summary": {"P": {"f1": [0.765, 0.016]}, "S": {"f1": [0.803, 0.005]}}}))
    (res / "manual2023_distill_fixed_rerun.json").write_text(json.dumps({
        "meta": {"student_cr": 4}, "per_seed": [{"train_sec": 27.3}],
        "summary": {"P": {"f1": [0.584, 0.011]}, "S": {"f1": [0.817, 0.010]},
                    "timing": {"train_sec_mean": 25.1}}}))
    (res / "manual2023_distill_fixed_cr8.json").write_text(json.dumps({
        "meta": {"student_cr": 8}, "per_seed": [{"train_sec": 19.7}],
        "summary": {"P": {"f1": [0.366, 0.023]}, "S": {"f1": [0.697, 0.011]}}}))
    rows = fixed_recipe_rows(tmp_path)
    assert [r["cr"] for r in rows] == [2, 4, 8]
    assert rows[0]["train_sec_mean"] == 31.0   # mean of per-seed times when no timing block
    assert rows[1]["train_sec_mean"] == 25.1   # timing block preferred
    assert rows[1]["p_f1"] == (0.584, 0.011)
    md = table_d_markdown(rows).lower()
    assert "fixed-recipe" in md  # the manuscript's term (Table 17)
    assert "ceiling" not in md and "recovery" not in md and "residual" not in md


def test_write_outputs_creates_parent_directories_and_writes_utf8(tmp_path: Path) -> None:
    """A fresh checkout has no manuscript/ directory; the writer must create
    the parents and encode the × and ± characters as UTF-8 regardless of locale."""
    from scripts.summarize_revision import DEFAULT_MD_OUT, write_outputs
    json_out = tmp_path / "a" / "b" / "summary.json"
    md_out = tmp_path / "c" / "tables.md"
    write_outputs({"table_A": [{"speedup": 2.1}]}, ["# tables", "| 2.1× | 0.98 ± 0.01 |"], json_out, md_out)
    assert json.loads(json_out.read_text(encoding="utf-8"))["table_A"][0]["speedup"] == 2.1
    assert md_out.read_text(encoding="utf-8") == "# tables\n| 2.1× | 0.98 ± 0.01 |"  # universal newlines: Windows writes CRLF
    assert DEFAULT_MD_OUT.endswith(".md") and DEFAULT_MD_OUT.startswith("outputs/")  # no manuscript/ in a checkout
