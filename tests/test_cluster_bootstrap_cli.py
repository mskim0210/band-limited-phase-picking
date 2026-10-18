"""Unit tests for the run-selection helper of scripts/cluster_bootstrap.py."""

from __future__ import annotations

from scripts.cluster_bootstrap import make_run_configs


def test_make_run_configs_expands_runs_and_seeds() -> None:
    cfgs = make_run_configs(["phasenet_baseline", "eqt_wavelet_cr4"], [42, 7])
    assert cfgs == [
        ("phasenet_baseline", "seed42", "outputs/checkpoints/phasenet_baseline_seed42/best_model.pt"),
        ("phasenet_baseline", "seed7", "outputs/checkpoints/phasenet_baseline_seed7/best_model.pt"),
        ("eqt_wavelet_cr4", "seed42", "outputs/checkpoints/eqt_wavelet_cr4_seed42/best_model.pt"),
        ("eqt_wavelet_cr4", "seed7", "outputs/checkpoints/eqt_wavelet_cr4_seed7/best_model.pt"),
    ]


def test_make_run_configs_default_matches_legacy_order() -> None:
    cfgs = make_run_configs(["l2_baseline", "wavelet_cr4"], [42, 123, 7])
    assert [c[2] for c in cfgs][:2] == [
        "outputs/checkpoints/l2_baseline_seed42/best_model.pt",
        "outputs/checkpoints/l2_baseline_seed123/best_model.pt",
    ]
    assert len(cfgs) == 6
