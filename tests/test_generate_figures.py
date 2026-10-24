"""Tests for the figure-generation reproduction path (Figs. 3, 5, 8).

The profile loader must select the single-model profiles the paper uses by
file name instead of keying every JSON in the directory by pipeline level
(where the multi-model and worker-sweep profiles would overwrite them), and
a missing input must be reported as an error instead of being skipped under
an "all figures generated" message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_figures import PAPER_PROFILES, load_profiles, missing_inputs, wall_clock_mean_min


def test_wall_clock_mean_min_comes_from_the_epoch_logs(tmp_path: Path) -> None:
    """Fig. 5 speedup annotations use the shipped epoch logs (as Table 5 does),
    not a separate training summary file."""
    for seed, minutes in ((42, 100.0), (123, 80.0), (7, 90.0)):
        d = tmp_path / "checkpoints" / f"fake_seed{seed}"
        d.mkdir(parents=True)
        (d / "epoch_log.csv").write_text("epoch,wall_clock_sec,val_p_f1\n1,10.0,0.5\n"
                                         f"2,{minutes * 60:.1f},0.9\n")
    assert wall_clock_mean_min("fake", root=tmp_path) == 90.0
    assert wall_clock_mean_min("nope", root=tmp_path) is None


def test_paper_profiles_name_the_single_model_files() -> None:
    assert PAPER_PROFILES == {"L0": "l0_stead_hdf5.json", "L1": "l1_seisbench_trace_blocks.json",
                              "L2": "l2_stead_memmap.json", "L3": "l3_cs_gaussian_cr8.json"}


def test_load_profiles_uses_only_the_named_files(tmp_path: Path) -> None:
    for fname, level, tag in (("l0_stead_hdf5.json", "L0", "paper"),
                              ("multimodel_csphaseNet_M_l0_stead_hdf5.json", "L0", "multimodel"),
                              ("exp3_nw8_l2_stead_memmap.json", "L2", "sweep"),
                              ("l2_stead_memmap.json", "L2", "paper")):
        (tmp_path / fname).write_text(json.dumps({"metadata": {"level": level}, "tag": tag}))
    prof = load_profiles(tmp_path, files={"L0": "l0_stead_hdf5.json", "L2": "l2_stead_memmap.json"})
    assert set(prof) == {"L0", "L2"}
    assert prof["L0"]["tag"] == "paper" and prof["L2"]["tag"] == "paper"


def test_load_profiles_raises_on_a_missing_named_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_profiles(tmp_path, files={"L0": "nope.json"})


def test_missing_inputs_lists_the_absent_files(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}")
    assert missing_inputs({"Fig. 5": tmp_path / "nope.json", "Fig. 8": tmp_path / "a.json"}) == ["Fig. 5"]
