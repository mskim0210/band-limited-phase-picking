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


def _profile(io: float, pre: float, h2d: float, comp: float, tps: float, gpu: float) -> dict:
    return {"metadata": {}, "summary": {"t_io": {"mean": io}, "t_preprocess": {"mean": pre},
                                        "t_h2d": {"mean": h2d}, "t_compute": {"mean": comp},
                                        "throughput_traces_per_sec": tps, "gpu_util_mean_pct": gpu}}


FAKE_PROFILES = {"L0": _profile(0.158, 0.0, 0.0003, 0.0188, 361.0, 10.9),
                 "L2": _profile(0.0001, 0.0, 0.0002, 0.0182, 3487.0, 66.3),
                 "L3": _profile(0.0001, 0.0, 0.0001, 0.0081, 7844.0, 8.5)}


def test_step_decomposition_data_has_the_four_paper_bars() -> None:
    """Fig. 3 of the paper: L0, L2, the wavelet CR = 4 bar (per-step time measured
    from the training steps, I/O and H2D from the shape-identical L3 profile, no
    GPU-utilization point) and the Gaussian CR = 8 bar."""
    from scripts.generate_figures import CR4_TOTAL_MS, step_decomposition_data
    data = step_decomposition_data(FAKE_PROFILES)
    assert list(data) == ["L0\n(HDF5)", "L2\n(memmap)", "L3 CR=4\n(recomm.)", "L3 CR=8\n(max)"]
    comps, tps, gpu = data["L3 CR=4\n(recomm.)"]
    assert abs(sum(comps) - CR4_TOTAL_MS) < 1e-9 and comps[0] == 0.10 and comps[2] == 0.06
    assert round(tps) == 6882 and gpu is None
    assert data["L3 CR=8\n(max)"][2] is None and data["L0\n(HDF5)"][2] == 10.9
    assert abs(data["L0\n(HDF5)"][0][0] - 158.0) < 1e-9  # ms


def test_generate_fig3_step_decomposition_writes_the_paper_file(tmp_path: Path) -> None:
    from scripts.generate_figures import generate_fig3_step_decomposition
    out = generate_fig3_step_decomposition(FAKE_PROFILES, tmp_path)
    assert out == tmp_path / "fig2_step_decomposition.png" and out.stat().st_size > 1000


def test_make_fig2_cr4_delegates_to_the_shared_builder_and_the_three_bar_generator_is_gone() -> None:
    import scripts.generate_figures as gf
    import scripts.make_fig2_cr4 as wrapper
    assert wrapper.build is gf.generate_fig3_step_decomposition
    assert not hasattr(gf, "generate_fig2")
