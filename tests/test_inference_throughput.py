"""Tests for scripts/inference_throughput.py (Table 10).

Forward-only throughput and end-to-end throughput with per-window CPU
wavelet encoding, for the full-band, dilated and CR = 4 models.
"""

from __future__ import annotations

import torch

from scripts.inference_throughput import measure_traces_per_sec, ratios


class CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x


def test_measure_traces_per_sec_times_only_the_measured_batches() -> None:
    m = CountingModel()
    x = torch.zeros(8, 3, 600)
    tps = measure_traces_per_sec(m, x, n_batches=5, warmup=2, device=torch.device("cpu"))
    assert m.calls == 7
    assert tps > 0


def test_measure_traces_per_sec_applies_optional_per_batch_encoding() -> None:
    seen = []

    def encode(xb: torch.Tensor) -> torch.Tensor:
        seen.append(xb.shape)
        return xb[:, :, ::4]

    m = CountingModel()
    x = torch.zeros(8, 3, 600)
    measure_traces_per_sec(m, x, n_batches=3, warmup=1, device=torch.device("cpu"), encode=encode)
    assert len(seen) == 4 and seen[0] == (8, 3, 600)


def test_ratios_match_table_10_definition() -> None:
    j = {"baseline": 7159, "dilated": 7055, "cr4": 6175,
         "baseline_forward_only": 7589, "dilated_forward_only": 7538, "cr4_forward_only": 21799}
    r = ratios(j)
    assert r["cr4_forward_only_vs_baseline"] == 2.87
    assert r["cr4_end_to_end_vs_baseline"] == 0.86
    assert r["dilated_forward_only_vs_baseline"] == 0.99
