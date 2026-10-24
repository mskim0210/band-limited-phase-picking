"""Tests for the P+S extension of scripts/analyze_s_snr_waveform.py.

The waveform-SNR stratification (Table 12) must report P-F1 next to S-F1 so
the low-SNR P gain of Table 8 (metadata SNR) can be checked under the
waveform-derived SNR.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from scripts.analyze_s_snr_waveform import BINS, bin_summary, per_trace_f1


class _PeakModel(nn.Module):
    """Ignores the input; emits a P peak at sample 1000 and an S peak at 2000."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.shape[0], 3, 6000), -5.0)
        logits[:, 0] = 5.0
        logits[:, 1, 1000] = 10.0
        logits[:, 2, 2000] = 10.0
        return logits


def test_per_trace_f1_scores_both_phases_and_skips_unkept_traces() -> None:
    X = np.random.RandomState(0).randn(3, 3, 6000).astype(np.float32)
    p_true = np.array([1000, 1500, 1000])
    s_true = np.array([2000, 2000, 2000])
    keep = np.array([True, True, False])
    f1 = per_trace_f1(_PeakModel().eval(), X, p_true, s_true, keep, bs=2)
    assert f1["p"][0] == 1.0 and f1["s"][0] == 1.0
    assert f1["p"][1] == 0.0 and f1["s"][1] == 1.0
    assert np.isnan(f1["p"][2]) and np.isnan(f1["s"][2])


def test_bin_summary_reports_n_and_mean_f1_per_phase() -> None:
    f1 = {"p": np.array([1.0, 0.0, 1.0, np.nan]), "s": np.array([1.0, 1.0, 0.0, np.nan])}
    snr = np.array([2.0, 3.0, 25.0, np.nan])
    keep = ~np.isnan(snr)
    out = bin_summary(f1, snr, keep, BINS)
    assert out["<5"] == {"n": 2, "p_f1": 0.5, "s_f1": 1.0}
    assert out[">20"] == {"n": 1, "p_f1": 1.0, "s_f1": 0.0}
    assert out["5-10"]["n"] == 0
