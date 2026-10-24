"""Waveform-derived SNR stratification of P-F1 and S-F1 for the four Section 3.4 models.

Explains the SNR-dependent sign flip of the S-F1
change at CR=4 (+3.0 pp below 5 dB, -0.9 pp above 20 dB) with data rather
than speculation, by stratifying the two controls (LP-only: band removal
only; dilated: receptive-field enlargement only) alongside baseline and
wavelet CR=4.

NOTE: STEAD's metadata snr_db is no longer available on disk (the raw
archive was removed after preprocessing), so this analysis uses a
waveform-derived SNR computed identically for every trace:
    SNR = 10 log10( var(Z[P .. P+3 s]) / var(Z[P-3 s .. P]) )
Traces without a P arrival (noise) or with P < 3 s are excluded. Binning
follows the paper's <5 / 5-10 / 10-20 / >20 dB scheme. Because the SNR
definition differs from STEAD metadata, absolute bin membership differs
slightly from Table VIII / S-I; all four models share the identical
stratification, so the comparison is internally consistent.

Usage:
    python scripts/analyze_s_snr_waveform.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import pick_peaks, compute_phase_metrics
from cs_pipeline.cs_aware_model import CSPhaseNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

from omegaconf import OmegaConf
MEM = Path(OmegaConf.load("configs/stead_experiment.yaml").data.memmap_dir)
CS = Path("data/cs_encoded")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 100
BINS = [("<5", -999, 5), ("5-10", 5, 10), ("10-20", 10, 20), (">20", 20, 999)]


def load_model(ck: str) -> CSPhaseNet:
    d = torch.load(ck, map_location="cpu", weights_only=False)
    m = CSPhaseNet(M=d["config"]["M"], N=6000,
                   base_channels=d["config"].get("base_channels", 16),
                   dilation=d["config"].get("dilation", 1))
    m.load_state_dict(d["model_state_dict"])
    return m.to(DEV).eval()


@torch.no_grad()
def per_trace_f1(model, X: np.ndarray, p_true: np.ndarray, s_true: np.ndarray,
                 keep: np.ndarray, bs: int = 256) -> dict[str, np.ndarray]:
    """Per-trace P-F1 and S-F1 (tol=10 samples) for kept traces; NaN elsewhere."""
    out = {"p": np.full(len(X), np.nan, np.float32),
           "s": np.full(len(X), np.nan, np.float32)}
    idx = np.where(keep)[0]
    for i0 in range(0, len(idx), bs):
        b = idx[i0:i0 + bs]
        x = torch.from_numpy(np.ascontiguousarray(X[b])).float()
        x = x / (x.std(dim=(1, 2), keepdim=True) + 1e-8)
        probs = torch.softmax(model(x.to(DEV)), 1).cpu().numpy()
        for j, t in enumerate(b):
            pred = pick_peaks(probs[j], threshold=0.3)
            out["p"][t] = compute_phase_metrics(pred["P"], [int(p_true[t])], 10)["f1"]
            out["s"][t] = compute_phase_metrics(pred["S"], [int(s_true[t])], 10)["f1"]
        if i0 % (bs * 100) == 0:
            logger.info("  %d/%d", i0, len(idx))
    return out


def bin_summary(f1: dict[str, np.ndarray], snr: np.ndarray, keep: np.ndarray,
                bins: list[tuple[str, float, float]]) -> dict[str, dict]:
    """Trace count and mean per-trace P-F1 / S-F1 in each SNR bin."""
    res: dict[str, dict] = {}
    for lab, lo, hi in bins:
        m = keep & (snr >= lo) & (snr < hi)
        res[lab] = {"n": int(m.sum())}
        for ph in ("p", "s"):
            res[lab][f"{ph}_f1"] = round(float(np.nanmean(f1[ph][m])), 4) if m.any() else float("nan")
    return res


def main() -> None:
    W = np.load(MEM / "test_waveforms.npy", mmap_mode="r")
    arr = np.load(MEM / "test_arrivals.npz")
    p, s = arr["p_samples"].astype(int), arr["s_samples"].astype(int)

    # waveform-derived SNR on Z (channel 2)
    snr = np.full(len(W), np.nan, np.float32)
    win = 3 * FS
    for i in range(len(W)):
        if p[i] < win or p[i] + win >= 6000 or s[i] < 0:
            continue
        z = np.asarray(W[i, 2], np.float32)
        vs = z[p[i]:p[i] + win].var()
        vn = z[p[i] - win:p[i]].var()
        if vn > 0:
            snr[i] = 10 * np.log10(max(vs / vn, 1e-9))
        if i % 20000 == 0:
            logger.info("snr %d/%d", i, len(W))
    keep = ~np.isnan(snr)
    logger.info("SNR computed for %d/%d traces", keep.sum(), len(W))
    excluded = {"total_test_traces": int(len(W)),
                "no_s_pick": int((s < 0).sum()),
                "p_before_3s": int(((p < win) & (s >= 0)).sum()),
                "p_window_past_end": int(((p + win >= 6000) & (p >= win) & (s >= 0)).sum()),
                "zero_noise_variance": int((keep.sum() == 0) or (len(W) - (s < 0).sum() - ((p < win) & (s >= 0)).sum() - ((p + win >= 6000) & (p >= win) & (s >= 0)).sum() - keep.sum())),
                "kept": int(keep.sum())}
    logger.info("exclusions: %s", excluded)


    models = {
        "baseline": ("outputs/checkpoints/l2_baseline_seed42/best_model.pt",
                     MEM / "test_waveforms.npy"),
        "lp_only": ("outputs/checkpoints/lowpass_cr1_seed42/best_model.pt",
                    CS / "lowpass_cr1/test_waveforms.npy"),
        "dilated": ("outputs/checkpoints/l2_baseline_dil4_seed42/best_model.pt",
                    MEM / "test_waveforms.npy"),
        "wavelet_cr4": ("outputs/checkpoints/wavelet_cr4_seed42/best_model.pt",
                        CS / "wavelet_cr4/test_waveforms.npy"),
    }

    res = {}
    for name, (ck, xpath) in models.items():
        X = np.load(xpath, mmap_mode="r")
        f1 = per_trace_f1(load_model(ck), X, p, s, keep)
        res[name] = bin_summary(f1, snr, keep, BINS)
        logger.info("%s: %s", name, {k: (v["p_f1"], v["s_f1"]) for k, v in res[name].items()})

    out = {"snr_definition": "10*log10(var(Z[P..P+3s])/var(Z[P-3s..P]))",
           "tolerance_samples": 10, "threshold": 0.3, "seed": 42,
           "exclusions": excluded, "results": res}
    Path("outputs/results/s_snr_waveform.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out["results"], indent=1))


if __name__ == "__main__":
    main()
