"""Quantify what band truncation does to the waveform itself (C&G R1-8).

For each CR in {2, 4, 8, 16} the k finest db4 detail bands that the wavelet
encoder discards are zeroed and the trace is reconstructed at full length
(the time-domain equivalent of the retained approximation).  Reported per CR:

* NRMSE over the whole trace and over 1-s windows after the P and S onsets,
* onset lag (cross-correlation shift, ms) in a ±1-s window around the P onset,
* mean signed pick residual (bias) and MAE of the trained CR models on the
  STEAD test set (read from ``outputs/results/test_evaluation_summary.json``).

Usage:
    python scripts/waveform_distortion.py --n-traces 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pywt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs_pipeline.sensing_matrix import WaveletSensingMatrix  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

FS = 100.0
CRS = (2, 4, 8, 16)


def fs_nominal(cr: int, fs: float = FS) -> float:
    """Nominal cutoff of the retained approximation band: fs / (2 * cr)."""
    return fs / (2.0 * cr)


def band_limited_reconstruct(x: np.ndarray, cr: int, wavelet: str = "db4",
                             level: int = 6) -> np.ndarray:
    """Zero the detail bands removed at ``cr`` and reconstruct at full length.

    Args:
        x: (B, C, N) array.
        cr: Compression ratio (2, 4, 8, 16) as in ``WaveletSensingMatrix``.
        wavelet: PyWavelets wavelet name.
        level: Decomposition level.

    Returns:
        (B, C, N) float32 reconstruction with the same length as ``x``.
    """
    keep = WaveletSensingMatrix._CR_TO_KEEP_LEVELS[cr]
    B, C, N = x.shape
    out = np.empty_like(x, dtype=np.float32)
    for b in range(B):
        for c in range(C):
            coeffs = pywt.wavedec(x[b, c], wavelet, level=level, mode="symmetric")
            for i in range(keep + 1, len(coeffs)):
                coeffs[i] = np.zeros_like(coeffs[i])
            out[b, c] = pywt.waverec(coeffs, wavelet, mode="symmetric")[:N]
    return out


def nrmse(a: np.ndarray, ref: np.ndarray) -> float:
    """||a - ref|| / ||ref|| (0 when identical, 1 when ``a`` is all zeros)."""
    denom = float(np.linalg.norm(ref))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(a - ref) / denom)


def onset_lag_samples(x: np.ndarray, xr: np.ndarray, onset: int, half_window: int = 100,
                      max_lag: int = 50) -> int:
    """Lag (samples) that best aligns ``xr`` to ``x`` around ``onset``.

    Positive means ``xr`` is delayed relative to ``x``.
    """
    lo, hi = max(onset - half_window, max_lag), min(onset + half_window, len(x) - max_lag)
    seg = x[lo:hi]
    best, best_val = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        v = float(np.dot(seg, xr[lo + lag:hi + lag]))  # xr shifted by +lag matches x
        if v > best_val:
            best_val, best = v, lag
    return best


def frequency_response(cr: int, freqs_hz: tuple[float, ...] = (1.0, 2.0, 5.0, 8.0, 10.0, 12.5, 15.0, 20.0, 25.0),
                       n: int = 6000, fs: float = FS) -> dict[float, float]:
    """NRMSE of a pure sinusoid after band truncation, per frequency.

    Characterizes the effective (soft) db4 roll-off relative to the nominal
    cutoff fs / (2 * cr).  0 = passed unchanged, 1 = fully removed.
    """
    t = np.arange(n) / fs
    out: dict[float, float] = {}
    for f in freqs_hz:
        x = np.sin(2 * np.pi * f * t).astype(np.float32)[None, None]
        xr = band_limited_reconstruct(x, cr)
        out[float(f)] = round(nrmse(xr[0, 0, 500:-500], x[0, 0, 500:-500]), 4)
    return out


def noise_transfer_function(cr: int, seed: int = 0, n_traces: int = 64, n: int = 6000,
                            fs: float = FS, nperseg: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Power transfer |H(f)|^2 of band truncation estimated from white noise.

    Welch PSD of the reconstruction divided by that of the input, averaged over
    ``n_traces`` independent Gaussian traces.  Smooth and free of the
    coherent-aliasing artefact that pure tones show at dyadic band edges.

    Returns:
        (freqs_hz, ratio) with ratio clipped to [0, 1].
    """
    from scipy.signal import welch

    rng = np.random.RandomState(seed)
    x = rng.randn(n_traces, 1, n).astype(np.float32)
    xr = band_limited_reconstruct(x, cr)
    f, pxx = welch(x[:, 0], fs=fs, nperseg=nperseg, axis=-1)
    _, prr = welch(xr[:, 0], fs=fs, nperseg=nperseg, axis=-1)
    ratio = prr.mean(0) / pxx.mean(0)
    return f, np.clip(ratio, 0.0, 1.0)


def _window_nrmse(x: np.ndarray, xr: np.ndarray, onset: int, n: int = 100) -> float | None:
    if onset <= 0 or onset + n > x.shape[-1]:
        return None
    return nrmse(xr[..., onset:onset + n], x[..., onset:onset + n])


def main() -> None:
    ap = argparse.ArgumentParser(description="Waveform distortion of band truncation")
    ap.add_argument("--config", default="configs/stead_experiment.yaml")
    ap.add_argument("--memmap-dir", default=None, help="default: data.memmap_dir from --config")
    ap.add_argument("--n-traces", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-summary", default="outputs/results/test_evaluation_summary.json")
    ap.add_argument("--output", default="outputs/results/waveform_distortion.json")
    args = ap.parse_args()
    if args.memmap_dir is None:
        from omegaconf import OmegaConf
        args.memmap_dir = OmegaConf.load(args.config).data.memmap_dir

    wf = np.load(f"{args.memmap_dir}/test_waveforms.npy", mmap_mode="r")
    arr = np.load(f"{args.memmap_dir}/test_arrivals.npz")
    p_all, s_all = arr["p_samples"], arr["s_samples"]
    rng = np.random.RandomState(args.seed)
    idx = np.sort(rng.choice(len(wf), min(args.n_traces, len(wf)), replace=False))
    logger.info("%d test traces sampled (seed %d)", len(idx), args.seed)

    acc = {cr: {"nrmse_full": [], "nrmse_p_win": [], "nrmse_s_win": [], "lag_ms": []}
           for cr in CRS}
    for n, i in enumerate(idx):
        x = np.asarray(wf[i], dtype=np.float32)[None]  # (1, 3, N)
        std = x.std()
        if std > 1e-8:
            x = x / std
        p, s = int(p_all[i]), int(s_all[i])
        for cr in CRS:
            xr = band_limited_reconstruct(x, cr)
            a = acc[cr]
            a["nrmse_full"].append(nrmse(xr, x))
            wp, ws = _window_nrmse(x, xr, p), _window_nrmse(x, xr, s)
            if wp is not None:
                a["nrmse_p_win"].append(wp)
                a["lag_ms"].append(onset_lag_samples(x[0, 2], xr[0, 2], p) * 1000.0 / FS)
            if ws is not None:
                a["nrmse_s_win"].append(ws)
        if (n + 1) % 1000 == 0:
            logger.info("  %d/%d", n + 1, len(idx))

    summary = json.load(open(args.eval_summary))["summary"]
    out: dict = {"meta": {"n_traces": int(len(idx)), "seed": args.seed,
                          "normalization": "per-trace std", "window_sec": 1.0},
                 "per_cr": {}}
    for cr in CRS:
        a = acc[cr]
        ev = summary.get(f"wavelet_cr{cr}", {})
        t01, tm = ev.get("tolerance_0.1s", {}), ev.get("timing", {})
        lags = np.asarray(a["lag_ms"])
        out["per_cr"][cr] = {
            "nrmse_full_mean": round(float(np.mean(a["nrmse_full"])), 4),
            "nrmse_full_median": round(float(np.median(a["nrmse_full"])), 4),
            "nrmse_p_win_median": round(float(np.median(a["nrmse_p_win"])), 4),
            "nrmse_s_win_median": round(float(np.median(a["nrmse_s_win"])), 4),
            "onset_lag_ms_median": round(float(np.median(lags)), 2),
            "onset_lag_ms_p95_abs": round(float(np.percentile(np.abs(lags), 95)), 2),
            "p_f1": t01.get("p_f1_mean"), "s_f1": t01.get("s_f1_mean"),
            "p_bias_ms": None if tm.get("p_residual_mean") is None else round(1000 * tm["p_residual_mean"], 1),
            "s_bias_ms": None if tm.get("s_residual_mean") is None else round(1000 * tm["s_residual_mean"], 1),
            "p_mae_ms": None if tm.get("p_mae_mean") is None else round(1000 * tm["p_mae_mean"], 1),
            "s_mae_ms": None if tm.get("s_mae_mean") is None else round(1000 * tm["s_mae_mean"], 1),
        }
    out["frequency_response_nrmse"] = {cr: frequency_response(cr) for cr in CRS}
    tf: dict = {}
    for cr in CRS:
        f, h = noise_transfer_function(cr)
        # -3 dB and -20 dB points of the power transfer function
        f3 = float(f[np.argmax(h < 0.5)]) if np.any(h < 0.5) else None
        f20 = float(f[np.argmax(h < 0.01)]) if np.any(h < 0.01) else None
        tf[cr] = {"nominal_cutoff_hz": fs_nominal(cr), "f_minus3dB_hz": f3, "f_minus20dB_hz": f20,
                  "power_ratio_at": {str(q): round(float(h[np.argmin(np.abs(f - q))]), 3)
                                     for q in (2, 5, 8, 10, 12.5, 15, 20, 25)}}
    out["noise_transfer_function"] = tf
    base = summary.get("l2_baseline", {})
    out["baseline"] = {"p_f1": base["tolerance_0.1s"]["p_f1_mean"], "s_f1": base["tolerance_0.1s"]["s_f1_mean"],
                       "p_bias_ms": round(1000 * base["timing"]["p_residual_mean"], 1),
                       "s_bias_ms": round(1000 * base["timing"]["s_residual_mean"], 1),
                       "p_mae_ms": round(1000 * base["timing"]["p_mae_mean"], 1),
                       "s_mae_ms": round(1000 * base["timing"]["s_mae_mean"], 1)}
    Path(args.output).write_text(json.dumps(out, indent=1))
    logger.info("saved %s", args.output)
    for cr, r in out["per_cr"].items():
        logger.info("CR=%-2s NRMSE full %.3f | P-win %.3f | S-win %.3f | lag med %.1f ms (p95 %.1f) | "
                    "P bias %s ms | S bias %s ms", cr, r["nrmse_full_mean"], r["nrmse_p_win_median"],
                    r["nrmse_s_win_median"], r["onset_lag_ms_median"], r["onset_lag_ms_p95_abs"],
                    r["p_bias_ms"], r["s_bias_ms"])


if __name__ == "__main__":
    main()
