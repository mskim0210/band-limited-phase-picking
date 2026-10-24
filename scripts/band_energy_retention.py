"""Energy retained by the band-truncation encoders (Eq. 2).

For each compression ratio, decomposes test traces with the same wavelet
settings as the encoder (db4, level 6, symmetric extension) and reports the
fraction of coefficient energy carried by the retained bands (E_r). For the
primary setting (CR = 4) it also reports the time-domain energy ratio
||x~||^2 / ||x||^2 of the actual encoder output (partial inverse transform
trimmed to M samples) and the per-component difference between the two, so
that the manuscript can state how closely the coefficient fraction tracks
the waveform energy instead of assuming equality (the symmetric boundary
extension makes the decomposition only approximately orthonormal). For DCT,
the retained energy is that of the first M low-frequency coefficients.
Values are computed per component and aggregated over all valid components
of the sampled traces.

Usage:
    python scripts/band_energy_retention.py --config configs/stead_experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pywt
from omegaconf import OmegaConf
from scipy.fft import dct as scipy_dct

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Retained coefficient arrays [cA_L, cD_L, ...] per nominal CR (db4, L=6)
KEEP = {2: 6, 4: 5, 8: 4, 16: 3}
DCT_M = {4: 1500}  # coefficients the DCT encoder keeps (N // CR); its output is zero-padded to 1506
# Encoder output length per nominal CR (symmetric extension, N = 6000)
M_OUT = {2: 3004, 4: 1506, 8: 756, 16: 382}
WAVELET, LEVEL, MODE = "db4", 6, "symmetric"


def component_retention(x: np.ndarray, keep: int, m: int,
                        wavelet: str = WAVELET, level: int = LEVEL,
                        mode: str = MODE) -> tuple[float, float]:
    """Return (coefficient-energy fraction, time-domain energy ratio).

    Args:
        x: one component of a trace (float64, length N).
        keep: number of leading coefficient arrays retained ([cA_L, cD_L, ...]).
        m: encoder output length the partial reconstruction is trimmed to.

    Returns:
        ``coef``: sum of squared retained coefficients over the sum over all
        coefficients (Eq. 2).  ``tdom``: ||x~||^2 / ||x||^2 where x~ is the
        partial inverse transform of the retained arrays trimmed to ``m``,
        i.e. exactly what the wavelet encoder emits.
    """
    coeffs = pywt.wavedec(x, wavelet, level=level, mode=mode)
    sq = [float((c ** 2).sum()) for c in coeffs]
    coef = sum(sq[:keep]) / sum(sq)
    xt = pywt.waverec(coeffs[:keep], wavelet, mode=mode)[:m]
    tdom = float((xt ** 2).sum()) / float((x ** 2).sum())
    return coef, tdom


def diff_stats(coef: np.ndarray, tdom: np.ndarray, within_pts: float = 0.3) -> dict[str, float]:
    """Summarize (tdom - coef) in percentage points over components."""
    d = (np.asarray(tdom, dtype=float) - np.asarray(coef, dtype=float)) * 100.0
    return {"mean_pts": round(float(d.mean()), 3),
            "median_pts": round(float(np.median(d)), 3),
            "p5_pts": round(float(np.percentile(d, 5)), 3),
            "p95_pts": round(float(np.percentile(d, 95)), 3),
            "max_abs_pts": round(float(np.abs(d).max()), 3),
            "within_pts": within_pts,
            "within_frac": round(float((np.abs(d) <= within_pts).mean()), 3),
            "n_components": int(len(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stead_experiment.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-traces", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output",
                    default="outputs/results/band_energy_retention.json")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    wf = np.load(Path(cfg.data.memmap_dir) / f"{args.split}_waveforms.npy",
                 mmap_mode="r")
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(wf), min(args.n_traces, len(wf)), replace=False)

    band = {cr: [] for cr in KEEP}
    tdom = {cr: [] for cr in KEEP}
    dct_ret = {cr: [] for cr in DCT_M}
    for n, i in enumerate(idx):
        tr = np.asarray(wf[i], dtype=np.float64)
        for c in range(tr.shape[0]):
            x = tr[c]
            if not np.isfinite(x).all() or (x ** 2).sum() < 1e-30:
                continue
            for cr, k in KEEP.items():
                cf, td = component_retention(x, k, M_OUT[cr])
                band[cr].append(cf * 100.0)
                tdom[cr].append(td * 100.0)
            d = scipy_dct(x, type=2, norm="ortho") ** 2
            dtot = d.sum()
            for cr, m in DCT_M.items():
                dct_ret[cr].append(d[:m].sum() / dtot * 100.0)
        if (n + 1) % 2500 == 0:
            logger.info("%d/%d traces", n + 1, len(idx))

    def stats(v):
        a = np.asarray(v)
        return {"mean": round(float(a.mean()), 2),
                "median": round(float(np.median(a)), 2),
                "std": round(float(a.std()), 2),
                "n_components": int(len(a))}

    out = {"wavelet_band": {f"CR{cr}": stats(v) for cr, v in band.items()},
           "time_domain": {f"CR{cr}": stats(v) for cr, v in tdom.items()},
           "coef_vs_time_domain": {
               f"CR{cr}": diff_stats(np.asarray(band[cr]) / 100.0, np.asarray(tdom[cr]) / 100.0)
               for cr in KEEP},
           "dct_first_M": {f"CR{cr}": stats(v) for cr, v in dct_ret.items()},
           "meta": {"wavelet": WAVELET, "level": LEVEL, "mode": MODE,
                    "m_out": M_OUT, "split": args.split,
                    "n_traces": int(len(idx)), "seed": args.seed}}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    logger.info("saved %s", path)
    for cr in sorted(KEEP):
        s, t, dd = (out["wavelet_band"][f"CR{cr}"], out["time_domain"][f"CR{cr}"],
                    out["coef_vs_time_domain"][f"CR{cr}"])
        logger.info("wavelet CR=%d: coef mean %.2f%% median %.2f%% | time-domain mean %.2f%% "
                    "median %.2f%% | diff mean %+.3f pts, median %+.3f, %.1f%% within %.1f pts",
                    cr, s["mean"], s["median"], t["mean"], t["median"], dd["mean_pts"],
                    dd["median_pts"], 100 * dd["within_frac"], dd["within_pts"])
    for cr in sorted(DCT_M):
        s = out["dct_first_M"][f"CR{cr}"]
        logger.info("DCT CR=%d: mean %.2f%%", cr, s["mean"])


if __name__ == "__main__":
    main()
