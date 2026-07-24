"""Energy retained by the band-truncation encoders (Eq. 2).

For each compression ratio, decomposes test traces with the same wavelet
settings as the encoder (db4, level 6, symmetric extension) and reports the
fraction of coefficient energy carried by the retained bands. For DCT, the
retained energy is that of the first M low-frequency coefficients. Values
are computed per component and aggregated over all valid components of the
sampled traces.

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
DCT_M = {4: 1506}


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
    dct_ret = {cr: [] for cr in DCT_M}
    for n, i in enumerate(idx):
        tr = np.asarray(wf[i], dtype=np.float64)
        for c in range(tr.shape[0]):
            x = tr[c]
            if not np.isfinite(x).all() or (x ** 2).sum() < 1e-30:
                continue
            coeffs = pywt.wavedec(x, "db4", level=6, mode="symmetric")
            sq = [(cc ** 2).sum() for cc in coeffs]
            tot = sum(sq)
            for cr, k in KEEP.items():
                band[cr].append(sum(sq[:k]) / tot * 100.0)
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
           "dct_first_M": {f"CR{cr}": stats(v) for cr, v in dct_ret.items()},
           "meta": {"wavelet": "db4", "level": 6, "mode": "symmetric",
                    "split": args.split, "n_traces": int(len(idx)),
                    "seed": args.seed}}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    logger.info("saved %s", path)
    for cr in sorted(KEEP):
        s = out["wavelet_band"][f"CR{cr}"]
        logger.info("wavelet CR=%d: mean %.2f%%  median %.2f%%",
                    cr, s["mean"], s["median"])
    for cr in sorted(DCT_M):
        s = out["dct_first_M"][f"CR{cr}"]
        logger.info("DCT CR=%d: mean %.2f%%", cr, s["mean"])


if __name__ == "__main__":
    main()
