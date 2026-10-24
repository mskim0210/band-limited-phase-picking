"""Share of P-onset (and pre-P noise) energy above a cutoff, per trace.

One definition for every dataset so the STEAD and Korean-borehole numbers
quoted in Sections 3.3, 3.7 and 4 are comparable: vertical component,
window [P, P + win_sec] (noise: [P - win_sec, P]), Welch PSD
(nperseg = 128, noverlap = 64), fraction of PSD power above the cutoff.
Cutoffs are the nominal band limits of CR = 2, 4, 8 (25, 12.5, 6.25 Hz).

Inputs:
    STEAD test memmap  (memmap_dir/test_waveforms.npy, test_arrivals.npz)
    Korean windows     (outputs/results/manual2023_windows.npz, built by
                        scripts/apply_manual2023.py; p = -1 when no P pick)

Usage:
    python scripts/onset_spectral_fraction.py [--n-stead 0] \
        --output outputs/results/onset_spectral_fraction.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

FS = 100
CUTOFFS_HZ = (25.0, 12.5, 6.25)
WINDOWS_SEC = (3.0, 1.0)


def fraction_above(z: np.ndarray, p: int, fs: int, win_sec: float, fc_hz: float,
                   nperseg: int = 128, before: bool = False) -> float:
    """Fraction of Welch PSD power above ``fc_hz`` in the onset window.

    Args:
        z: One-component trace.
        p: P-onset sample; the window is ``[p, p + win)`` or, with
            ``before=True``, the noise window ``[p - win, p)``.
        fs: Sampling rate (Hz).
        win_sec: Window length (s).
        fc_hz: Cutoff frequency (Hz).
        nperseg: Welch segment length (capped at the window length).
        before: Use the pre-onset noise window instead.

    Returns:
        Fraction in [0, 1], or NaN when the window leaves the trace or has
        no power.
    """
    win = int(round(win_sec * fs))
    lo, hi = (p - win, p) if before else (p, p + win)
    if p < 0 or lo < 0 or hi > len(z):
        return float("nan")
    x = np.asarray(z[lo:hi], dtype=np.float64)
    seg = min(nperseg, win)
    f, pxx = welch(x, fs=fs, nperseg=seg, noverlap=seg // 2)
    tot = float(pxx.sum())
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")
    return float(pxx[f > fc_hz].sum() / tot)


def summarize_fractions(fr: np.ndarray) -> dict:
    """n, median, mean and share of traces with the majority above the cutoff."""
    v = np.asarray(fr, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0, "median": float("nan"), "mean": float("nan"), "majority_above": float("nan")}
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
            "majority_above": float((v > 0.5).mean())}


def _psd(z: np.ndarray, p: int, fs: int, win_sec: float, before: bool,
         nperseg: int = 128) -> tuple[np.ndarray, np.ndarray] | None:
    win = int(round(win_sec * fs))
    lo, hi = (p - win, p) if before else (p, p + win)
    if p < 0 or lo < 0 or hi > len(z):
        return None
    seg = min(nperseg, win)
    f, pxx = welch(np.asarray(z[lo:hi], dtype=np.float64), fs=fs, nperseg=seg, noverlap=seg // 2)
    tot = float(pxx.sum())
    return (f, pxx / tot) if np.isfinite(tot) and tot > 0 else None


def dataset_fractions(W: np.ndarray, p: np.ndarray, zch: int, idx: np.ndarray,
                      label: str) -> dict:
    """Per-cutoff / per-window summaries for the traces in ``idx``."""
    out: dict = {"n_traces_with_p": int((p[idx] >= 0).sum())}
    for win_sec in WINDOWS_SEC:
        fr = {(fc, side): [] for fc in CUTOFFS_HZ for side in ("p_onset", "pre_p_noise")}
        for i in idx:
            z = W[i, zch]
            for side, before in (("p_onset", False), ("pre_p_noise", True)):
                r = _psd(z, int(p[i]), FS, win_sec, before)
                for fc in CUTOFFS_HZ:
                    fr[(fc, side)].append(float("nan") if r is None else float(r[1][r[0] > fc].sum()))
        for fc in CUTOFFS_HZ:
            key = f"win{win_sec:g}s_above{fc:g}Hz"
            out[key] = {side: summarize_fractions(np.array(fr[(fc, side)]))
                        for side in ("p_onset", "pre_p_noise")}
            logger.info("%s %s: P-onset median %.3f (majority %.3f, n=%d) | noise median %.3f",
                        label, key, out[key]["p_onset"]["median"],
                        out[key]["p_onset"]["majority_above"], out[key]["p_onset"]["n"],
                        out[key]["pre_p_noise"]["median"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stead_experiment.yaml")
    ap.add_argument("--memmap-dir", default=None, help="default: data.memmap_dir of --config")
    ap.add_argument("--korean", default="outputs/results/manual2023_windows.npz")
    ap.add_argument("--n-stead", type=int, default=0,
                    help="random subset of STEAD test traces (0 = all traces with a P pick)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="outputs/results/onset_spectral_fraction.json")
    args = ap.parse_args()

    result = {"definition": "Z component; onset window [P, P+win], noise window [P-win, P]; "
                            "Welch nperseg=128 noverlap=64; fraction of PSD power above cutoff",
              "fs_hz": FS, "cutoffs_hz": list(CUTOFFS_HZ), "windows_sec": list(WINDOWS_SEC)}

    mem = Path(args.memmap_dir or OmegaConf.load(args.config).data.memmap_dir)
    W = np.load(mem / "test_waveforms.npy", mmap_mode="r")
    arr = np.load(mem / "test_arrivals.npz")
    p = arr["p_samples"].astype(int)
    idx = np.where(p >= 0)[0]
    if args.n_stead and args.n_stead < len(idx):
        idx = np.sort(np.random.RandomState(args.seed).choice(idx, args.n_stead, replace=False))
    logger.info("STEAD test: %d traces", len(idx))
    result["stead_test"] = dataset_fractions(W, p, zch=2, idx=idx, label="STEAD")

    k = np.load(args.korean, allow_pickle=True)
    pk = k["p"].astype(int)
    kidx = np.where(pk >= 0)[0]
    logger.info("Korean borehole: %d traces with P", len(kidx))
    result["korean_borehole"] = dataset_fractions(k["windows"], pk, zch=2, idx=kidx, label="Korea")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()
