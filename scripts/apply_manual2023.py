"""Zero-shot evaluation against the SE-Korea 2023 Q4 MANUAL pick catalog.

Reference = analyst manual picks (outputs/results/manual2023_catalog.csv,
built by scripts/manual2023_catalog.py). Each case is one (event, station)
60-s trace (OT-10 s .. OT+50 s, 100 Hz, pre-cut mseed).

For each model in {baseline, wavelet CR=2/4/8} reports, per phase (P and S):
  recall     manual picks matched by a predicted peak within tol
  precision  matched peaks / all predicted peaks on traces WITH a manual pick
             of that phase (unpicked traces are excluded as ambiguous)
  F1, signed mean residual, MAE, std
  false triggers per hour (unmatched peaks / total picked-trace duration)

Component order matches STEAD training: [E, N, Z] = [HH2, HH1, HHZ].

Usage:
    python scripts/apply_manual2023.py
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from obspy import read
from scipy.signal import find_peaks

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs_pipeline.cs_aware_model import CSPhaseNet
from cs_pipeline.sensing_matrix import create_sensing_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

FS = 100.0
N = 6000
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = Path("outputs/results/manual2023_windows.npz")


def build_windows(cat: pd.DataFrame):
    """Read all (event, station) traces -> (n,3,6000) + pick samples."""
    W, P, S, EV, CATG, STA = [], [], [], [], [], []
    n_miss = 0
    for i, r in cat.iterrows():
        arr = np.zeros((3, N), np.float32)
        ok = True
        for ci, comp in enumerate(("HH2", "HH1", "HHZ")):  # [E,N,Z]
            g = glob.glob(f"{r.path}/KG.{r.station}.*.{comp}.mseed")
            if not g:
                ok = False
                break
            d = read(g[0])[0]
            d.detrend("linear")
            d.detrend("demean")
            x = d.data.astype(np.float32)
            if len(x) < N - 5:
                ok = False
                break
            arr[ci, :min(len(x), N)] = x[:N]
        if not ok:
            n_miss += 1
            continue
        W.append(arr)
        P.append(int(round(r.p_sec * FS)) if pd.notna(r.p_sec) else -1)
        S.append(int(round(r.s_sec * FS)) if pd.notna(r.s_sec) else -1)
        EV.append(r.event_id)
        CATG.append(r.comp)
        STA.append(r.station)
        if (i + 1) % 400 == 0:
            logger.info("traces %d/%d", i + 1, len(cat))
    W = np.stack(W)
    logger.info("Windows: %d (missing waveforms: %d)", len(W), n_miss)
    np.savez(CACHE, windows=W, p=np.array(P), s=np.array(S),
             ev=np.array(EV), catg=np.array(CATG), sta=np.array(STA))
    return W, np.array(P), np.array(S), np.array(EV), np.array(CATG)


def load_model(ckpt: str) -> CSPhaseNet:
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = CSPhaseNet(M=d["config"]["M"], N=6000,
                   base_channels=d["config"].get("base_channels", 16))
    m.load_state_dict(d["model_state_dict"])
    return m.to(DEV).eval()


def encode_norm(W: np.ndarray, mat) -> np.ndarray:
    x = W / (W.std(axis=(1, 2), keepdims=True) + 1e-8)
    if mat is not None:
        x = mat.encode(torch.from_numpy(x.astype(np.float32))).numpy()
        x = x / (x.std(axis=(1, 2), keepdims=True) + 1e-8)
    return x.astype(np.float32)


@torch.no_grad()
def predict_probs(model, X: np.ndarray, bs: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEV)
        out.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out, 0)  # (n,3,6000)


def eval_phase(probs: np.ndarray, gt: np.ndarray, ch: int,
               thr: float, tol_samp: int) -> dict:
    """Peak-based P/S metrics on traces with a manual pick of this phase."""
    tp = fp = fn = 0
    res = []
    n_traces = 0
    for i in range(len(probs)):
        if gt[i] < 0:
            continue
        n_traces += 1
        pks = find_peaks(probs[i, ch], height=thr, distance=100)[0]
        if len(pks) == 0:
            fn += 1
            continue
        d = np.abs(pks - gt[i])
        j = int(np.argmin(d))
        if d[j] <= tol_samp:
            tp += 1
            res.append((pks[j] - gt[i]) / FS)
            fp += len(pks) - 1
        else:
            fn += 1
            fp += len(pks)
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    hours = n_traces * (N / FS) / 3600.0
    return {
        "n_picks": n_traces, "tp": tp, "fp": fp, "fn": fn,
        "recall": round(rec, 3), "precision": round(prec, 3),
        "f1": round(f1, 3),
        "res_mean": round(float(np.mean(res)), 3) if res else None,
        "res_mae": round(float(np.mean(np.abs(res))), 3) if res else None,
        "res_std": round(float(np.std(res)), 3) if res else None,
        "false_per_hour": round(fp / hours, 2) if hours > 0 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--tol-sec", type=float, default=0.5)
    ap.add_argument("--output", default="outputs/results/manual2023_apply.json")
    args = ap.parse_args()

    cat = pd.read_csv("outputs/results/manual2023_catalog.csv")
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        W, P, S, EV, CATG = d["windows"], d["p"], d["s"], d["ev"], d["catg"]
        logger.info("Loaded cache: %d windows", len(W))
    else:
        W, P, S, EV, CATG = build_windows(cat)

    models = {
        "baseline": ("outputs/checkpoints/l2_baseline_seed42/best_model.pt", None),
        "wavelet_cr2": ("outputs/checkpoints/wavelet_cr2_seed42/best_model.pt",
                        create_sensing_matrix("wavelet", N=6000, CR=2)),
        "wavelet_cr4": ("outputs/checkpoints/wavelet_cr4_seed42/best_model.pt",
                        create_sensing_matrix("wavelet", N=6000, CR=4)),
        "wavelet_cr8": ("outputs/checkpoints/wavelet_cr8_seed42/best_model.pt",
                        create_sensing_matrix("wavelet", N=6000, CR=8)),
    }

    tol_samp = int(args.tol_sec * FS)
    out = {}
    for name, (ck, mat) in models.items():
        X = encode_norm(W, mat)
        probs = predict_probs(load_model(ck), X)
        r = {"P": eval_phase(probs, P, 1, args.threshold, tol_samp),
             "S": eval_phase(probs, S, 2, args.threshold, tol_samp)}
        # per-category P recall (natural / blast / extra)
        r["P_recall_by_category"] = {
            c: eval_phase(probs[CATG == c], P[CATG == c], 1,
                          args.threshold, tol_samp)["recall"]
            for c in np.unique(CATG)}
        out[name] = r
        logger.info("%s done", name)

    meta = {"n_traces": int(len(W)), "n_events": int(len(np.unique(EV))),
            "threshold": args.threshold, "tol_sec": args.tol_sec,
            "reference": "manual analyst picks (2023 Q4 SE-Korea)"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": meta, "results": out}, open(args.output, "w"), indent=2)

    print(f"\n=== Zero-shot vs MANUAL catalog (n_traces={len(W)}, "
          f"P picks={(P >= 0).sum()}, S picks={(S >= 0).sum()}, "
          f"thr={args.threshold}, tol={args.tol_sec}s) ===")
    hdr = (f"{'model':12s} {'phase':5s} {'recall':>7s} {'prec':>7s} {'F1':>7s} "
           f"{'mean':>7s} {'MAE':>6s} {'std':>6s} {'FT/hr':>7s}")
    print(hdr)
    for m, r in out.items():
        for ph in ("P", "S"):
            e = r[ph]
            print(f"{m:12s} {ph:5s} {e['recall']:>7.3f} {e['precision']:>7.3f} "
                  f"{e['f1']:>7.3f} {(e['res_mean'] if e['res_mean'] is not None else 0):>6.3f}s "
                  f"{(e['res_mae'] if e['res_mae'] is not None else 0):>5.3f}s "
                  f"{(e['res_std'] if e['res_std'] is not None else 0):>5.3f}s "
                  f"{(e['false_per_hour'] if e['false_per_hour'] is not None else 0):>7.2f}")


if __name__ == "__main__":
    main()
