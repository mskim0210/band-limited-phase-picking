"""Domain adaptation on SE-Korea 2023 Q4, evaluated against MANUAL picks.

Event-split (75/25) protocol on the cached windows
(outputs/results/manual2023_windows.npz from scripts/apply_manual2023.py):

  1. zero-shot   : STEAD checkpoints (baseline, CR=2/4/8) on held-out events
  2. fine-tune   : supervised by the manual P/S picks of the TRAIN events
  3. distillation: catalog-free -- wavelet CR=4 student trained to match the
                   zero-shot baseline teacher's soft output on the TRAIN
                   events (no picks used); manual picks used ONLY to evaluate

Split hygiene: events whose 60-s windows physically overlap (origin times
< 60 s apart) are clustered and assigned to the same split side, so no test
event's energy appears in any training window.

False-trigger rate (ft_per_hour_pooled): unmatched P and S peaks summed,
divided by the total duration of ALL evaluated windows; peaks on traces
lacking a manual pick of that phase are excluded as unlabelable.

Also emits, per config: a threshold sweep (P/S precision-recall curves,
thresholds 0.05-0.95 at tol 0.5 s) and a tolerance sweep (recall at
tol = 0.1/0.2/0.5 s at threshold 0.3).

Usage:
    python scripts/adapt_manual2023.py --epochs 15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from scipy.signal import find_peaks

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_manual2023 import CACHE, FS, N, DEV, encode_norm, load_model, predict_probs
from cs_pipeline.sensing_matrix import create_sensing_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

SIGMA = 10.0  # label Gaussian width in samples (0.1 s), as in STEAD training
THR_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
TOL_SWEEP = [0.1, 0.2, 0.5]


def make_labels(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    lab = np.zeros((len(p), 3, N), np.float32)
    x = np.arange(N, dtype=np.float32)
    for i in range(len(p)):
        if p[i] >= 0:
            lab[i, 1] = np.exp(-((x - p[i]) ** 2) / (2 * SIGMA ** 2))
        if s[i] >= 0:
            lab[i, 2] = np.exp(-((x - s[i]) ** 2) / (2 * SIGMA ** 2))
        lab[i, 0] = np.clip(1 - lab[i, 1] - lab[i, 2], 0, 1)
    return lab


def finetune(model, Xtr, Ltr, epochs, lr=1e-4, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([0.05, 0.40, 0.55]).to(DEV))
    model.train()
    idx = np.arange(len(Xtr))
    for _ in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            x = torch.from_numpy(Xtr[b]).to(DEV)
            y = torch.from_numpy(Ltr[b]).to(DEV)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    model.eval()


def distill(student, Xs, T, epochs, lr=1e-4, bs=64):
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    student.train()
    idx = np.arange(len(Xs))
    for _ in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            x = torch.from_numpy(Xs[b]).to(DEV)
            t = torch.from_numpy(T[b]).to(DEV)
            opt.zero_grad()
            logp = Fnn.log_softmax(student(x), dim=1)
            loss = -(t * logp).sum(1).mean()
            loss.backward()
            opt.step()
    student.eval()


def phase_counts(probs: np.ndarray, gt: np.ndarray, ch: int,
                 thr: float, tol_samp: int) -> dict:
    """TP/FP/FN + residuals for one phase on traces with a manual pick."""
    tp = fp = fn = 0
    res = []
    for i in range(len(probs)):
        if gt[i] < 0:
            continue
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
    return {"tp": tp, "fp": fp, "fn": fn, "res": res}


def phase_metrics(c: dict) -> dict:
    rec = c["tp"] / max(c["tp"] + c["fn"], 1)
    prec = c["tp"] / max(c["tp"] + c["fp"], 1)
    return {
        "n_picks": c["tp"] + c["fn"], "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        "recall": round(rec, 3), "precision": round(prec, 3),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-12), 3),
        "res_mean": round(float(np.mean(c["res"])), 3) if c["res"] else None,
        "res_mae": round(float(np.mean(np.abs(c["res"]))), 3) if c["res"] else None,
        "res_std": round(float(np.std(c["res"])), 3) if c["res"] else None,
    }


def full_eval(probs: np.ndarray, P: np.ndarray, S: np.ndarray,
              thr: float, tol_samp: int) -> dict:
    cp = phase_counts(probs, P, 1, thr, tol_samp)
    cs = phase_counts(probs, S, 2, thr, tol_samp)
    hours = len(probs) * (N / FS) / 3600.0  # ALL evaluated windows
    out = {"P": phase_metrics(cp), "S": phase_metrics(cs),
           "ft_per_hour_pooled": round((cp["fp"] + cs["fp"]) / hours, 2)}
    return out


def sweeps(probs: np.ndarray, P: np.ndarray, S: np.ndarray,
           tol_samp: int, thr_main: float) -> dict:
    thr_sweep = []
    for t in THR_SWEEP:
        e = full_eval(probs, P, S, t, tol_samp)
        thr_sweep.append({"thr": t,
                          "P": {k: e["P"][k] for k in ("recall", "precision", "f1")},
                          "S": {k: e["S"][k] for k in ("recall", "precision", "f1")},
                          "ft_per_hour_pooled": e["ft_per_hour_pooled"]})
    tol_sweep = []
    for tol in TOL_SWEEP:
        e = full_eval(probs, P, S, thr_main, int(tol * FS))
        tol_sweep.append({"tol_sec": tol,
                          "P_recall": e["P"]["recall"], "S_recall": e["S"]["recall"],
                          "P_precision": e["P"]["precision"],
                          "S_precision": e["S"]["precision"]})
    return {"threshold_sweep": thr_sweep, "tolerance_sweep": tol_sweep}


def cluster_split(EV: np.ndarray, ot_by_ev: dict, test_frac: float,
                  seed: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Split by event, clustering events with |dOT| < 60 s to the same side."""
    events = np.array(sorted(np.unique(EV).tolist()))
    ots = np.array([ot_by_ev[e] for e in events])
    order = np.argsort(ots)
    cluster_of = np.zeros(len(events), dtype=int)
    cid = 0
    for k, idx in enumerate(order):
        if k > 0 and (ots[idx] - ots[order[k - 1]]) < 60.0:
            cluster_of[idx] = cluster_of[order[k - 1]]
        else:
            cluster_of[idx] = cid
            cid += 1
    clusters = np.unique(cluster_of)
    rng = np.random.RandomState(seed)
    rng.shuffle(clusters)
    n_test_cl = int(len(clusters) * test_frac)
    test_cl = set(clusters[:n_test_cl].tolist())
    ev_is_test = {e: (cluster_of[i] in test_cl) for i, e in enumerate(events)}
    te = np.array([ev_is_test[e] for e in EV])
    n_test_ev = sum(ev_is_test.values())
    return ~te, te, len(events) - n_test_ev, n_test_ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--tol-sec", type=float, default=0.5)
    ap.add_argument("--output", default="outputs/results/manual2023_adapt.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(CACHE, allow_pickle=True)
    W, P, S, EV = d["windows"], d["p"], d["s"], d["ev"]
    logger.info("Loaded %d windows, %d events", len(W), len(np.unique(EV)))

    cat = pd.read_csv("outputs/results/manual2023_catalog.csv", parse_dates=["ot"])
    ot_by_ev = {e: g.ot.iloc[0].timestamp()
                for e, g in cat.groupby("event_id")}
    tr, te, n_tr_ev, n_te_ev = cluster_split(EV, ot_by_ev, args.test_frac, args.seed)
    logger.info("cluster-aware event split: %d train / %d test events | "
                "windows %d / %d", n_tr_ev, n_te_ev, tr.sum(), te.sum())

    tol_samp = int(args.tol_sec * FS)
    thr = args.threshold
    Lab = make_labels(P, S)
    mats = {2: create_sensing_matrix("wavelet", N=6000, CR=2),
            4: create_sensing_matrix("wavelet", N=6000, CR=4),
            8: create_sensing_matrix("wavelet", N=6000, CR=8)}
    Xb = encode_norm(W, None)
    X = {cr: encode_norm(W, m) for cr, m in mats.items()}

    out, sw = {}, {}

    def record(name, probs):
        out[name] = full_eval(probs, P[te], S[te], thr, tol_samp)
        sw[name] = sweeps(probs, P[te], S[te], tol_samp, thr)
        logger.info("%s done", name)

    # ---- zero-shot: baseline + CR=2/4/8 ----
    base = load_model("outputs/checkpoints/l2_baseline_seed42/best_model.pt")
    record("baseline_zeroshot", predict_probs(base, Xb[te]))
    for cr in (2, 4, 8):
        m = load_model(f"outputs/checkpoints/wavelet_cr{cr}_seed42/best_model.pt")
        record(f"cr{cr}_zeroshot", predict_probs(m, X[cr][te]))

    # ---- fine-tune: baseline + CR=4 ----
    finetune(base, Xb[tr], Lab[tr], args.epochs)
    record("baseline_finetune", predict_probs(base, Xb[te]))
    st = load_model("outputs/checkpoints/wavelet_cr4_seed42/best_model.pt")
    finetune(st, X[4][tr], Lab[tr], args.epochs)
    record("cr4_finetune", predict_probs(st, X[4][te]))

    # ---- CR=4: catalog-free distillation from zero-shot baseline teacher ----
    teacher = load_model("outputs/checkpoints/l2_baseline_seed42/best_model.pt")
    T = predict_probs(teacher, Xb[tr])  # soft labels, no picks used
    st2 = load_model("outputs/checkpoints/wavelet_cr4_seed42/best_model.pt")
    distill(st2, X[4][tr], T, args.epochs)
    record("cr4_distill", predict_probs(st2, X[4][te]))

    meta = {"n_train_windows": int(tr.sum()), "n_test_windows": int(te.sum()),
            "n_train_events": n_tr_ev, "n_test_events": n_te_ev,
            "epochs": args.epochs, "threshold": thr, "tol_sec": args.tol_sec,
            "seed": args.seed, "split": "event-level, <60 s clusters kept together",
            "ft_definition": "unmatched P+S peaks per hour over ALL evaluated "
                             "test windows (picked-phase traces only contribute peaks)",
            "reference": "manual analyst picks (2023 Q4 SE-Korea), held-out events"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": meta, "results": out, "sweeps": sw},
              open(args.output, "w"), indent=2)

    print(f"\n=== Adaptation vs MANUAL catalog (held-out {te.sum()} windows / "
          f"{n_te_ev} events, thr={thr}, tol={args.tol_sec}s) ===")
    print(f"{'config':20s} {'phase':5s} {'recall':>7s} {'prec':>7s} {'F1':>7s} "
          f"{'MAE':>6s} {'FT/h':>7s}")
    for k, r in out.items():
        for ph in ("P", "S"):
            e = r[ph]
            print(f"{k:20s} {ph:5s} {e['recall']:>7.3f} {e['precision']:>7.3f} "
                  f"{e['f1']:>7.3f} {(e['res_mae'] if e['res_mae'] is not None else 0):>5.3f}s "
                  f"{r['ft_per_hour_pooled']:>7.2f}")


if __name__ == "__main__":
    main()
