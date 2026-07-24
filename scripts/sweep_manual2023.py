"""Symmetric hyperparameter sweep for the SE-Korea CR=4 adaptation recipes.

Tests whether the fine-tune baseline is under-tuned by
giving BOTH recipes an identical tuning budget:

  fine-tune : lr {1e-4, 3e-5, 1e-5} x freeze {none, encoder} x
              class weights {stead (0.05/0.40/0.55), uniform}  -> 12 configs
  distill   : lr {1e-4, 3e-5} x temperature {1, 2}             -> 4 configs

Protocol per split seed (42/123/7):
  * outer split: cluster-aware 75/25 event split (same as adapt_manual2023)
  * inner split: the TRAIN events are further split 80/20 (cluster-aware);
    each config trains on the inner-train windows with early stopping on the
    inner-val mean F1 ((P-F1 + S-F1)/2 at thr 0.3, tol 0.5 s; max 30 epochs,
    patience 5, best-epoch weights kept)
  * the config with the best inner-val score is evaluated ONCE on the
    held-out test events

Also times every stage of the deployment workflow (teacher inference,
soft-label generation, encoding, student training) -> adaptation_cost.

Usage:
    python scripts/sweep_manual2023.py
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_manual2023 import CACHE, FS, N, DEV, encode_norm, load_model
from adapt_manual2023 import cluster_split, full_eval, make_labels
from cs_pipeline.sensing_matrix import create_sensing_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_EPOCHS = 30
PATIENCE = 5
BS = 64
FT_GRID = list(itertools.product([1e-4, 3e-5, 1e-5], ["none", "encoder"],
                                 ["stead", "uniform"]))
DI_GRID = list(itertools.product([1e-4, 3e-5], [1.0, 2.0]))


@torch.no_grad()
def logits_of(model, X, bs=256):
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).to(DEV)).cpu())
    return torch.cat(out, 0)


def probs_np(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, 1).numpy()


def val_score(model, Xv, Pv, Sv, thr=0.3, tol_samp=50) -> float:
    e = full_eval(probs_np(logits_of(model, Xv)), Pv, Sv, thr, tol_samp)
    return (e["P"]["f1"] + e["S"]["f1"]) / 2


def freeze_encoder(model) -> None:
    for name in ("stem", "enc1", "enc2", "enc3", "enc4"):
        for p in getattr(model, name).parameters():
            p.requires_grad = False


def train_early_stop(model, Xtr, target, loss_fn, lr, Xv, Pv, Sv):
    """Generic loop: loss_fn(logits, target_batch) with early stopping."""
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    idx = np.arange(len(Xtr))
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        np.random.shuffle(idx)
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            x = torch.from_numpy(Xtr[b]).to(DEV)
            t = torch.from_numpy(target[b]).to(DEV)
            opt.zero_grad()
            loss = loss_fn(model(x), t)
            loss.backward()
            opt.step()
        model.eval()
        sc = val_score(model, Xv, Pv, Sv)
        if sc > best:
            best, best_ep, bad = sc, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return best, best_ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--tol-sec", type=float, default=0.5)
    ap.add_argument("--output", default="outputs/results/manual2023_sweep.json")
    args = ap.parse_args()
    tol_samp = int(args.tol_sec * FS)
    thr = args.threshold

    d = np.load(CACHE, allow_pickle=True)
    W, P, S, EV = d["windows"], d["p"], d["s"], d["ev"]
    cat = pd.read_csv("outputs/results/manual2023_catalog.csv", parse_dates=["ot"])
    ot_by_ev = {e: g.ot.iloc[0].timestamp() for e, g in cat.groupby("event_id")}

    mat4 = create_sensing_matrix("wavelet", N=6000, CR=4)
    t0 = time.perf_counter()
    Xb = encode_norm(W, None)
    X4 = encode_norm(W, mat4)
    t_encode_all = time.perf_counter() - t0
    Lab = make_labels(P, S)

    results = {"ft": [], "distill": []}
    cost = {}

    for seed in (42, 123, 7):
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr, te, _, n_te_ev = cluster_split(EV, ot_by_ev, 0.25, seed)
        # inner 80/20 split of the outer-train events (cluster-aware)
        tri, vai, _, _ = cluster_split(EV[tr], ot_by_ev, 0.20, seed + 1000)
        tr_idx = np.where(tr)[0]
        itr = tr_idx[tri]        # inner-train window indices
        iva = tr_idx[vai]        # inner-val window indices
        logger.info("seed %d | windows inner-train %d / inner-val %d / test %d",
                    seed, len(itr), len(iva), te.sum())

        # ---- teacher (needed by distill grid): time the stages once ----
        teacher = load_model("outputs/checkpoints/l2_baseline_seed42/best_model.pt")
        t0 = time.perf_counter()
        T_logits = logits_of(teacher, Xb[itr])
        t_teacher = time.perf_counter() - t0

        # ---- fine-tune grid ----
        best_ft = None
        for lr, freeze, wname in FT_GRID:
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = load_model("outputs/checkpoints/wavelet_cr4_seed42/best_model.pt")
            if freeze == "encoder":
                freeze_encoder(m)
            w = torch.tensor([0.05, 0.40, 0.55] if wname == "stead"
                             else [1 / 3, 1 / 3, 1 / 3]).to(DEV)
            crit = nn.CrossEntropyLoss(weight=w)
            sc, ep = train_early_stop(m, X4[itr], Lab[itr],
                                      lambda o, t: crit(o, t), lr,
                                      X4[iva], P[iva], S[iva])
            if best_ft is None or sc > best_ft["val"]:
                best_ft = {"val": sc, "lr": lr, "freeze": freeze,
                           "weights": wname, "epochs": ep, "model": m}
            logger.info("ft lr=%g freeze=%s w=%s -> val %.3f (ep %d)",
                        lr, freeze, wname, sc, ep)
        e = full_eval(probs_np(logits_of(best_ft["model"], X4[te])),
                      P[te], S[te], thr, tol_samp)
        results["ft"].append({"seed": seed, "chosen": {k: best_ft[k] for k in
                              ("lr", "freeze", "weights", "epochs", "val")},
                              "test": e})
        logger.info("seed %d FT chosen %s -> test P-F1 %.3f S-F1 %.3f",
                    seed, results["ft"][-1]["chosen"], e["P"]["f1"], e["S"]["f1"])

        # ---- distill grid ----
        best_di, t_train_best = None, None
        for lr, temp in DI_GRID:
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = load_model("outputs/checkpoints/wavelet_cr4_seed42/best_model.pt")
            Tp = torch.softmax(T_logits / temp, 1).numpy()

            def soft_ce(o, t):
                return -(t * Fnn.log_softmax(o, dim=1)).sum(1).mean()
            t0 = time.perf_counter()
            sc, ep = train_early_stop(m, X4[itr], Tp, soft_ce, lr,
                                      X4[iva], P[iva], S[iva])
            t_train = time.perf_counter() - t0
            if best_di is None or sc > best_di["val"]:
                best_di = {"val": sc, "lr": lr, "temperature": temp,
                           "epochs": ep, "model": m}
                t_train_best = t_train
            logger.info("distill lr=%g T=%g -> val %.3f (ep %d)", lr, temp, sc, ep)
        e = full_eval(probs_np(logits_of(best_di["model"], X4[te])),
                      P[te], S[te], thr, tol_samp)
        results["distill"].append({"seed": seed, "chosen": {k: best_di[k] for k in
                                   ("lr", "temperature", "epochs", "val")},
                                   "test": e})
        logger.info("seed %d DI chosen %s -> test P-F1 %.3f S-F1 %.3f",
                    seed, results["distill"][-1]["chosen"], e["P"]["f1"], e["S"]["f1"])

        if seed == 42:
            cost = {"n_adapt_windows": int(len(itr)),
                    "teacher_soft_labels_sec": round(t_teacher, 2),
                    "student_distill_train_sec_best": round(t_train_best, 1),
                    "encode_all_1965_windows_sec": round(t_encode_all, 2)}

    def agg(rows, ph, f):
        v = [r["test"][ph][f] for r in rows]
        return round(float(np.mean(v)), 3), round(float(np.std(v)), 3)

    summary = {}
    for k in ("ft", "distill"):
        summary[k] = {ph: {f: agg(results[k], ph, f)
                           for f in ("recall", "precision", "f1", "res_mae")}
                      for ph in ("P", "S")}
        summary[k]["ft_per_hour_pooled"] = round(float(np.mean(
            [r["test"]["ft_per_hour_pooled"] for r in results[k]])), 1)
        summary[k]["chosen_configs"] = [r["chosen"] for r in results[k]]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"ft_grid": [str(c) for c in FT_GRID],
                        "distill_grid": [str(c) for c in DI_GRID],
                        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                        "selection": "inner 80/20 cluster-aware event split, "
                                     "mean of P/S F1 at thr 0.3",
                        "threshold": thr, "tol_sec": args.tol_sec},
               "adaptation_cost": cost,
               "per_seed": results, "summary": summary},
              open(args.output, "w"), indent=2)

    print("\n=== Tuned recipes (held-out test, mean±std over 3 splits) ===")
    for k in ("ft", "distill"):
        s = summary[k]
        print(f"{k:8s} P: rec {s['P']['recall']} prec {s['P']['precision']} "
              f"F1 {s['P']['f1']} | S: F1 {s['S']['f1']} | FT/h {s['ft_per_hour_pooled']}")
        print(f"         chosen: {s['chosen_configs']}")
    print("cost:", cost)


if __name__ == "__main__":
    main()
