"""Station hold-out: does catalog-free distillation generalize to an unseen
station?

Five-fold over borehole sites (TEBD/TEDG/TEJH/TEJN/TENG; two sensor depths
per site). For each fold, the CR=4 student is distilled from the zero-shot
full-band teacher using ONLY windows from the other four sites, then
evaluated on the held-out site's windows (against manual picks). Events are
shared across stations by design---the question probed is cross-STATION
generalization of the recovery, matching the deployment scenario where a
network gains a new station.

Default recipe (Adam 1e-4, soft-CE, T=1, 15 epochs) for comparability with
Table XI's distilled row.

Usage:
    python scripts/station_holdout_manual2023.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_manual2023 import CACHE, FS, DEV, encode_norm, load_model, predict_probs
from adapt_manual2023 import distill, full_eval
from cs_pipeline.sensing_matrix import create_sensing_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

SITES = ["TEBD", "TEDG", "TEJH", "TEJN", "TENG"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--tol-sec", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output",
                    default="outputs/results/manual2023_station_holdout.json")
    args = ap.parse_args()
    tol_samp = int(args.tol_sec * FS)

    d = np.load(CACHE, allow_pickle=True)
    W, P, S, STA = d["windows"], d["p"], d["s"], d["sta"]
    site = np.array([s[:4] for s in STA])

    mat4 = create_sensing_matrix("wavelet", N=6000, CR=4)
    Xb = encode_norm(W, None)
    X4 = encode_norm(W, mat4)

    teacher = load_model("outputs/checkpoints/l2_baseline_seed42/best_model.pt")

    folds = []
    for hold in SITES:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        te = site == hold
        tr = ~te
        T = predict_probs(teacher, Xb[tr])
        st = load_model("outputs/checkpoints/wavelet_cr4_seed42/best_model.pt")
        before = full_eval(predict_probs(st, X4[te]), P[te], S[te],
                           args.threshold, tol_samp)
        distill(st, X4[tr], T, args.epochs)
        after = full_eval(predict_probs(st, X4[te]), P[te], S[te],
                          args.threshold, tol_samp)
        folds.append({"held_out_site": hold, "n_test_windows": int(te.sum()),
                      "before": before, "after": after})
        logger.info("%s (n=%d): P-F1 %.3f -> %.3f | S-F1 %.3f -> %.3f",
                    hold, te.sum(), before["P"]["f1"], after["P"]["f1"],
                    before["S"]["f1"], after["S"]["f1"])

    def agg(key, ph, f):
        v = [x[key][ph][f] for x in folds]
        return round(float(np.mean(v)), 3), round(float(np.std(v)), 3)

    summary = {key: {ph: {f: agg(key, ph, f) for f in ("recall", "precision", "f1")}
                     for ph in ("P", "S")} for key in ("before", "after")}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"epochs": args.epochs, "threshold": args.threshold,
                        "tol_sec": args.tol_sec,
                        "protocol": "5-fold site hold-out; distill on 4 sites, "
                                    "evaluate on the unseen site"},
               "folds": folds, "summary": summary}, open(args.output, "w"),
              indent=2)

    print("\n=== Station hold-out (5-fold, unseen site) ===")
    for key in ("before", "after"):
        s = summary[key]
        print(f"{key:7s} P-F1 {s['P']['f1']} | S-F1 {s['S']['f1']} "
              f"(rec {s['P']['recall']}, prec {s['P']['precision']})")


if __name__ == "__main__":
    main()
