"""Fully label-free distillation: fixed recipe, no labeled validation anywhere.

Recipe fixed a priori (lr = 1e-4, temperature = 1, full 30-epoch budget,
batch 64) and applied to ALL training-side windows of each event split.
Manual picks are used only for the final held-out evaluation, never for
model selection or stopping.

Usage:
    python scripts/distill_fixed_recipe.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_manual2023 import CACHE, FS, DEV, encode_norm, load_model
from adapt_manual2023 import cluster_split, full_eval
from sweep_manual2023 import logits_of, probs_np
from cs_pipeline.sensing_matrix import create_sensing_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

LR = 1e-4
TEMP = 1.0
EPOCHS = 30
BS = 64


def distill_fixed(model, Xtr: np.ndarray, target: np.ndarray) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    idx = np.arange(len(Xtr))
    for ep in range(1, EPOCHS + 1):
        model.train()
        np.random.shuffle(idx)
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            x = torch.from_numpy(Xtr[b]).to(DEV)
            t = torch.from_numpy(target[b]).to(DEV)
            opt.zero_grad()
            loss = -(t * Fnn.log_softmax(model(x), dim=1)).sum(1).mean()
            loss.backward()
            opt.step()
        if ep % 5 == 0:
            logger.info("  epoch %d/%d done", ep, EPOCHS)
    model.eval()


def main() -> None:
    thr, tol_samp = 0.3, int(0.5 * FS)
    d = np.load(CACHE, allow_pickle=True)
    W, P, S, EV = d["windows"], d["p"], d["s"], d["ev"]
    cat = pd.read_csv("outputs/results/manual2023_catalog.csv", parse_dates=["ot"])
    ot_by_ev = {e: g.ot.iloc[0].timestamp() for e, g in cat.groupby("event_id")}

    mat4 = create_sensing_matrix("wavelet", N=6000, CR=4)
    Xb = encode_norm(W, None)
    X4 = encode_norm(W, mat4)

    out = {"meta": {"recipe": f"lr={LR}, T={TEMP}, fixed {EPOCHS} epochs, "
                              "all training-side windows, no labeled validation",
                    "threshold": thr, "tol_sec": 0.5},
           "per_seed": []}

    for seed in (42, 123, 7):
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr, te, _, _ = cluster_split(EV, ot_by_ev, 0.25, seed)
        tr_idx = np.where(tr)[0]
        logger.info("seed %d | adapt windows %d | test windows %d",
                    seed, len(tr_idx), int(te.sum()))

        teacher = load_model("outputs/checkpoints/l2_baseline_seed42/best_model.pt")
        Tp = torch.softmax(logits_of(teacher, Xb[tr_idx]) / TEMP, 1).numpy()

        student = load_model("outputs/checkpoints/wavelet_cr4_seed42/best_model.pt")
        t0 = time.perf_counter()
        distill_fixed(student, X4[tr_idx], Tp)
        t_train = time.perf_counter() - t0

        e = full_eval(probs_np(logits_of(student, X4[te])), P[te], S[te],
                      thr, tol_samp)
        logger.info("seed %d -> test P-F1 %.3f  S-F1 %.3f  (train %.1fs)",
                    seed, e["P"]["f1"], e["S"]["f1"], t_train)
        out["per_seed"].append({"seed": seed, "train_sec": round(t_train, 1),
                                "test": e})

    def agg(phase, key):
        v = [r["test"][phase][key] for r in out["per_seed"]]
        return round(float(np.mean(v)), 3), round(float(np.std(v)), 3)

    out["summary"] = {ph: {k: agg(ph, k) for k in
                           ("f1", "recall", "precision", "res_mae")}
                      for ph in ("P", "S")}
    out["summary"]["ft_per_hour_pooled"] = round(float(np.mean(
        [r["test"]["ft_per_hour_pooled"] for r in out["per_seed"]])), 1)

    path = Path("outputs/results/manual2023_distill_fixed.json")
    path.write_text(json.dumps(out, indent=1))
    logger.info("saved %s", path)
    logger.info("SUMMARY %s", json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
