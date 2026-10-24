"""Fully label-free distillation: fixed recipe, no labeled validation anywhere.

Recipe fixed a priori (lr = 1e-4, temperature = 1, full 30-epoch budget,
batch 64) and applied to ALL training-side windows of each event split.
Manual picks are used only for the final held-out evaluation, never for
model selection or stopping.

The same run also reports, so that the manuscript quotes one protocol only:
  * the matching-tolerance sensitivity (recall/precision at 0.1/0.2/0.5 s,
    threshold 0.3) of the distilled student and of the zero-shot teacher on
    the same held-out windows (Table 16);
  * the adaptation cost: number of adaptation windows, teacher soft-label
    time, student distillation time (GPU) and the wavelet encoding time of
    all cached windows (CPU, PyWavelets) (Section 3.7).

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
TOLS = (0.1, 0.2, 0.5)  # matching tolerances of the sensitivity check (s)


def tolerance_rows(probs: np.ndarray, P: np.ndarray, S: np.ndarray, thr: float,
                   tols: tuple[float, ...] = TOLS) -> list[dict]:
    """Recall and precision of both phases at each matching tolerance."""
    rows = []
    for tol in tols:
        e = full_eval(probs, P, S, thr, int(round(tol * FS)))
        rows.append({"tol_sec": tol,
                     "P_recall": e["P"]["recall"], "S_recall": e["S"]["recall"],
                     "P_precision": e["P"]["precision"], "S_precision": e["S"]["precision"]})
    return rows


def summarize_runs(per_seed: list[dict]) -> dict:
    """Mean ± std of the test metrics plus timing and tolerance-sweep means."""
    def agg(phase, key):
        v = [r["test"][phase][key] for r in per_seed]
        return round(float(np.mean(v)), 3), round(float(np.std(v)), 3)

    out = {ph: {k: agg(ph, k) for k in ("f1", "recall", "precision", "res_mae")}
           for ph in ("P", "S")}
    out["ft_per_hour_pooled"] = round(float(np.mean(
        [r["test"]["ft_per_hour_pooled"] for r in per_seed])), 1)
    out["timing"] = {
        "train_sec_mean": round(float(np.mean([r["train_sec"] for r in per_seed])), 1),
        "teacher_sec_mean": round(float(np.mean([r["teacher_sec"] for r in per_seed])), 2),
        "n_adapt_windows_mean": round(float(np.mean([r["n_adapt_windows"] for r in per_seed])), 1),
    }
    sweep = {}
    for name, key in (("student", "student_tolerance"), ("teacher_zero_shot", "teacher_tolerance")):
        sweep[name] = {}
        for tol in sorted({row["tol_sec"] for r in per_seed for row in r[key]}):
            rows = [row for r in per_seed for row in r[key] if row["tol_sec"] == tol]
            sweep[name][str(tol)] = {m: round(float(np.mean([row[m] for row in rows])), 3)
                                     for m in ("P_recall", "S_recall", "P_precision", "S_precision")}
    out["tolerance_sweep"] = sweep
    return out


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
    import argparse
    ap = argparse.ArgumentParser(description="Fixed-recipe label-free distillation")
    ap.add_argument("--cr", type=int, default=4, choices=[2, 4, 8],
                    help="Student compression ratio (wavelet_cr{cr}_seed42 checkpoint)")
    ap.add_argument("--output", default=None,
                    help="Output JSON (default: manual2023_distill_fixed_cr{cr}.json; "
                         "the original CR=4 file is never overwritten)")
    args = ap.parse_args()
    cr = args.cr
    out_path = Path(args.output or (
        "outputs/results/manual2023_distill_fixed_rerun.json" if cr == 4
        else f"outputs/results/manual2023_distill_fixed_cr{cr}.json"))

    thr, tol_samp = 0.3, int(0.5 * FS)
    d = np.load(CACHE, allow_pickle=True)
    W, P, S, EV = d["windows"], d["p"], d["s"], d["ev"]
    cat = pd.read_csv("outputs/results/manual2023_catalog.csv", parse_dates=["ot"])
    ot_by_ev = {e: g.ot.iloc[0].timestamp() for e, g in cat.groupby("event_id")}

    mat4 = create_sensing_matrix("wavelet", N=6000, CR=cr)
    Xb = encode_norm(W, None)
    t0 = time.perf_counter()
    X4 = encode_norm(W, mat4)  # CPU wavelet encoding of every cached window
    t_encode = time.perf_counter() - t0

    out = {"meta": {"recipe": f"lr={LR}, T={TEMP}, fixed {EPOCHS} epochs, "
                              "all training-side windows, no labeled validation",
                    "student_cr": cr, "threshold": thr, "tol_sec": 0.5,
                    "tolerance_sweep_sec": list(TOLS),
                    "n_windows_total": int(len(W)),
                    "encode_all_windows_sec": round(t_encode, 2),
                    "device": str(DEV)},
           "per_seed": []}

    def _sync():
        if DEV.type == "cuda":
            torch.cuda.synchronize()

    for seed in (42, 123, 7):
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr, te, _, _ = cluster_split(EV, ot_by_ev, 0.25, seed)
        tr_idx = np.where(tr)[0]
        logger.info("seed %d | adapt windows %d | test windows %d",
                    seed, len(tr_idx), int(te.sum()))

        teacher = load_model("outputs/checkpoints/l2_baseline_seed42/best_model.pt")
        _sync()
        t0 = time.perf_counter()
        Tp = torch.softmax(logits_of(teacher, Xb[tr_idx]) / TEMP, 1).numpy()
        _sync()
        t_teacher = time.perf_counter() - t0

        student = load_model(f"outputs/checkpoints/wavelet_cr{cr}_seed42/best_model.pt")
        _sync()
        t0 = time.perf_counter()
        distill_fixed(student, X4[tr_idx], Tp)
        _sync()
        t_train = time.perf_counter() - t0

        student_probs = probs_np(logits_of(student, X4[te]))
        teacher_probs = probs_np(logits_of(teacher, Xb[te]))
        e = full_eval(student_probs, P[te], S[te], thr, tol_samp)
        logger.info("seed %d -> test P-F1 %.3f  S-F1 %.3f  (teacher %.2fs, train %.1fs)",
                    seed, e["P"]["f1"], e["S"]["f1"], t_teacher, t_train)
        out["per_seed"].append({
            "seed": seed, "n_adapt_windows": int(len(tr_idx)), "n_test_windows": int(te.sum()),
            "teacher_sec": round(t_teacher, 2), "train_sec": round(t_train, 1), "test": e,
            "student_tolerance": tolerance_rows(student_probs, P[te], S[te], thr),
            "teacher_tolerance": tolerance_rows(teacher_probs, P[te], S[te], thr)})

    out["summary"] = summarize_runs(out["per_seed"])

    path = out_path
    path.write_text(json.dumps(out, indent=1))
    logger.info("saved %s", path)
    logger.info("SUMMARY %s", json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
