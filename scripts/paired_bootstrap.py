"""Paired trace-bootstrap CIs for F1 differences between two models.

For each pair (run_a, run_b) and seed, both checkpoints are evaluated on the
same STEAD test traces (per-trace tp/fp/fn from
``scripts.cluster_bootstrap.get_per_trace_results``, cached as npz).  The same
resampled traces are then used for both models and for every seed, giving a
95% percentile CI of the seed-mean paired difference ΔF1 = F1_b − F1_a (and
per-seed CIs).  This is the statistic behind the "statistically resolvable"
statements for Table 11 (CR = 4 vs. full-band) and the LP-only control
(Table 9).

Usage:
    python scripts/paired_bootstrap.py --B 10000 \
        --output outputs/results/paired_bootstrap.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

COLS = ("p_tp", "p_fp", "p_fn", "s_tp", "s_fp", "s_fn")
PAIRS = (("CSPhaseNet", "l2_baseline", "wavelet_cr4"),
         ("PhaseNet", "phasenet_baseline", "phasenet_wavelet_cr4"),
         ("EQTransformer", "eqt_baseline", "eqt_wavelet_cr4"),
         ("CSPhaseNet LP-only", "l2_baseline", "lowpass_cr1"))


def f1_from_counts(tp, fp, fn) -> np.ndarray:
    """Element-wise F1 = 2tp / (2tp + fp + fn); 0 where undefined."""
    tp, fp, fn = (np.asarray(x, dtype=np.float64) for x in (tp, fp, fn))
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1.0), 0.0)


def counts_array(rows: list[dict]) -> np.ndarray:
    """(n, 6) int32 array with columns ``COLS`` from per-trace dicts."""
    return np.array([[r[c] for c in COLS] for r in rows], dtype=np.int32)


def _resampled_sums(M: np.ndarray, B: int, seed: int, chunk: int = 100,
                    clusters: np.ndarray | None = None) -> np.ndarray:
    """Column sums of ``M`` over B resamples with replacement.

    Traces are resampled individually, or—when ``clusters`` (one id per
    trace) is given—whole clusters are drawn and every trace of a drawn
    cluster is counted once per draw (event-cluster bootstrap).
    """
    n = M.shape[0]
    rng = np.random.RandomState(seed)
    Mf = M.astype(np.float32)
    if clusters is not None:
        _, inv = np.unique(np.asarray(clusters), return_inverse=True)
        K = int(inv.max()) + 1
    out = np.empty((B, M.shape[1]), dtype=np.float64)
    for i0 in range(0, B, chunk):
        b = min(chunk, B - i0)
        W = np.zeros((b, n), dtype=np.float32)
        for j in range(b):
            if clusters is None:
                W[j] = np.bincount(rng.randint(0, n, n), minlength=n)
            else:
                W[j] = np.bincount(rng.randint(0, K, K), minlength=K)[inv]
        out[i0:i0 + b] = W @ Mf
    return out


def paired_delta_ci(counts_a: list[np.ndarray], counts_b: list[np.ndarray],
                    B: int = 10000, seed: int = 42, chunk: int = 100,
                    clusters: np.ndarray | None = None) -> dict:
    """Paired bootstrap of ΔF1 = F1_b − F1_a (percentage points).

    Args:
        counts_a: Per seed, an (n, 6) count array for model a (same trace order).
        counts_b: Per seed, the matching array for model b.
        B: Number of resamples (the same resampled traces for both models and
            every seed).
        seed: RNG seed.
        clusters: Optional cluster id per trace (e.g. STEAD source_id); when
            given, clusters are resampled instead of traces.

    Returns:
        ``{"B", "n_traces", "n_seeds", "P": {...}, "S": {...}}`` where each
        phase holds the seed-mean ``delta_pts`` with ``ci_low_pts`` /
        ``ci_high_pts`` (2.5 / 97.5 percentiles) and a ``per_seed`` list.
    """
    S = len(counts_a)
    n = counts_a[0].shape[0]
    assert len(counts_b) == S and all(c.shape == (n, 6) for c in counts_a + counts_b)
    M = np.concatenate([*counts_a, *counts_b], axis=1)  # (n, 12 S)
    sums = _resampled_sums(M, B, seed, chunk, clusters)
    full = M.sum(axis=0, dtype=np.int64)[None, :].astype(np.float64)

    def f1(block: np.ndarray, ph: str) -> np.ndarray:
        o = 0 if ph == "P" else 3
        return f1_from_counts(block[:, o], block[:, o + 1], block[:, o + 2])

    def blk(arr: np.ndarray, model: int, s: int) -> np.ndarray:
        c0 = 6 * (model * S + s)
        return arr[:, c0:c0 + 6]

    res: dict = {"B": int(B), "n_traces": int(n), "n_seeds": S,
                 "resampling": "trace" if clusters is None else "event_cluster"}
    if clusters is not None:
        res["n_clusters"] = int(len(np.unique(np.asarray(clusters))))
    for ph in ("P", "S"):
        D = np.stack([f1(blk(sums, 1, s), ph) - f1(blk(sums, 0, s), ph) for s in range(S)], axis=1)
        D0 = np.array([float(f1(blk(full, 1, s), ph)[0] - f1(blk(full, 0, s), ph)[0]) for s in range(S)])
        mean = D.mean(axis=1)
        pct = lambda v, q: round(100 * float(np.percentile(v, q)), 2)  # noqa: E731
        res[ph] = {"delta_pts": round(100 * float(D0.mean()), 2),
                   "ci_low_pts": pct(mean, 2.5), "ci_high_pts": pct(mean, 97.5),
                   "per_seed": [{"delta_pts": round(100 * float(D0[s]), 2),
                                 "ci_low_pts": pct(D[:, s], 2.5), "ci_high_pts": pct(D[:, s], 97.5)}
                                for s in range(S)]}
    return res


def load_counts(run: str, seed: int, cfg, cache_dir: Path) -> np.ndarray:
    """Per-trace counts for one checkpoint, cached as npz (GPU inference once)."""
    p = cache_dir / f"{run}_seed{seed}.npz"
    if p.exists():
        return np.load(p)["counts"]
    from scripts.cluster_bootstrap import get_per_trace_results, make_run_configs
    _, _, ckpt = make_run_configs([run], [seed])[0]
    logger.info("inference: %s seed %d (%s)", run, seed, ckpt)
    c = counts_array(get_per_trace_results(ckpt, cfg))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, counts=c)
    return c


def test_cluster_ids(cfg) -> np.ndarray:
    """STEAD source_id per test trace (memmap order); noise traces get unique ids."""
    import pandas as pd
    meta = pd.read_csv(cfg.data.stead_csv, usecols=["source_id"], low_memory=False)
    idx = np.random.RandomState(cfg.seed).permutation(len(meta))
    n_train = int(len(meta) * cfg.data.split_ratio[0])
    n_val = int(len(meta) * cfg.data.split_ratio[1])
    sid = meta["source_id"].astype("string").to_numpy(dtype=object)[idx[n_train + n_val:]]
    return np.array([f"_nan_{i}" if (v is pd.NA or v is None) else str(v) for i, v in enumerate(sid)],
                    dtype=object)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stead_experiment.yaml")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    ap.add_argument("--cache", default="outputs/results/per_trace_counts")
    ap.add_argument("--output", default="outputs/results/paired_bootstrap.json")
    args = ap.parse_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    cache = Path(args.cache)

    out: dict = {"B": args.B, "seeds": args.seeds, "resampling": "with replacement; identical resampled "
                 "traces (or event clusters) for both models and all seeds; percentile 95% CI of the "
                 "seed-mean difference. Top-level P/S: trace-level; event_cluster: traces sharing a "
                 "STEAD source_id drawn together", "pairs": {}}
    clusters = test_cluster_ids(cfg)
    logger.info("test clusters: %d over %d traces", len(np.unique(clusters)), len(clusters))
    for label, run_a, run_b in PAIRS:
        ca = [load_counts(run_a, s, cfg, cache) for s in args.seeds]
        cb = [load_counts(run_b, s, cfg, cache) for s in args.seeds]
        r = paired_delta_ci(ca, cb, B=args.B)
        rc = paired_delta_ci(ca, cb, B=args.B, clusters=clusters)
        r["event_cluster"] = {k: rc[k] for k in ("P", "S", "n_clusters", "resampling")}
        f1a = {ph: round(float(np.mean([f1_from_counts(*(c[:, o:o + 3].sum(axis=0))).item() for c in ca])), 4)
               for ph, o in (("P", 0), ("S", 3))}
        f1b = {ph: round(float(np.mean([f1_from_counts(*(c[:, o:o + 3].sum(axis=0))).item() for c in cb])), 4)
               for ph, o in (("P", 0), ("S", 3))}
        out["pairs"][label] = {"run_a": run_a, "run_b": run_b, "f1_a": f1a, "f1_b": f1b, **r}
        logger.info("%s: ΔP %+.2f [%+.2f, %+.2f]  ΔS %+.2f [%+.2f, %+.2f] pts | cluster P [%+.2f, %+.2f] S [%+.2f, %+.2f]",
                    label, r["P"]["delta_pts"], r["P"]["ci_low_pts"], r["P"]["ci_high_pts"],
                    r["S"]["delta_pts"], r["S"]["ci_low_pts"], r["S"]["ci_high_pts"],
                    rc["P"]["ci_low_pts"], rc["P"]["ci_high_pts"], rc["S"]["ci_low_pts"], rc["S"]["ci_high_pts"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    logger.info("saved %s", args.output)


if __name__ == "__main__":
    main()
