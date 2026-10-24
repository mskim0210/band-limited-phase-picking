"""Audit the cached event-level split: source_id disjointness and row coverage.

Maps every (file_id, row) of outputs/splits/event_split_seed{seed}.npz back to
its STEAD CSV row (the memmap files are a seed-42 permutation of the CSV cut
at 85/5/10 %), reads the ``source_id`` of that row and reports

* unique events and noise traces per split,
* pairwise source_id intersections (all must be 0),
* whether every memmap row is used exactly once.

Usage:
    python scripts/check_event_split.py --config configs/stead_experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs_pipeline.event_split import FILE_IDS, _offsets  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def overlap_report(sids: dict[str, np.ndarray]) -> dict:
    """Unique events, noise traces and pairwise source_id intersections.

    Args:
        sids: Per split, an object array of source_id strings (None/NaN = noise).
    """
    sets, noise = {}, {}
    for name, arr in sids.items():
        vals = [v for v in arr.tolist() if v is not None and v == v]
        sets[name] = set(vals)
        noise[name] = int(len(arr) - len(vals))
    inter = {f"{a}∩{b}": len(sets[a] & sets[b]) for a, b in PAIRS if a in sets and b in sets}
    return {"unique_events": {k: len(v) for k, v in sets.items()},
            "noise_traces": noise,
            "pairwise_intersection": inter,
            "disjoint": all(v == 0 for v in inter.values())}


def coverage_report(idx: dict[str, np.ndarray], n_rows: dict[int, int]) -> dict:
    """Check that the (file_id, row) pairs cover every memmap row exactly once."""
    seen = {fid: np.zeros(n, dtype=np.int64) for fid, n in n_rows.items()}
    for arr in idx.values():
        for fid in np.unique(arr[:, 0]):
            rows = arr[arr[:, 0] == fid, 1]
            np.add.at(seen[int(fid)], rows, 1)
    total = int(sum(n_rows.values()))
    once = all(bool(np.all(v == 1)) for v in seen.values())
    return {"rows_total": total, "rows_indexed": int(sum(len(a) for a in idx.values())),
            "rows_used_once": once}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stead_experiment.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-cache", default=None,
                    help="default outputs/splits/event_split_seed{seed}.npz")
    ap.add_argument("--output", default="outputs/results/event_split_check.json")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    cache = Path(args.split_cache or f"outputs/splits/event_split_seed{args.seed}.npz")
    ratio = tuple(cfg.data.split_ratio) if "split_ratio" in cfg.data else (0.85, 0.05, 0.10)

    meta = pd.read_csv(cfg.data.stead_csv, usecols=["source_id"], low_memory=False)
    sid_all = meta["source_id"].astype("string").to_numpy(dtype=object)
    sid_all = np.array([None if (v is pd.NA or v is None) else str(v) for v in sid_all], dtype=object)
    n_total = len(sid_all)
    n_train, n_val = _offsets(n_total, ratio)
    offsets = {0: 0, 1: n_train, 2: n_train + n_val}
    n_rows = {0: n_train, 1: n_val, 2: n_total - n_train - n_val}
    perm = np.random.RandomState(args.seed).permutation(n_total)

    d = np.load(cache)
    idx = {k: d[k] for k in FILE_IDS}
    sids = {}
    for name, arr in idx.items():
        csv_rows = perm[np.array([offsets[int(f)] for f in arr[:, 0]]) + arr[:, 1]]
        sids[name] = sid_all[csv_rows]
    report = {"split_cache": str(cache), "seed": args.seed, "split_ratio": list(ratio),
              "n_csv_rows": n_total, "memmap_rows": n_rows,
              "rows_per_split": {k: int(len(v)) for k, v in idx.items()},
              **overlap_report(sids), **coverage_report(idx, n_rows)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1))
    logger.info("%s", json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
