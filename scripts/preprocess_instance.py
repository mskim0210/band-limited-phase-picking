"""Build a STEAD-style memmap from an INSTANCE subset (second-dataset experiment).

Crops INSTANCE traces P-centered to (3, 6000) and writes the same memmap layout
as the STEAD pipeline ({split}_waveforms.npy + {split}_arrivals.npz +
metadata.json), so the existing encode/train/evaluate scripts run unchanged on
INSTANCE simply by pointing --memmap-dir / config at the output.

Raw (un-normalized) waveforms are stored; normalization happens at load time,
matching the STEAD convention.

Usage:
    python scripts/preprocess_instance.py --out data/instance_memmap \
        --n-train 150000 --n-val 20000 --n-test 30000
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_LEN = 6000  # 60 s @ 100 Hz


def _crop_p_centered(wf: np.ndarray, p: int, s: int) -> tuple[np.ndarray, int, int]:
    """P-centered crop to TARGET_LEN; returns cropped (3,6000) and adjusted p,s."""
    total = wf.shape[1]
    half = TARGET_LEN // 2
    center = p if p >= 0 else total // 2
    start = max(0, center - half)
    end = start + TARGET_LEN
    if end > total:
        end = total
        start = max(0, end - TARGET_LEN)
    cropped = wf[:, start:end].astype(np.float32)
    if cropped.shape[1] < TARGET_LEN:
        pad = np.zeros((3, TARGET_LEN), dtype=np.float32)
        pad[:, : cropped.shape[1]] = cropped
        cropped = pad
    p2 = p - start if p >= 0 else -1
    s2 = s - start if s >= 0 else -1
    if not (0 <= p2 < TARGET_LEN):
        p2 = -1
    if not (0 <= s2 < TARGET_LEN):
        s2 = -1
    return cropped, p2, s2


def _arrival(meta: dict, key: str) -> int:
    raw = meta.get(key, np.nan)
    try:
        if isinstance(raw, float) and np.isnan(raw):
            return -1
        return int(float(raw))
    except (ValueError, TypeError):
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="INSTANCE → STEAD-style memmap subset")
    parser.add_argument("--out", default="data/instance_memmap")
    parser.add_argument("--n-train", type=int, default=150000)
    parser.add_argument("--n-val", type=int, default=20000)
    parser.add_argument("--n-test", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import seisbench.data as sbd

    inst = sbd.InstanceCounts()
    md = inst.metadata
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # INSTANCE split names → our split names
    split_map = [("train", "train", args.n_train),
                 ("dev", "val", args.n_val),
                 ("test", "test", args.n_test)]

    rng = np.random.RandomState(args.seed)
    metadata: dict = {"source": "INSTANCE (seisbench InstanceCounts)",
                      "target_len": TARGET_LEN, "splits": {}}

    for inst_split, our_split, n in split_map:
        idx_all = md.index[md["split"] == inst_split].tolist()
        if n < len(idx_all):
            sel = sorted(rng.choice(idx_all, size=n, replace=False).tolist())
        else:
            sel = idx_all
        n_act = len(sel)
        logger.info("Split %s→%s: %d traces", inst_split, our_split, n_act)

        wf_mm = np.lib.format.open_memmap(
            out / f"{our_split}_waveforms.npy", mode="w+",
            dtype=np.float32, shape=(n_act, 3, TARGET_LEN))
        p_arr = np.full(n_act, -1, dtype=np.int64)
        s_arr = np.full(n_act, -1, dtype=np.int64)

        for j, real_idx in enumerate(sel):
            wf, meta = inst.get_sample(real_idx)
            wf = np.asarray(wf, dtype=np.float32)
            if wf.shape[0] != 3:
                wf = wf[:3] if wf.shape[0] > 3 else np.pad(wf, ((0, 3 - wf.shape[0]), (0, 0)))
            p = _arrival(meta, "trace_P_arrival_sample")
            s = _arrival(meta, "trace_S_arrival_sample")
            cropped, p2, s2 = _crop_p_centered(wf, p, s)
            wf_mm[j] = cropped
            p_arr[j] = p2
            s_arr[j] = s2
            if (j + 1) % 10000 == 0:
                logger.info("  %s %d/%d", our_split, j + 1, n_act)

        wf_mm.flush()
        del wf_mm
        np.savez(out / f"{our_split}_arrivals.npz", p_samples=p_arr, s_samples=s_arr)
        metadata["splits"][our_split] = {
            "count": n_act,
            "p_present": int((p_arr >= 0).sum()),
            "s_present": int((s_arr >= 0).sum()),
        }
        logger.info("  saved %s (P present %d, S present %d)",
                    our_split, int((p_arr >= 0).sum()), int((s_arr >= 0).sum()))

    metadata["created"] = datetime.now(timezone.utc).isoformat()
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Done. Memmap at %s", out)


if __name__ == "__main__":
    main()
