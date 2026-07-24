"""Preprocess STEAD HDF5 into split-wise numpy memmap files (L0 -> L2).

Converts merge.hdf5 (98 GB; per-trace random access is the training
bottleneck quantified in the paper) into contiguous float32 memmap files
per split. Waveform values are preserved bit-exactly; the deterministic
seed-42 permutation defines the 85/5/10 train/val/test split used
throughout the paper.

Usage:
    python scripts/preprocess_stead_memmap.py --config configs/stead_experiment.yaml
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess STEAD HDF5 → numpy memmap")
    parser.add_argument("--config", type=str, default="configs/stead_experiment.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    hdf5_path = cfg.data.stead_hdf5
    csv_path = cfg.data.stead_csv
    output_dir = Path(cfg.data.memmap_dir)
    seed = cfg.seed
    split_ratio = tuple(cfg.data.split_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # ── Phase 1: Build split mapping from CSV ──
    logger.info("Phase 1: Loading CSV and building split map...")
    t0 = time.time()

    df = pd.read_csv(
        csv_path,
        usecols=["trace_name", "p_arrival_sample", "s_arrival_sample"],
        low_memory=False,
    )

    all_names = df["trace_name"].values
    p_arr_raw = df["p_arrival_sample"].values
    s_arr_raw = df["s_arrival_sample"].values

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_names))

    n_total = len(indices)
    n_train = int(split_ratio[0] * n_total)
    n_val = int(split_ratio[1] * n_total)

    split_indices = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    # trace_name → (split, position_in_split)
    split_map = {}
    for split_name, idx_arr in split_indices.items():
        for pos, global_idx in enumerate(idx_arr):
            split_map[all_names[global_idx]] = (split_name, pos)

    split_counts = {k: len(v) for k, v in split_indices.items()}
    logger.info(
        "Split counts: train=%d, val=%d, test=%d (%.1fs)",
        split_counts["train"],
        split_counts["val"],
        split_counts["test"],
        time.time() - t0,
    )

    # ── Phase 2: Create memmap files and arrival arrays ──
    logger.info("Phase 2: Creating memmap files...")

    memmaps = {}
    p_arrays = {}
    s_arrays = {}

    for split_name, count in split_counts.items():
        npy_path = output_dir / f"{split_name}_waveforms.npy"
        memmaps[split_name] = np.lib.format.open_memmap(
            npy_path, mode="w+", dtype=np.float32, shape=(count, 3, 6000))
        p_arrays[split_name] = np.full(count, -1, dtype=np.int32)
        s_arrays[split_name] = np.full(count, -1, dtype=np.int32)

    # Pre-build arrival lookup from CSV (indexed by global position)
    p_arrivals_global = np.where(np.isnan(p_arr_raw), -1, p_arr_raw).astype(np.int32)
    s_arrivals_global = np.where(np.isnan(s_arr_raw), -1, s_arr_raw).astype(np.int32)

    # Map arrivals to split positions
    for split_name, idx_arr in split_indices.items():
        for pos, global_idx in enumerate(idx_arr):
            p_arrays[split_name][pos] = p_arrivals_global[global_idx]
            s_arrays[split_name][pos] = s_arrivals_global[global_idx]

    # ── Phase 3: Sequential HDF5 scan → write to memmaps ──
    logger.info("Phase 3: Sequential HDF5 scan (this takes ~30-40 min)...")
    t0 = time.time()

    written = 0
    skipped = 0

    with h5py.File(hdf5_path, "r") as f:
        keys = list(f["data"].keys())
        for trace_name in tqdm(keys, desc="Converting", unit="trace"):
            if trace_name not in split_map:
                skipped += 1
                continue

            split_name, pos = split_map[trace_name]
            waveform = f["data"][trace_name][:]  # (6000, 3)
            memmaps[split_name][pos] = waveform.T  # (3, 6000)
            written += 1

    elapsed = time.time() - t0
    logger.info(
        "HDF5 scan complete: %d written, %d skipped, %.1f min (%.0f traces/sec)",
        written,
        skipped,
        elapsed / 60,
        written / elapsed,
    )

    # ── Phase 4: Save arrivals and metadata ──
    logger.info("Phase 4: Saving arrivals and metadata...")

    for split_name in split_counts:
        # Flush memmap
        memmaps[split_name].flush()

        # Save arrivals
        np.savez(
            output_dir / f"{split_name}_arrivals.npz",
            p_samples=p_arrays[split_name],
            s_samples=s_arrays[split_name],
        )

    metadata = {
        "source_hdf5": str(hdf5_path),
        "source_csv": str(csv_path),
        "seed": seed,
        "split_ratio": list(split_ratio),
        "splits": {
            name: {
                "count": int(count),
                "waveforms": f"{name}_waveforms.npy",
                "arrivals": f"{name}_arrivals.npz",
            }
            for name, count in split_counts.items()
        },
        "created": datetime.now().isoformat(),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Done! Output: %s", output_dir)
    logger.info("Files: %s", list(output_dir.glob("*")))


if __name__ == "__main__":
    main()
