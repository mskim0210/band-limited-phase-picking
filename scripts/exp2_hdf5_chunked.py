"""HDF5 chunked + cache-optimized baseline comparison.

Tests whether the 9.6x I/O speedup is an artifact of a *suboptimal* HDF5
baseline. On a representative subset we compare per-trace random-read latency
across three layouts that share identical waveform values:

  1. HDF5 per-trace datasets  -- STEAD's native layout (one dataset per trace)
  2. HDF5 chunked single dataset -- (N,3,6000), chunks=(1,3,6000),
     opened with a 256 MB chunk cache (rdcc_nbytes)
  3. memmap                    -- contiguous .npy

Reads are warm (page-cached), which isolates the per-trace access/API overhead
(B-tree traversal, h5py call path, PHIL) from raw disk I/O. Expected: chunked
HDF5 beats per-trace HDF5 but still trails memmap, i.e. the bottleneck is the
per-dataset layout/API, not a lack of chunking optimization.

Usage:
    python scripts/exp2_hdf5_chunked.py --config configs/stead_experiment.yaml \
        --n-traces 50000 --n-reads 20000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

RDCC_NBYTES = 256 * 1024 * 1024  # 256 MB chunk cache
RDCC_NSLOTS = 10007


def build_variants(src_memmap: np.ndarray, n: int, tmp_dir: Path) -> dict[str, Path]:
    """Materialize per-trace HDF5, chunked HDF5, and memmap for the first n traces."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "hdf5_pertrace": tmp_dir / "subset_pertrace.h5",
        "hdf5_chunked": tmp_dir / "subset_chunked.h5",
        "memmap": tmp_dir / "subset_memmap.npy",
    }

    if not paths["hdf5_pertrace"].exists():
        logger.info("Building per-trace HDF5 (%d datasets) ...", n)
        with h5py.File(paths["hdf5_pertrace"], "w") as f:
            grp = f.create_group("data")
            for i in range(n):
                grp.create_dataset(str(i), data=np.asarray(src_memmap[i]), dtype="f4")
                if (i + 1) % 10000 == 0:
                    logger.info("  pertrace %d/%d", i + 1, n)

    if not paths["hdf5_chunked"].exists():
        logger.info("Building chunked HDF5 (1 dataset, chunks=(1,3,6000)) ...")
        with h5py.File(paths["hdf5_chunked"], "w") as f:
            dset = f.create_dataset(
                "data", shape=(n, 3, 6000), dtype="f4", chunks=(1, 3, 6000)
            )
            for i0 in range(0, n, 10000):
                i1 = min(i0 + 10000, n)
                dset[i0:i1] = np.asarray(src_memmap[i0:i1])
                logger.info("  chunked %d/%d", i1, n)

    if not paths["memmap"].exists():
        logger.info("Building memmap subset ...")
        mm = np.lib.format.open_memmap(
            paths["memmap"], mode="w+", dtype=np.float32, shape=(n, 3, 6000)
        )
        for i0 in range(0, n, 10000):
            i1 = min(i0 + 10000, n)
            mm[i0:i1] = np.asarray(src_memmap[i0:i1])
        mm.flush()
        del mm

    return paths


def profile_reads(paths: dict[str, Path], n: int, n_reads: int, seed: int) -> dict:
    """Warm-cache random per-trace read latency for each layout."""
    rng = np.random.RandomState(seed)
    order = rng.randint(0, n, size=n_reads)

    out = {}

    # Warm the page cache + measure per-trace random reads.
    def _time(read_fn) -> dict:
        for i in order[: min(2000, n_reads)]:  # warmup pass
            read_fn(int(i))
        t0 = time.perf_counter()
        for i in order:
            read_fn(int(i))
        dt = time.perf_counter() - t0
        return {
            "mean_ms_per_trace": round(dt / n_reads * 1e3, 4),
            "throughput_traces_per_sec": round(n_reads / dt, 1),
        }

    # 1. per-trace HDF5
    f_pt = h5py.File(paths["hdf5_pertrace"], "r")
    grp = f_pt["data"]
    out["hdf5_pertrace"] = _time(lambda i: grp[str(i)][()])
    f_pt.close()

    # 2. chunked HDF5 with large chunk cache
    f_ck = h5py.File(
        paths["hdf5_chunked"], "r", rdcc_nbytes=RDCC_NBYTES, rdcc_nslots=RDCC_NSLOTS
    )
    dset = f_ck["data"]
    out["hdf5_chunked"] = _time(lambda i: dset[i])
    f_ck.close()

    # 3. memmap
    mm = np.load(paths["memmap"], mmap_mode="r")
    out["memmap"] = _time(lambda i: np.array(mm[i]))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked-HDF5 vs. memmap random-read latency")
    parser.add_argument("--config", default="configs/stead_experiment.yaml")
    parser.add_argument("--n-traces", type=int, default=50000)
    parser.add_argument("--n-reads", type=int, default=20000)
    parser.add_argument(
        "--tmp-dir", default="./_exp2_tmp"
    )
    parser.add_argument("--output", default="outputs/profiles/exp2_hdf5_chunked.json")
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    src = np.load(
        Path(cfg.data.memmap_dir) / "train_waveforms.npy", mmap_mode="r"
    )
    n = min(args.n_traces, len(src))
    tmp_dir = Path(args.tmp_dir)

    paths = build_variants(src, n, tmp_dir)
    logger.info("Profiling %d random reads over %d traces ...", args.n_reads, n)
    reads = profile_reads(paths, n, args.n_reads, cfg.seed)

    payload = {"n_traces": n, "n_reads": args.n_reads, "layouts": reads}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved: %s", out)

    print("\n=== Warm-cache per-trace random read ===")
    print(f"{'layout':>16s} {'ms/trace':>10s} {'traces/sec':>12s} {'vs memmap':>10s}")
    mm_ms = reads["memmap"]["mean_ms_per_trace"]
    for k in ("hdf5_pertrace", "hdf5_chunked", "memmap"):
        r = reads[k]
        print(
            f"{k:>16s} {r['mean_ms_per_trace']:>10.4f} "
            f"{r['throughput_traces_per_sec']:>12.0f} "
            f"{r['mean_ms_per_trace']/mm_ms:>9.1f}x"
        )

    if not args.keep_tmp:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Removed tmp dir: %s", tmp_dir)


if __name__ == "__main__":
    main()
