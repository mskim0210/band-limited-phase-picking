"""DataLoader num_workers sensitivity sweep.

Measures the full training-step throughput, GPU utilization, and I/O stall
fraction as a function of DataLoader num_workers, for L0 (HDF5 per-trace) and
L2 (memmap). Using the real CSPhaseNet step (forward + backward) keeps the
consumer rate realistic and avoids prefetch-buffer artifacts. The HDF5 path is
expected to show diminishing returns (CPU-bound per-trace deserialization),
whereas memmap saturates the GPU with few workers.

Reuses the validated profiling loop in profile_baseline.run_profile.

Usage:
    python scripts/exp3_num_workers_sweep.py --config configs/stead_experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.profile_baseline import run_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

WORKER_SETTINGS = [0, 2, 4, 8, 16]


def _read_summary(json_path: Path) -> dict:
    with open(json_path) as f:
        saved = json.load(f)
    s = saved.get("summary", {})
    return {
        "throughput_traces_per_sec": s.get("throughput_traces_per_sec", 0.0),
        "t_io_mean_s": s.get("t_io", {}).get("mean", 0.0),
        "t_total_mean_s": s.get("t_total", {}).get("mean", 0.0),
        "gpu_util_mean_pct": s.get("gpu_util_mean_pct", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="num_workers sweep (L0 vs. L2)")
    parser.add_argument("--config", default="configs/stead_experiment.yaml")
    parser.add_argument("--levels", nargs="+", default=["L2", "L0"])
    parser.add_argument(
        "--output", default="outputs/profiles/exp3_num_workers_sweep.json"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    output_dir = Path(cfg.profiling.output_dir)
    results: dict[str, dict[str, dict]] = {}

    for level in args.levels:
        results[level] = {}
        for nw in WORKER_SETTINGS:
            cfg.training.num_workers = nw
            cfg._output_prefix = f"exp3_nw{nw}_"
            logger.info("=== %s  num_workers=%d ===", level, nw)
            try:
                out_path = run_profile(level, cfg, output_dir)
                summ = _read_summary(out_path)
            except Exception:
                logger.exception("Failed %s nw=%d", level, nw)
                summ = {"throughput_traces_per_sec": 0.0, "t_io_mean_s": 0.0,
                        "t_total_mean_s": 0.0, "gpu_util_mean_pct": 0.0}
            results[level][str(nw)] = summ
            logger.info(
                "%s nw=%2d -> %.0f tr/s, t_io=%.4fs, GPU=%.1f%%",
                level, nw, summ["throughput_traces_per_sec"],
                summ["t_io_mean_s"], summ["gpu_util_mean_pct"],
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_size": cfg.training.batch_size,
        "num_steps": cfg.profiling.num_steps,
        "worker_settings": WORKER_SETTINGS,
        "results": results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved: %s", out)

    print("\n=== Training throughput (traces/sec) vs num_workers ===")
    print(f"{'level':>8s} " + " ".join(f"{nw:>9d}" for nw in WORKER_SETTINGS))
    for level in args.levels:
        row = " ".join(
            f"{results[level][str(nw)]['throughput_traces_per_sec']:>9.0f}"
            for nw in WORKER_SETTINGS
        )
        print(f"{level:>8s} {row}")


if __name__ == "__main__":
    main()
