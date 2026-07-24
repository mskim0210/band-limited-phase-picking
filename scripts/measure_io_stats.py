"""Measure per-process I/O statistics (read bytes, syscall counts) for L0/L2/L3.

Validates the "89% I/O stall" claim with system-level psutil counters.
Runs n_steps training iterations and records delta in read_bytes / read_count.

Output: outputs/profiles/io_stats.json

Usage:
    python scripts/measure_io_stats.py --config configs/stead_experiment.yaml --level L0
    python scripts/measure_io_stats.py --config configs/stead_experiment.yaml --level L2
    python scripts/measure_io_stats.py --config configs/stead_experiment.yaml --level L3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def measure_epoch_io(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    level_name: str,
    n_steps: int = 100,
    warmup_steps: int = 10,
    batch_size: int = 64,
) -> dict:
    """Measure actual I/O statistics for n_steps training iterations.

    Args:
        dataloader: DataLoader for the target level.
        model: CSPhaseNet model.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CUDA device.
        level_name: "L0", "L2", or "L3".
        n_steps: Number of measured steps (after warmup).
        warmup_steps: Steps to skip before measurement.
        batch_size: Batch size for per-trace calculation.

    Returns:
        Dict with level, read_bytes, read_count, write_bytes, etc.
    """
    proc = psutil.Process(os.getpid())
    model.train()
    data_iter = iter(dataloader)

    def _sum_io_tree() -> dict[str, int]:
        """Sum I/O counters for main process + all DataLoader workers.

        Uses read_chars (Linux rchar) which includes page-cache-served reads,
        unlike read_bytes which only counts actual storage-layer fetches.
        """
        totals = {"read_chars": 0, "read_count": 0,
                  "write_chars": 0, "write_count": 0,
                  "read_bytes": 0, "write_bytes": 0}
        targets = [proc] + proc.children(recursive=True)
        for p in targets:
            try:
                io = p.io_counters()
                totals["read_chars"] += io.read_chars
                totals["read_count"] += io.read_count
                totals["write_chars"] += io.write_chars
                totals["write_count"] += io.write_count
                totals["read_bytes"] += io.read_bytes
                totals["write_bytes"] += io.write_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return totals

    # Warmup — let page cache / HDF5 caches stabilize
    logger.info("Warmup: %d steps ...", warmup_steps)
    for i in range(warmup_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    # Flush page cache reads from warmup
    torch.cuda.synchronize()

    # Record start I/O counters (main + workers)
    io_start = _sum_io_tree()
    t_start = time.perf_counter()

    # Measured steps
    logger.info("Measuring: %d steps ...", n_steps)
    for i in range(n_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            logger.info("  Step %d/%d", i + 1, n_steps)

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    io_end = _sum_io_tree()

    total_traces = n_steps * batch_size
    read_chars = io_end["read_chars"] - io_start["read_chars"]
    read_count = io_end["read_count"] - io_start["read_count"]
    read_bytes = io_end["read_bytes"] - io_start["read_bytes"]
    write_bytes = io_end["write_bytes"] - io_start["write_bytes"]
    write_count = io_end["write_count"] - io_start["write_count"]
    wall_time = t_end - t_start

    return {
        "level": level_name,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "total_traces": total_traces,
        "wall_time_s": round(wall_time, 3),
        "read_chars": read_chars,
        "read_bytes_disk": read_bytes,
        "read_count": read_count,
        "write_bytes": write_bytes,
        "write_count": write_count,
        "read_chars_per_trace": round(read_chars / total_traces, 1),
        "read_chars_MB": round(read_chars / 1e6, 2),
        "read_chars_GB": round(read_chars / 1e9, 4),
    }


def build_loader_and_model(
    level: str, cfg: "DictConfig"
) -> tuple[DataLoader, nn.Module, nn.Module, torch.optim.Optimizer, torch.device, int]:
    """Build dataloader + model for the specified level.

    Returns:
        (dataloader, model, criterion, optimizer, device, batch_size)
    """
    from cs_pipeline.cs_aware_model import CSPhaseNet

    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers
    sigma = cfg.training.label_sigma
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if level == "L0":
        from scripts.profile_baseline import STEADHDF5Dataset, _hdf5_worker_init

        dataset = STEADHDF5Dataset(
            hdf5_path=cfg.data.stead_hdf5,
            csv_path=cfg.data.stead_csv,
            split="train",
            split_ratio=list(cfg.data.split_ratio),
            seed=cfg.seed,
            sigma=sigma,
        )
        worker_init_fn = _hdf5_worker_init
        M = 6000
    elif level == "L2":
        from scripts.profile_baseline import STEADMemmapDataset

        dataset = STEADMemmapDataset(
            memmap_dir=cfg.data.memmap_dir,
            split="train",
            sigma=sigma,
        )
        worker_init_fn = None
        M = 6000
    elif level == "L3":
        from cs_pipeline.cs_dataset import CompressedSTEADDataset

        cs_dir = Path(cfg.data.cs_dir) / "wavelet_cr4"
        dataset = CompressedSTEADDataset(
            compressed_dir=cs_dir,
            split="train",
            sigma=sigma,
        )
        worker_init_fn = None
        M = 6000 // 4  # CR=4 → 1500
    else:
        raise ValueError(f"Unsupported level: {level}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    model = CSPhaseNet(M=M, N=6000, base_channels=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    return dataloader, model, criterion, optimizer, device, batch_size


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Measure per-process I/O stats")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stead_experiment.yaml",
    )
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        choices=["L0", "L2", "L3"],
    )
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/profiles/io_stats.json",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build pipeline
    dataloader, model, criterion, optimizer, device, batch_size = (
        build_loader_and_model(args.level, cfg)
    )

    # Measure
    result = measure_epoch_io(
        dataloader=dataloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        level_name=args.level,
        n_steps=args.n_steps,
        warmup_steps=args.warmup_steps,
        batch_size=batch_size,
    )

    # Add metadata
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["device"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )

    # Merge into existing file (accumulate all levels)
    existing: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    existing[args.level] = result

    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info("Saved to %s", output_path)
    logger.info(
        "=== %s I/O Stats ===\n"
        "  Read chars:    %.2f MB (%.4f GB)\n"
        "  Read syscalls: %d\n"
        "  Disk reads:    %.2f MB\n"
        "  Chars/trace:   %.0f B (%.1f KB)\n"
        "  Wall time:     %.1f s",
        args.level,
        result["read_chars_MB"],
        result["read_chars_GB"],
        result["read_count"],
        result["read_bytes_disk"] / 1e6,
        result["read_chars_per_trace"],
        result["read_chars_per_trace"] / 1024,
        result["wall_time_s"],
    )


if __name__ == "__main__":
    main()
