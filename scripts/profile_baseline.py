"""Baseline profiling for L0 (HDF5) and L2 (memmap) data loading.

Measures step time decomposition (t_io, t_preprocess, t_h2d, t_compute)
and GPU utilization for each data loading strategy.

Output: outputs/profiles/l0_hdf5.json, outputs/profiles/l2_memmap.json

Usage:
    python scripts/profile_baseline.py --config configs/stead_experiment.yaml
    python scripts/profile_baseline.py --config configs/stead_experiment.yaml --level L0
    python scripts/profile_baseline.py --config configs/stead_experiment.yaml --level L2
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

# Ensure cs_pipeline is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs_pipeline.profiler import GPUMonitor, StepProfiler, save_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class STEADHDF5Dataset(Dataset):
    """Minimal HDF5 random-read dataset for L0 profiling.

    Reads individual traces from the STEAD merge.hdf5 file.
    This is intentionally slow — it demonstrates the B-tree lookup bottleneck.

    Args:
        hdf5_path: Path to merge.hdf5.
        csv_path: Path to merge.csv (for trace name list).
        split: One of "train", "val", "test".
        split_ratio: Train/val/test split ratios.
        seed: Random seed for splitting.
        sigma: Gaussian label width in samples.
    """

    def __init__(
        self,
        hdf5_path: str,
        csv_path: str,
        split: str = "train",
        split_ratio: list[float] | None = None,
        seed: int = 42,
        sigma: float = 10.0,
    ) -> None:
        split_ratio = split_ratio or [0.85, 0.05, 0.10]
        self._hdf5_path = hdf5_path
        self._sigma = sigma
        self._file: h5py.File | None = None

        # Verify HDF5 file exists
        if not Path(hdf5_path).exists():
            raise FileNotFoundError(
                f"STEAD HDF5 not found: {hdf5_path}. "
                "Set data.stead_hdf5 in configs/stead_experiment.yaml"
            )

        # Load trace names from CSV
        logger.info("Loading trace names from %s ...", csv_path)
        df = pd.read_csv(csv_path)
        trace_names = df["trace_name"].values

        # Deterministic split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(trace_names))
        n_train = int(len(indices) * split_ratio[0])
        n_val = int(len(indices) * split_ratio[1])

        if split == "train":
            sel = indices[:n_train]
        elif split == "val":
            sel = indices[n_train : n_train + n_val]
        else:
            sel = indices[n_train + n_val :]

        self._trace_names = trace_names[sel]
        logger.info(
            "STEADHDF5Dataset(%s): %d traces", split, len(self._trace_names)
        )

    def _open(self) -> h5py.File:
        """Lazy-open HDF5 file (not fork-safe, opened per-worker)."""
        if self._file is None:
            self._file = h5py.File(self._hdf5_path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self._trace_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a single trace from HDF5 (slow B-tree lookup).

        Returns:
            waveform: (3, 6000) float32 tensor, std-normalized.
            label: (3, 6000) float32 tensor [P(Noise), P(P), P(S)].
        """
        f = self._open()
        name = self._trace_names[idx]
        dataset = f["data"][name]

        # Waveform: dataset shape is (6000, 3) in STEAD
        data = dataset[()]  # (6000, 3)
        waveform = data.T.astype(np.float32)  # (3, 6000)

        # Std normalization
        std = waveform.std()
        if std > 1e-8:
            waveform /= std

        # Arrivals from attributes (may be str or numeric in STEAD HDF5)
        p_raw = dataset.attrs.get("p_arrival_sample", "")
        s_raw = dataset.attrs.get("s_arrival_sample", "")
        try:
            p_sample = int(float(p_raw)) if p_raw != "" else -1
        except (ValueError, TypeError):
            p_sample = -1
        try:
            s_sample = int(float(s_raw)) if s_raw != "" else -1
        except (ValueError, TypeError):
            s_sample = -1
        label = self._make_gaussian_label(
            p_sample if p_sample >= 0 else None,
            s_sample if s_sample >= 0 else None,
        )

        return torch.from_numpy(waveform), torch.from_numpy(label)

    def _make_gaussian_label(
        self,
        p_sample: int | None,
        s_sample: int | None,
        n_samples: int = 6000,
    ) -> np.ndarray:
        """Generate (3, 6000) Gaussian label array."""
        label = np.zeros((3, n_samples), dtype=np.float32)
        x = np.arange(n_samples, dtype=np.float32)

        if p_sample is not None and 0 <= p_sample < n_samples:
            label[1] = np.exp(-((x - p_sample) ** 2) / (2 * self._sigma**2))
        if s_sample is not None and 0 <= s_sample < n_samples:
            label[2] = np.exp(-((x - s_sample) ** 2) / (2 * self._sigma**2))

        # Noise channel = 1 - P - S
        label[0] = np.clip(1.0 - label[1] - label[2], 0.0, 1.0)
        return label


def _hdf5_worker_init(worker_id: int) -> None:
    """Re-open HDF5 file in each DataLoader worker (h5py not fork-safe)."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._file = None  # Force re-open in this process


class STEADMemmapDataset(Dataset):
    """Numpy memmap dataset for L2 profiling.


    Args:
        memmap_dir: Directory with {split}_waveforms.npy and {split}_arrivals.npz.
        split: One of "train", "val", "test".
        sigma: Gaussian label width in samples.
    """

    def __init__(
        self,
        memmap_dir: str,
        split: str = "train",
        sigma: float = 10.0,
    ) -> None:
        data_dir = Path(memmap_dir)
        self._sigma = sigma

        waveform_path = data_dir / f"{split}_waveforms.npy"
        if not waveform_path.exists():
            raise FileNotFoundError(
                f"Memmap not found: {waveform_path}. "
                "Run scripts/preprocess_stead_memmap.py first."
            )

        self.waveforms = np.load(str(waveform_path), mmap_mode="r")

        arrivals = np.load(str(data_dir / f"{split}_arrivals.npz"))
        self.p_samples = arrivals["p_samples"]
        self.s_samples = arrivals["s_samples"]

        logger.info(
            "STEADMemmapDataset(%s): %d traces (memmap)", split, len(self.p_samples)
        )

    def __len__(self) -> int:
        return len(self.p_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load trace from memmap, normalize, generate Gaussian label.

        Returns:
            waveform: (3, 6000) float32 tensor, std-normalized.
            label: (3, 6000) float32 tensor [P(Noise), P(P), P(S)].
        """
        waveform = self.waveforms[idx].copy().astype(np.float32)  # (3, 6000)

        std = waveform.std()
        if std > 1e-8:
            waveform /= std

        p = self.p_samples[idx]
        s = self.s_samples[idx]
        label = self._make_gaussian_label(
            int(p) if p >= 0 else None,
            int(s) if s >= 0 else None,
        )

        return torch.from_numpy(waveform), torch.from_numpy(label)

    def _make_gaussian_label(
        self,
        p_sample: int | None,
        s_sample: int | None,
        n_samples: int = 6000,
    ) -> np.ndarray:
        """Generate (3, 6000) Gaussian label array."""
        label = np.zeros((3, n_samples), dtype=np.float32)
        x = np.arange(n_samples, dtype=np.float32)

        if p_sample is not None and 0 <= p_sample < n_samples:
            label[1] = np.exp(-((x - p_sample) ** 2) / (2 * self._sigma**2))
        if s_sample is not None and 0 <= s_sample < n_samples:
            label[2] = np.exp(-((x - s_sample) ** 2) / (2 * self._sigma**2))

        label[0] = np.clip(1.0 - label[1] - label[2], 0.0, 1.0)
        return label


# ---------------------------------------------------------------------------
# Model for profiling: CSPhaseNet (1.45M params — same as actual training)
# ---------------------------------------------------------------------------
# NOTE: Previously used DummyPhaseNet (36K params) which gave unrealistically
# low t_compute and GPU util (1-5%). Replaced with actual CSPhaseNet to get
# representative measurements.


# ---------------------------------------------------------------------------
# Profiling loop
# ---------------------------------------------------------------------------


def run_profile(
    level: str,
    cfg: "DictConfig",
    output_dir: Path,
) -> Path:
    """Run step-decomposed profiling for a given level.

    Args:
        level: "L0" (HDF5) or "L2" (memmap).
        cfg: OmegaConf config object.
        output_dir: Directory for JSON output.

    Returns:
        Path to the saved JSON profile.
    """
    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers
    num_steps = cfg.profiling.num_steps
    warmup_steps = cfg.profiling.warmup_steps
    sigma = cfg.training.label_sigma

    # -- Dataset --
    if level == "L0":
        dataset = STEADHDF5Dataset(
            hdf5_path=cfg.data.stead_hdf5,
            csv_path=cfg.data.stead_csv,
            split="train",
            split_ratio=list(cfg.data.split_ratio),
            seed=cfg.seed,
            sigma=sigma,
        )
        worker_init_fn = _hdf5_worker_init
        dataset_name = "stead_hdf5"
    elif level == "L2":
        dataset = STEADMemmapDataset(
            memmap_dir=cfg.data.memmap_dir,
            split="train",
            sigma=sigma,
        )
        worker_init_fn = None
        dataset_name = "stead_memmap"
    elif level == "L1":
        import seisbench.data as sbd

        class SeisBenchSTEADDataset(torch.utils.data.Dataset):
            """SeisBench STEAD trace blocks wrapper for L1 profiling."""

            def __init__(self, split: str = "train", sigma: float = 10.0) -> None:
                self._stead = sbd.STEAD()
                mask = self._stead.metadata["split"] == split
                self._indices = mask[mask].index.tolist()
                self._sigma = sigma
                logger.info("SeisBenchSTEAD(%s): %d traces", split, len(self._indices))

            def __len__(self) -> int:
                return len(self._indices)

            def __getitem__(self, idx: int) -> tuple:
                real_idx = self._indices[idx]
                wf, meta = self._stead.get_sample(real_idx)
                wf = wf.astype(np.float32)
                std = wf.std()
                if std > 1e-8:
                    wf /= std
                p_raw = meta.get("trace_p_arrival_sample", -1)
                s_raw = meta.get("trace_s_arrival_sample", -1)
                try:
                    p = int(float(p_raw)) if not (isinstance(p_raw, float) and np.isnan(p_raw)) else -1
                except (ValueError, TypeError):
                    p = -1
                try:
                    s = int(float(s_raw)) if not (isinstance(s_raw, float) and np.isnan(s_raw)) else -1
                except (ValueError, TypeError):
                    s = -1
                label = self._make_label(p if p >= 0 else None, s if s >= 0 else None)
                return torch.from_numpy(wf), torch.from_numpy(label)

            def _make_label(self, p, s, n=6000):
                label = np.zeros((3, n), dtype=np.float32)
                x = np.arange(n, dtype=np.float32)
                if p is not None and 0 <= p < n:
                    label[1] = np.exp(-((x - p) ** 2) / (2 * self._sigma ** 2))
                if s is not None and 0 <= s < n:
                    label[2] = np.exp(-((x - s) ** 2) / (2 * self._sigma ** 2))
                label[0] = np.clip(1.0 - label[1] - label[2], 0.0, 1.0)
                return label

        dataset = SeisBenchSTEADDataset(split="train", sigma=sigma)
        worker_init_fn = None
        dataset_name = "seisbench_trace_blocks"
    elif level == "L3":
        from cs_pipeline.cs_dataset import CompressedSTEADDataset

        cs_matrix = getattr(cfg, "_cs_matrix", "gaussian")
        cs_cr = getattr(cfg, "_cs_cr", 8)
        cs_dir = Path(cfg.data.cs_dir) / f"{cs_matrix}_cr{cs_cr}"
        if not cs_dir.exists():
            raise FileNotFoundError(
                f"CS encoded data not found: {cs_dir}. "
                "Run: python scripts/preprocess_cs.py first."
            )
        dataset = CompressedSTEADDataset(
            compressed_dir=cs_dir,
            split="train",
            sigma=sigma,
        )
        worker_init_fn = None
        dataset_name = f"cs_{cs_matrix}_cr{cs_cr}"
    else:
        raise ValueError(f"Unsupported level: {level}. Use L0, L2, or L3.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    # -- Model + optimizer (CSPhaseNet — same 1.45M param model used in training) --
    from cs_pipeline.cs_aware_model import CSPhaseNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if level == "L3":
        cs_cr = getattr(cfg, "_cs_cr", 8)
        M = 6000 // cs_cr
        bytes_per_trace = 3 * M * 4
    else:
        M = 6000
        bytes_per_trace = 3 * 6000 * 4
    base_ch = getattr(cfg, "_base_channels", 16)
    model = CSPhaseNet(M=M, N=6000, base_channels=base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.CrossEntropyLoss()

    # -- Profiler + GPU monitor --
    profiler = StepProfiler(
        warmup_steps=warmup_steps,
        bytes_per_trace=bytes_per_trace,
    )
    gpu_monitor = GPUMonitor(
        interval=cfg.profiling.gpu_log_interval,
        gpu_index=cfg.profiling.gpu_index,
    )

    # -- Profiling loop --
    logger.info("Starting %s profiling: %d steps (warmup=%d)", level, num_steps, warmup_steps)
    gpu_monitor.start()
    model.train()

    step = 0
    data_iter = iter(dataloader)
    while step < num_steps:
        # I/O phase
        profiler.start_io()
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        profiler.end_io(batch_size=x.size(0))

        # Preprocess phase (std normalization already done in dataset)
        profiler.start_preprocess()
        # Placeholder — normalization is in dataset.__getitem__
        profiler.end_preprocess()

        # H2D phase
        profiler.start_h2d()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        profiler.end_h2d()

        # Compute phase
        profiler.start_compute()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        profiler.end_compute()

        if (step + 1) % 20 == 0:
            logger.info("Step %d/%d complete", step + 1, num_steps)

        step += 1

    gpu_records = gpu_monitor.stop()
    step_result = profiler.summary()

    # -- Metadata --
    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu"
    )
    metadata = {
        "level": level,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_steps": num_steps,
        "warmup_steps": warmup_steps,
        "device": device_name,
        "base_channels": base_ch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # -- Save --
    out_prefix = getattr(cfg, "_output_prefix", "")
    filename = f"{out_prefix}{level.lower()}_{dataset_name}.json"
    output_path = save_profile(step_result, gpu_records, metadata, output_dir / filename)

    # -- Log summary (read from saved JSON to include GPU stats merged by save_profile) --
    import json as _json

    with open(output_path) as _f:
        saved = _json.load(_f)
    summary = saved.get("summary", {})
    logger.info(
        "=== %s Summary ===\n"
        "  t_io:         mean=%.4fs  p95=%.4fs\n"
        "  t_preprocess: mean=%.4fs  p95=%.4fs\n"
        "  t_h2d:        mean=%.4fs  p95=%.4fs\n"
        "  t_compute:    mean=%.4fs  p95=%.4fs\n"
        "  t_total:      mean=%.4fs  p95=%.4fs\n"
        "  throughput:   %.1f traces/sec\n"
        "  GPU util:     %.1f%%",
        level,
        summary.get("t_io", {}).get("mean", 0),
        summary.get("t_io", {}).get("p95", 0),
        summary.get("t_preprocess", {}).get("mean", 0),
        summary.get("t_preprocess", {}).get("p95", 0),
        summary.get("t_h2d", {}).get("mean", 0),
        summary.get("t_h2d", {}).get("p95", 0),
        summary.get("t_compute", {}).get("mean", 0),
        summary.get("t_compute", {}).get("p95", 0),
        summary.get("t_total", {}).get("mean", 0),
        summary.get("t_total", {}).get("p95", 0),
        summary.get("throughput_traces_per_sec", 0),
        summary.get("gpu_util_mean_pct", 0),
    )

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint: profile L0 and/or L2 baselines."""
    parser = argparse.ArgumentParser(description="Baseline I/O profiling (L0/L2/L3)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stead_experiment.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        choices=["L0", "L1", "L2", "L3"],
        help="Profile a specific level only (default: L0 and L2)",
    )
    parser.add_argument(
        "--cs-matrix",
        type=str,
        default="gaussian",
        choices=["gaussian", "dct", "wavelet"],
        help="CS matrix type for L3 profiling",
    )
    parser.add_argument(
        "--cs-cr",
        type=int,
        default=8,
        choices=[4, 8, 16],
        help="Compression ratio for L3 profiling",
    )
    parser.add_argument(
        "--base-ch",
        type=int,
        default=16,
        help="CSPhaseNet base_channels (16=S/1.45M, 32=M/5.8M, 64=L/23M)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Prefix for output filename (e.g. 'multimodel_csphaseNet_M_')",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg._cs_matrix = args.cs_matrix
    cfg._cs_cr = args.cs_cr
    cfg._base_channels = args.base_ch
    cfg._output_prefix = args.output_prefix
    output_dir = Path(cfg.profiling.output_dir)

    levels = [args.level] if args.level else ["L0", "L2"]

    for level in levels:
        logger.info("=" * 60)
        logger.info("Profiling %s ...", level)
        logger.info("=" * 60)
        try:
            result_path = run_profile(level, cfg, output_dir)
            logger.info("Result saved: %s", result_path)
        except Exception:
            logger.exception("Failed to profile %s", level)

    logger.info("Profiling complete.")


if __name__ == "__main__":
    main()
