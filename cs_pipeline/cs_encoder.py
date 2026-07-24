"""Offline CS pre-encoding of STEAD memmap waveforms.

Transforms original (3, 6000) waveforms into compressed (3, M) representations
using the configured encoder (wavelet band truncation, DCT, Butterworth,
LP-only, or Gaussian). Output is stored as new memmap files with labels
copied unchanged.

This is a one-time preprocessing cost that enables compressed-domain (L3) training.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-trace encoding
# ---------------------------------------------------------------------------


def encode_trace(
    waveform: np.ndarray,
    sensing_matrix: Any,
) -> np.ndarray:
    """Encode a single trace using a sensing matrix.

    Args:
        waveform: (C, N) waveform array (typically C=3, N=6000).
        sensing_matrix: Object with ``.encode()`` method that accepts
            a torch Tensor (1, C, N) and returns (1, C, M).

    Returns:
        (C, M) compressed waveform as float32 ndarray.
    """
    import torch

    # (C, N) → (1, C, N) tensor
    x = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
    y = sensing_matrix.encode(x)
    return y.squeeze(0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

# Module-level globals set by _worker_init
_worker_waveforms = None
_worker_matrix = None


def _worker_init(
    waveform_path: str,
    matrix_type: str,
    N: int,
    CR: int,
    seed: int,
) -> None:
    """Initialize worker process: open memmap and create sensing matrix."""
    global _worker_waveforms, _worker_matrix
    from cs_pipeline.sensing_matrix import create_sensing_matrix

    _worker_waveforms = np.load(waveform_path, mmap_mode="r")
    _worker_matrix = create_sensing_matrix(matrix_type, N=N, CR=CR, seed=seed)


def _worker_encode_chunk(chunk_indices: list[int]) -> list[np.ndarray]:
    """Encode a chunk of traces in the worker process.

    Args:
        chunk_indices: List of trace indices to encode.

    Returns:
        List of (C, M) encoded arrays.
    """
    results = []
    for idx in chunk_indices:
        waveform = _worker_waveforms[idx]  # (3, N)
        encoded = encode_trace(np.array(waveform), _worker_matrix)
        results.append(encoded)
    return results


# ---------------------------------------------------------------------------
# Dataset-level encoding
# ---------------------------------------------------------------------------


def encode_dataset(
    memmap_dir: str | Path,
    output_dir: str | Path,
    matrix_type: str,
    CR: int,
    seed: int = 42,
    num_workers: int = 8,
    splits: list[str] | None = None,
    chunk_size: int = 1000,
) -> Path:
    """Encode an entire STEAD memmap dataset into compressed memmap.

    Args:
        memmap_dir: Source directory with {split}_waveforms.npy and
            {split}_arrivals.npz files.
        output_dir: Destination directory for encoded files.
        matrix_type: 'gaussian' or 'dct'.
        CR: Compression ratio (4, 8, or 16).
        seed: Random seed for matrix generation.
        num_workers: Number of parallel workers.
        splits: Splits to encode. Default: ['train', 'val', 'test'].
        chunk_size: Traces per multiprocessing chunk.

    Returns:
        Path to output directory.
    """
    memmap_dir = Path(memmap_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = splits or ["train", "val", "test"]

    # Detect N from first available split
    N = None
    for s in splits:
        p = memmap_dir / f"{s}_waveforms.npy"
        if p.exists():
            sample = np.load(str(p), mmap_mode="r")
            N = sample.shape[2] if sample.ndim == 3 else 6000
            break
    if N is None:
        N = 6000

    # Get actual M from sensing matrix (wavelet M differs from N//CR)
    from cs_pipeline.sensing_matrix import create_sensing_matrix

    _probe_matrix = create_sensing_matrix(matrix_type, N=N, CR=CR, seed=seed)
    M = _probe_matrix.M

    metadata: dict[str, Any] = {
        "matrix_type": matrix_type,
        "CR": CR,
        "N": N,
        "M": M,
        "seed": seed,
        "source_dir": str(memmap_dir),
        "splits": {},
        "created": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "CS Encoding: matrix=%s, CR=%d, M=%d, workers=%d",
        matrix_type, CR, M, num_workers,
    )

    for split in splits:
        waveform_path = memmap_dir / f"{split}_waveforms.npy"
        arrivals_path = memmap_dir / f"{split}_arrivals.npz"

        if not waveform_path.exists():
            logger.warning("Skipping %s: %s not found", split, waveform_path)
            continue

        waveforms = np.load(str(waveform_path), mmap_mode="r")
        n_traces = len(waveforms)
        n_channels = waveforms.shape[1] if waveforms.ndim == 3 else 3

        logger.info(
            "Encoding %s: %d traces, (%d,%d) → (%d,%d)",
            split, n_traces, n_channels, N, n_channels, M,
        )

        # Create output memmap
        out_path = output_dir / f"{split}_waveforms.npy"
        out_memmap = np.lib.format.open_memmap(
            str(out_path),
            mode="w+",
            dtype=np.float32,
            shape=(n_traces, n_channels, M),
        )

        t_start = time.time()

        # Build chunk indices
        chunks = []
        for i in range(0, n_traces, chunk_size):
            chunks.append(list(range(i, min(i + chunk_size, n_traces))))

        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_traces, desc=f"  {split}", unit="traces")
        except ImportError:
            pbar = None
            logger.info("tqdm not installed — progress bar disabled")

        if num_workers > 1:
            with Pool(
                processes=num_workers,
                initializer=_worker_init,
                initargs=(str(waveform_path), matrix_type, N, CR, seed),
            ) as pool:
                for chunk_idx, chunk in enumerate(chunks):
                    results = pool.apply(
                        _worker_encode_chunk,
                        (chunk,),
                    )
                    for j, encoded in enumerate(results):
                        out_memmap[chunk[0] + j] = encoded
                    if pbar is not None:
                        pbar.update(len(chunk))
        else:
            # Single-process fallback
            from cs_pipeline.sensing_matrix import create_sensing_matrix

            matrix = create_sensing_matrix(matrix_type, N=N, CR=CR, seed=seed)
            for chunk in chunks:
                for idx in chunk:
                    waveform = np.array(waveforms[idx])
                    out_memmap[idx] = encode_trace(waveform, matrix)
                if pbar is not None:
                    pbar.update(len(chunk))

        if pbar is not None:
            pbar.close()

        out_memmap.flush()
        del out_memmap

        t_elapsed = time.time() - t_start
        logger.info(
            "  %s done: %.1fs (%.0f traces/sec)",
            split, t_elapsed, n_traces / t_elapsed if t_elapsed > 0 else 0,
        )

        # Copy labels
        if arrivals_path.exists():
            shutil.copy2(str(arrivals_path), str(output_dir / arrivals_path.name))
            logger.info("  Copied %s", arrivals_path.name)

        metadata["splits"][split] = {
            "n_traces": n_traces,
            "encoding_time_sec": round(t_elapsed, 2),
        }

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved: %s", meta_path)

    return output_dir
