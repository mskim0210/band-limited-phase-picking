"""Offline pre-encoding of STEAD memmap dataset.

Generates compressed memmap files for L3/L4 experiments.
Supports CR={4,8,16} × matrix={gaussian,dct} = 6 encoding sets.

Output structure:
    data/cs_encoded/
    ├── gaussian_cr4/  (train/val/test_waveforms.npy + arrivals + metadata.json)
    ├── gaussian_cr8/
    ├── gaussian_cr16/
    ├── dct_cr4/
    ├── dct_cr8/
    └── dct_cr16/

Usage:
    # Single set
    python scripts/preprocess_cs.py --config configs/stead_experiment.yaml --cr 8 --matrix gaussian

    # All 6 sets
    python scripts/preprocess_cs.py --config configs/stead_experiment.yaml --all

    # Custom paths
    python scripts/preprocess_cs.py --memmap-dir /path/to/memmap --output-dir ./data/cs_encoded --cr 8 --matrix dct
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs_pipeline.cs_encoder import encode_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ALL_CRS = [4, 8, 16]
ALL_MATRICES = ["gaussian", "dct"]


def main() -> None:
    """CLI entrypoint for CS pre-encoding."""
    parser = argparse.ArgumentParser(description="Offline pre-encoding of STEAD memmap splits")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stead_experiment.yaml",
        help="Experiment config YAML",
    )
    parser.add_argument(
        "--memmap-dir",
        type=str,
        default=None,
        help="Override source memmap directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output base directory",
    )
    parser.add_argument(
        "--cr",
        type=int,
        default=None,
        choices=[1, 2, 4, 8, 16],
        help="Compression ratio (single set mode; use 1 for lowpass)",
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default=None,
        choices=["wavelet", "gaussian", "dct", "butterworth", "lowpass"],
        help="Matrix type (single set mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Encode all 6 sets (CR={4,8,16} × matrix={gaussian,dct})",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    memmap_dir = args.memmap_dir or cfg.data.memmap_dir
    output_base = Path(args.output_dir or cfg.data.cs_dir)
    seed = args.seed or cfg.seed

    # Determine what to encode
    if args.all:
        jobs = [(m, cr) for m in ALL_MATRICES for cr in ALL_CRS]
    elif args.cr is not None and args.matrix is not None:
        jobs = [(args.matrix, args.cr)]
    elif args.cr is not None or args.matrix is not None:
        logger.error("Both --cr and --matrix must be specified together, or use --all")
        sys.exit(1)
    else:
        # Default: all 6 sets
        jobs = [(m, cr) for m in ALL_MATRICES for cr in ALL_CRS]

    logger.info("CS Pre-encoding: %d job(s), source=%s", len(jobs), memmap_dir)

    for matrix_type, cr in jobs:
        output_dir = output_base / f"{matrix_type}_cr{cr}"
        logger.info("=" * 60)
        logger.info("Encoding: %s CR=%d → %s", matrix_type, cr, output_dir)
        logger.info("=" * 60)

        try:
            encode_dataset(
                memmap_dir=memmap_dir,
                output_dir=output_dir,
                matrix_type=matrix_type,
                CR=cr,
                seed=seed,
                num_workers=args.num_workers,
            )
        except Exception:
            logger.exception("Failed: %s CR=%d", matrix_type, cr)

    logger.info("All CS pre-encoding jobs complete.")


if __name__ == "__main__":
    main()
