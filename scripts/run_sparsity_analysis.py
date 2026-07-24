"""Run STEAD waveform sparsity analysis.

Analyzes Wavelet and DCT domain sparsity to justify CS pre-encoding.
Outputs JSON results and publication-quality figures.

Usage:
    python scripts/run_sparsity_analysis.py --config configs/stead_experiment.yaml
    python scripts/run_sparsity_analysis.py --config configs/stead_experiment.yaml --n-traces 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs_pipeline.sparsity_analysis import analyze_sparsity, plot_sparsity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint: load config, run sparsity analysis, save results."""
    parser = argparse.ArgumentParser(description="STEAD sparsity analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stead_experiment.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=None,
        help="Override number of traces (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    sp = cfg.sparsity

    n_traces = args.n_traces or sp.n_traces
    seed = args.seed or sp.seed
    output_dir = Path(args.output_dir or sp.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load memmap
    memmap_dir = Path(cfg.data.memmap_dir)
    waveform_path = memmap_dir / "test_waveforms.npy"
    if not waveform_path.exists():
        logger.error("Memmap not found: %s", waveform_path)
        sys.exit(1)

    logger.info("Loading memmap from %s ...", waveform_path)
    waveforms = np.load(str(waveform_path), mmap_mode="r")
    logger.info("Waveform shape: %s, dtype: %s", waveforms.shape, waveforms.dtype)

    # Run analysis
    logger.info(
        "Starting sparsity analysis: n_traces=%d, wavelet=%s, level=%d",
        n_traces, sp.wavelet, sp.wavelet_level,
    )
    results = analyze_sparsity(
        waveforms=waveforms,
        n_traces=n_traces,
        seed=seed,
        wavelet=sp.wavelet,
        wavelet_level=sp.wavelet_level,
        kept_ratios=list(sp.kept_ratios),
        compression_ratios=list(sp.compression_ratios),
    )

    # Save JSON
    json_path = output_dir / "sparsity_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", json_path)

    # Generate figures
    fig_paths = plot_sparsity(results, output_dir)
    for p in fig_paths:
        logger.info("Figure saved: %s", p)

    # Log key findings
    kf = results["key_findings"]
    logger.info(
        "\n"
        "========== KEY FINDINGS ==========\n"
        "  Wavelet 20%% energy: %.2f%%\n"
        "  DCT 20%% energy:     %.2f%%\n"
        "  CS justified:        %s\n"
        "  Best transform:      %s\n"
        "  CR=8 Wavelet energy: %.2f%%\n"
        "  CR=8 DCT energy:     %.2f%%\n"
        "==================================",
        kf["wavelet_20pct_energy"],
        kf["dct_20pct_energy"],
        kf["cs_justified"],
        kf["best_transform"],
        kf["cr8_wavelet_energy"],
        kf["cr8_dct_energy"],
    )


if __name__ == "__main__":
    main()
