"""P-onset PSD analysis for implicit denoising evidence.

Computes average Welch PSD for three waveform segments:
  - Pre-P noise (2s before P-arrival)
  - P-onset signal (2s after P-arrival)
  - S-onset signal (2s centered on S-arrival)

Generates the PSD figure with wavelet detail-band overlays (Fig. 6).

Usage:
    python scripts/analyze_psd.py --config configs/stead_experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

FS = 100  # STEAD sample rate
WINDOW_SEC = 2  # seconds per segment
WINDOW_SAMPLES = WINDOW_SEC * FS  # 200 samples


def analyze_psd(
    memmap_dir: str,
    csv_path: str,
    n_traces: int = 1000,
    seed: int = 42,
    split_ratio: list[float] | None = None,
) -> dict:
    """Compute average PSD for pre-P noise, P-onset, and S-onset segments.

    Args:
        memmap_dir: Path to STEAD memmap directory.
        csv_path: Path to STEAD merge.csv.
        n_traces: Number of traces to sample.
        seed: Random seed.
        split_ratio: Train/val/test split ratios.

    Returns:
        Dict with frequencies and mean PSDs per segment.
    """
    split_ratio = split_ratio or [0.85, 0.05, 0.10]

    # Load memmap waveforms (test set)
    waveforms = np.load(
        str(Path(memmap_dir) / "test_waveforms.npy"), mmap_mode="r"
    )
    arrivals = np.load(str(Path(memmap_dir) / "test_arrivals.npz"))
    p_samples = arrivals["p_samples"]
    s_samples = arrivals["s_samples"]

    n_total = len(p_samples)
    logger.info("Test set: %d traces", n_total)

    # Load SNR from CSV for stratification
    logger.info("Loading SNR from %s ...", csv_path)
    df = pd.read_csv(csv_path, usecols=["snr_db"], low_memory=False)
    n_csv = len(df)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_csv)
    n_train = int(n_csv * split_ratio[0])
    n_val = int(n_csv * split_ratio[1])
    test_indices = indices[n_train + n_val:]

    def parse_snr(val: object) -> float:
        try:
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).strip("[] ")
            return float(np.mean([float(v) for v in s.split()]))
        except (ValueError, TypeError):
            return float("nan")

    test_snrs = np.array([
        parse_snr(df.iloc[test_indices[i]]["snr_db"])
        if i < len(test_indices) else float("nan")
        for i in range(n_total)
    ])

    # Sample traces with valid P and S arrivals
    valid = []
    for i in range(n_total):
        p, s = int(p_samples[i]), int(s_samples[i])
        if (p >= WINDOW_SAMPLES and p + WINDOW_SAMPLES < 6000 and
                s >= WINDOW_SAMPLES and s + WINDOW_SAMPLES < 6000 and
                s > p):
            valid.append(i)

    logger.info("Valid traces (P&S in range): %d / %d", len(valid), n_total)
    rng2 = np.random.RandomState(seed)
    selected = rng2.choice(valid, size=min(n_traces, len(valid)), replace=False)

    # Compute PSD per segment
    psd_noise_all, psd_p_all, psd_s_all = [], [], []
    snr_selected = []

    for idx in selected:
        wf = waveforms[idx].astype(np.float32)  # (3, 6000)
        z = wf[2]  # Z-component (vertical, best for P)

        p = int(p_samples[idx])
        s = int(s_samples[idx])

        # Segments
        noise_seg = z[p - WINDOW_SAMPLES: p]
        p_seg = z[p: p + WINDOW_SAMPLES]
        s_start = max(0, s - WINDOW_SAMPLES // 2)
        s_seg = z[s_start: s_start + WINDOW_SAMPLES]

        # Welch PSD
        f, psd_n = welch(noise_seg, fs=FS, nperseg=128, noverlap=64)
        _, psd_p = welch(p_seg, fs=FS, nperseg=128, noverlap=64)
        _, psd_s = welch(s_seg, fs=FS, nperseg=128, noverlap=64)

        psd_noise_all.append(psd_n)
        psd_p_all.append(psd_p)
        psd_s_all.append(psd_s)
        snr_selected.append(test_snrs[idx])

    psd_noise_all = np.array(psd_noise_all)
    psd_p_all = np.array(psd_p_all)
    psd_s_all = np.array(psd_s_all)
    snr_selected = np.array(snr_selected)

    # Mean PSDs (all)
    result = {
        "frequencies": f.tolist(),
        "n_traces": len(selected),
        "all": {
            "noise_mean": np.mean(psd_noise_all, axis=0).tolist(),
            "p_onset_mean": np.mean(psd_p_all, axis=0).tolist(),
            "s_onset_mean": np.mean(psd_s_all, axis=0).tolist(),
        },
    }

    # SNR-stratified
    for label, lo, hi in [("low_snr", -999, 10), ("high_snr", 20, 999)]:
        mask = (snr_selected >= lo) & (snr_selected < hi)
        n_bin = mask.sum()
        if n_bin > 10:
            result[label] = {
                "n_traces": int(n_bin),
                "noise_mean": np.mean(psd_noise_all[mask], axis=0).tolist(),
                "p_onset_mean": np.mean(psd_p_all[mask], axis=0).tolist(),
                "s_onset_mean": np.mean(psd_s_all[mask], axis=0).tolist(),
            }
            logger.info("  %s: %d traces", label, n_bin)

    return result


def plot_psd(result: dict, output_dir: str | Path) -> Path:
    """Generate PSD figure with wavelet band overlays.

    Args:
        result: Output of analyze_psd().
        output_dir: Directory for figure output.

    Returns:
        Path to saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    freq = np.array(result["frequencies"])

    def _to_db(arr: list) -> np.ndarray:
        return 10 * np.log10(np.array(arr) + 1e-20)

    # Panel data
    panels = [
        ("all", f"(a) All traces ($N$={result['n_traces']:,})", True),
    ]
    if "low_snr" in result:
        n_low = result["low_snr"]["n_traces"]
        panels.append(("low_snr", f"(b) Low SNR (SNR<10 dB, $N$={n_low})", False))

    fig, axes = plt.subplots(1, len(panels), figsize=(7, 2.8), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (key, title, show_legend) in zip(axes, panels):
        data = result[key]
        noise = _to_db(data["noise_mean"])
        p_onset = _to_db(data["p_onset_mean"])
        s_onset = _to_db(data["s_onset_mean"])

        # Band shading
        ax.axvspan(25, 50, alpha=0.15, color="red",
                   label=r"$c_{D_1}$ (removed at CR$\geq$2)")
        ax.axvspan(12.5, 25, alpha=0.15, color="orange",
                   label=r"$c_{D_2}$ (removed at CR$\geq$4)")

        # PSD curves
        ax.plot(freq, noise, color="gray", linewidth=1.0, label="Pre-P noise")
        ax.plot(freq, p_onset, color="blue", linewidth=1.0, label="P-onset signal")
        ax.plot(freq, s_onset, color="red", linewidth=1.0, label="S-onset signal")

        # Reference lines
        ax.axvline(12.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.axvline(25, color="gray", linestyle=":", linewidth=0.5, alpha=0.7)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, 50)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

        if show_legend:
            ax.legend(loc="lower left", fontsize=5.5, framealpha=0.9)

    axes[0].set_ylabel("Power Spectral Density (dB)")
    fig.tight_layout()

    fig_path = output_dir / "fig_psd_bands.png"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    logger.info("Saved %s", fig_path)

    return fig_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="PSD analysis for denoising evidence")
    parser.add_argument("--config", default="configs/stead_experiment.yaml")
    parser.add_argument("--n-traces", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    result = analyze_psd(
        memmap_dir=cfg.data.memmap_dir,
        csv_path=cfg.data.stead_csv,
        n_traces=args.n_traces,
        seed=cfg.seed,
        split_ratio=list(cfg.data.split_ratio),
    )

    # Save results
    out_json = Path("outputs/results/psd_analysis.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s", out_json)

    # Generate figure
    fig_path = plot_psd(result, args.output_dir)
    logger.info("Done. Figure: %s", fig_path)


if __name__ == "__main__":
    main()
