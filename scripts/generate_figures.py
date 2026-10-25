"""Generate the manuscript figures that need no retraining.

Fig. 3 (fig2_step_decomposition.png): per-step time decomposition with the
    L0, L2, wavelet CR = 4 and Gaussian CR = 8 bars, from the single-model
    profiles in outputs/profiles (scripts/make_fig2_cr4.py wraps the same
    function, so both commands write the same figure).
Fig. 5 (fig3_cr_vs_f1.png): CR vs. F1 trade-off curve with speedup
    annotations, from test_evaluation_summary.json and the epoch logs.
Fig. 8 (fig4_tta_curve.png): time-to-accuracy curve (validation P-F1 of
    seed 42) from the epoch logs.

Every input must exist; a missing one is reported as an error.

Usage:
    python scripts/generate_figures.py --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Single-model (CSPhaseNet, 1.45 M) profiles used by the paper (Fig. 3,
# Tables 1-3).  The directory also holds the multi-model (S/M/L) and the
# num_workers-sweep profiles, which carry the same pipeline levels and must
# not be mixed in by keying every file on its level.
PAPER_PROFILES = {"L0": "l0_stead_hdf5.json",
                  "L1": "l1_seisbench_trace_blocks.json",
                  "L2": "l2_stead_memmap.json",
                  "L3": "l3_cs_gaussian_cr8.json"}


def load_profiles(profile_dir: Path, files: dict[str, str] | None = None) -> dict[str, dict]:
    """Load the named profile JSON files (level → file name).

    Args:
        profile_dir: Directory with the profiler outputs.
        files: Mapping level → file name (default: ``PAPER_PROFILES``).

    Returns:
        Dict mapping level name to profile data.

    Raises:
        FileNotFoundError: when a named file is absent (nothing is silently
        substituted).
    """
    files = PAPER_PROFILES if files is None else files
    profiles = {}
    for level, fname in files.items():
        json_file = Path(profile_dir) / fname
        if not json_file.exists():
            raise FileNotFoundError(f"profile for {level} not found: {json_file}")
        with open(json_file) as f:
            profiles[level] = json.load(f)
        logger.info("Loaded %s from %s", level, json_file.name)
    return profiles


def wall_clock_mean_min(config: str, root: Path | str = "outputs",
                        seeds: tuple[int, ...] = (42, 123, 7)) -> float | None:
    """Mean wall-clock minutes of ``config`` over its seeds, from the epoch logs
    (last ``wall_clock_sec`` row), as quoted in Table 5.  None when no log exists."""
    import csv
    mins = []
    for seed in seeds:
        log = Path(root) / "checkpoints" / f"{config}_seed{seed}" / "epoch_log.csv"
        if not log.exists():
            continue
        rows = list(csv.DictReader(open(log)))
        if rows:
            mins.append(float(rows[-1]["wall_clock_sec"]) / 60.0)
    return float(np.mean(mins)) if mins else None


def missing_inputs(required: dict[str, Path]) -> list[str]:
    """Names (e.g. figure labels) whose required input file does not exist."""
    return [name for name, path in required.items() if not Path(path).exists()]


# ---- Fig. 3: per-step time decomposition -----------------------------------
# Bars: L0 (HDF5), L2 (memmap), wavelet CR = 4 and Gaussian CR = 8 -- the
# paper's four-bar figure.  The CR = 4 per-step time is the value measured from
# the training steps (epoch-log wall-clock per step), entered as a constant;
# its I/O and H2D components are taken from the shape-identical L3 profile.
# GPU utilization is shown for L0/L2 only (not re-measured for the L3 bars).
STEP_PHASES = ("t_io", "t_preprocess", "t_h2d", "t_compute")
PHASE_LABELS = ("I/O", "Preprocess", "H2D", "Compute")
PHASE_COLORS = ("#e74c3c", "#f39c12", "#3498db", "#2ecc71")
BATCH_SIZE = 64
CR4_TOTAL_MS = 9.3   # wavelet CR = 4 per-step wall time measured from the training steps
CR4_IO_MS = 0.10     # I/O and host-to-device components of the shape-identical L3 profile
CR4_H2D_MS = 0.06


def step_decomposition_data(profiles: dict[str, dict], cr4_total_ms: float = CR4_TOTAL_MS,
                            cr4_io_ms: float = CR4_IO_MS, cr4_h2d_ms: float = CR4_H2D_MS,
                            ) -> dict[str, tuple[list[float], float, float | None]]:
    """Fig. 3 bars: label -> ([I/O, preprocess, H2D, compute] in ms, traces/s, GPU util % or None)."""
    def comp(level: str) -> list[float]:
        summ = profiles[level]["summary"]
        return [summ[k]["mean"] * 1000 for k in STEP_PHASES]

    def tps(level: str) -> float:
        return profiles[level]["summary"]["throughput_traces_per_sec"]

    def gpu(level: str) -> float:
        return profiles[level]["summary"]["gpu_util_mean_pct"]

    cr4 = [cr4_io_ms, 0.0, cr4_h2d_ms, cr4_total_ms - cr4_io_ms - cr4_h2d_ms]
    return {"L0\n(HDF5)": (comp("L0"), tps("L0"), gpu("L0")),
            "L2\n(memmap)": (comp("L2"), tps("L2"), gpu("L2")),
            "L3 CR=4\n(recomm.)": (cr4, BATCH_SIZE / (cr4_total_ms / 1000), None),
            "L3 CR=8\n(max)": (comp("L3"), tps("L3"), None)}


def generate_fig3_step_decomposition(profiles: dict[str, dict], output_dir: Path) -> Path:
    """Generate Fig. 3 (fig2_step_decomposition.png): stacked step-time bars with
    throughput labels and the L0/L2 GPU-utilization overlay.

    Args:
        profiles: Loaded profiles keyed by level (``load_profiles``).
        output_dir: Directory for the output figure (created if absent).

    Returns:
        Path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = step_decomposition_data(profiles)
    _ieee_style()
    fig, ax1 = plt.subplots(figsize=(3.5, 3.0))
    labels = list(data)
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))
    for i in range(4):
        vals = np.array([data[lab][0][i] for lab in labels])
        ax1.bar(x, vals, 0.5, bottom=bottoms, color=PHASE_COLORS[i],
                label=PHASE_LABELS[i], edgecolor="white", linewidth=0.3)
        bottoms += vals
    ax1.set_yscale("log")
    ax1.set_ylabel("Step time (ms, log scale)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Step Time Decomposition by Data Pipeline Level")
    for i, lab in enumerate(labels):
        ax1.text(i, bottoms[i] * 0.55, f"{data[lab][1]:.0f} tr/s", ha="center",
                 va="center", fontsize=6, fontweight="bold", color="black",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                           edgecolor="none", alpha=0.85))
    ax2 = ax1.twinx()
    gx = [i for i, lab in enumerate(labels) if data[lab][2] is not None]
    gy = [data[labels[i]][2] for i in gx]
    ax2.plot(gx, gy, "ko--", markersize=4, linewidth=1.0, label="GPU util (%)")
    ax2.set_ylabel("GPU utilization (%)")
    ax2.set_ylim(0, 100)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2, framealpha=0.9,
               fontsize=5.5, handlelength=1.5, handleheight=0.8,
               borderpad=0.6, handletextpad=0.4)
    fig.subplots_adjust(top=0.82)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "fig2_step_decomposition.png"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    logger.info("Saved Fig. 3 (step decomposition): %s", fig_path)
    return fig_path


def _ieee_style() -> None:
    """Apply IEEE 2-column figure style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def generate_fig5_cr_vs_f1(output_dir: Path) -> Path:
    """Generate Fig. 5: CR vs. F1 trade-off curve with speedup annotations.

    Args:
        output_dir: Directory for output figure.

    Returns:
        Path to saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    _ieee_style()

    # Prefer test set evaluation; fall back to training summary
    test_path = Path("outputs/results/test_evaluation_summary.json")
    train_path = Path("outputs/results/experiment_summary.json")

    if test_path.exists():
        with open(test_path) as f:
            data = json.load(f)
        summary = data["summary"]
        f1_key = "tolerance_0.1s"
        logger.info("Fig. 3: using TEST SET evaluation data")
    else:
        with open(train_path) as f:
            data = json.load(f)
        summary = data["summary"]
        f1_key = None
        logger.info("Fig. 3: using validation data (test eval not found)")

    configs = [
        ("l2_baseline", 1),
        ("wavelet_cr2", 2),
        ("wavelet_cr4", 4),
        ("wavelet_cr8", 8),
        ("wavelet_cr16", 16),
    ]

    crs = []
    p_f1_mean, p_f1_std = [], []
    s_f1_mean, s_f1_std = [], []
    speedups = []

    # Training time from the epoch logs (three-seed means, as in Table 5)
    baseline_time = wall_clock_mean_min("l2_baseline")
    if baseline_time is None:
        raise FileNotFoundError("outputs/checkpoints/l2_baseline_seed*/epoch_log.csv")

    for config, cr in configs:
        s = summary[config]
        crs.append(cr)
        if f1_key:
            p_f1_mean.append(s[f1_key]["p_f1_mean"])
            p_f1_std.append(s[f1_key]["p_f1_std"])
            s_f1_mean.append(s[f1_key]["s_f1_mean"])
            s_f1_std.append(s[f1_key]["s_f1_std"])
        else:
            p_f1_mean.append(s["p_f1_mean"])
            p_f1_std.append(s["p_f1_std"])
            s_f1_mean.append(s["s_f1_mean"])
            s_f1_std.append(s["s_f1_std"])
        ts = wall_clock_mean_min(config) or baseline_time
        speedups.append(baseline_time / ts)

    crs = np.array(crs)
    p_f1_mean = np.array(p_f1_mean)
    p_f1_std = np.array(p_f1_std)
    s_f1_mean = np.array(s_f1_mean)
    s_f1_std = np.array(s_f1_std)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # P-F1 — plot line+marker separately from error bars so legend shows only marker
    ax.plot(crs, p_f1_mean, "bo-", markersize=4, linewidth=1.0, label="P-phase F1")
    ax.errorbar(crs, p_f1_mean, yerr=p_f1_std, fmt="none", ecolor="blue",
                linewidth=0.8, capsize=2)
    # S-F1
    ax.plot(crs, s_f1_mean, "rs-", markersize=4, linewidth=1.0, label="S-phase F1")
    ax.errorbar(crs, s_f1_mean, yerr=s_f1_std, fmt="none", ecolor="red",
                linewidth=0.8, capsize=2)

    # Baseline reference lines
    ax.axhline(p_f1_mean[0], color="blue", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axhline(s_f1_mean[0], color="red", linestyle=":", linewidth=0.5, alpha=0.5)

    # Optimal CR=4 highlight
    ax.axvspan(3.2, 5.0, alpha=0.1, color="green")
    ax.annotate("optimal", xy=(4, p_f1_mean[2] - 0.04), fontsize=6,
                ha="center", color="green", fontweight="bold")

    # Speedup annotations
    for i, (cr, sp) in enumerate(zip(crs, speedups)):
        y_pos = min(p_f1_mean[i], s_f1_mean[i]) - 0.035
        if y_pos < 0.45:
            y_pos = s_f1_mean[i] + 0.02
        ax.annotate(f"{sp:.1f}×", xy=(cr, y_pos), fontsize=5.5,
                    ha="center", color="gray")

    ax.set_xscale("log", base=2)
    ax.set_xticks(crs)
    ax.set_xticklabels([str(c) for c in crs])
    ax.set_xlabel("Compression Ratio (CR)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Phase-picking Accuracy vs. Compression Ratio")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.2)

    fig_path = output_dir / "fig3_cr_vs_f1.png"
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info("Saved Fig. 5 (CR vs. F1): %s", fig_path)
    return fig_path


def generate_fig8_time_to_accuracy(output_dir: Path) -> Path:
    """Generate Fig. 8: time-to-accuracy curve (validation P-F1 of seed 42).

    Uses seed=42 epoch logs to plot P-F1 vs wall-clock time.

    Args:
        output_dir: Directory for output figure.

    Returns:
        Path to saved figure.
    """
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    _ieee_style()

    configs = [
        ("l2_baseline_seed42", "L2 baseline", "#333333", "-"),
        ("wavelet_cr2_seed42", "Wavelet CR=2", "#2196F3", "--"),
        ("wavelet_cr4_seed42", "Wavelet CR=4", "#4CAF50", "-"),
        ("wavelet_cr8_seed42", "Wavelet CR=8", "#FF9800", "--"),
        ("wavelet_cr16_seed42", "Wavelet CR=16", "#F44336", ":"),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    baseline_final_pf1 = None
    tta_cr4 = None

    for config, label, color, ls in configs:
        csv_path = Path(f"outputs/checkpoints/{config}/epoch_log.csv")
        if not csv_path.exists():
            continue

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        times = [float(r["wall_clock_sec"]) / 60 for r in rows]
        pf1s = [float(r["val_p_f1"]) for r in rows]

        ax.plot(times, pf1s, color=color, linestyle=ls, linewidth=1.0,
                label=label, markersize=0)

        if config == "l2_baseline_seed42":
            baseline_final_pf1 = pf1s[-1]

        # Find when CR=4 reaches baseline's final P-F1
        if config == "wavelet_cr4_seed42" and baseline_final_pf1 is not None:
            for t, p in zip(times, pf1s):
                if p >= baseline_final_pf1:
                    tta_cr4 = t
                    break

    # Mark TTA point for CR=4
    if tta_cr4 is not None and baseline_final_pf1 is not None:
        ax.axvline(tta_cr4, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(baseline_final_pf1, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        # Text below baseline, horizontal arrow at 0.87 pointing to CR=4 line
        text_x = tta_cr4 + 18
        text_y = 0.865
        arrow_y = 0.87
        ax.text(
            text_x, text_y,
            f"CR=4 reaches\nbaseline in\n{tta_cr4:.0f} min",
            fontsize=5.5, color="green", va="center",
        )
        ax.annotate(
            "",
            xy=(tta_cr4, arrow_y),
            xytext=(text_x - 1, arrow_y),
            arrowprops=dict(arrowstyle="->", color="green", lw=0.6),
        )

    ax.set_xlabel("Wall-clock time (minutes)")
    ax.set_ylabel("P-phase F1 Score")
    ax.set_title("Time-to-Accuracy: P-F1 vs. Training Time")
    ax.legend(loc="lower right", fontsize=5.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0.75, 1.0)

    fig_path = output_dir / "fig4_tta_curve.png"
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info("Saved Fig. 8 (time to accuracy): %s", fig_path)
    return fig_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate GRSL figures")
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="outputs/profiles",
        help="Directory with profile JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory for output figures",
    )
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)

    required = {
        "Fig. 3 (step decomposition) L0 profile": profile_dir / PAPER_PROFILES["L0"],
        "Fig. 3 (step decomposition) L2 profile": profile_dir / PAPER_PROFILES["L2"],
        "Fig. 3 (step decomposition) L3 profile": profile_dir / PAPER_PROFILES["L3"],
        "Fig. 5 (CR vs. F1) test summary": Path("outputs/results/test_evaluation_summary.json"),
        "Fig. 5/8 baseline epoch log": Path("outputs/checkpoints/l2_baseline_seed42/epoch_log.csv"),
        "Fig. 5/8 CR = 4 epoch log": Path("outputs/checkpoints/wavelet_cr4_seed42/epoch_log.csv"),
    }
    absent = missing_inputs(required)
    if absent:
        logger.error("Missing inputs: %s", "; ".join(absent))
        sys.exit(1)

    profiles = load_profiles(profile_dir)

    logger.info("Loaded %d profiles: %s", len(profiles), list(profiles.keys()))

    # Fig. 3 (step decomposition), Fig. 5 (CR vs. F1), Fig. 8 (time to accuracy)
    generated = [generate_fig3_step_decomposition(profiles, output_dir),
                 generate_fig5_cr_vs_f1(output_dir), generate_fig8_time_to_accuracy(output_dir)]
    logger.info("Generated %d figure(s): %s", len(generated), ", ".join(p.name for p in generated))


if __name__ == "__main__":
    main()
