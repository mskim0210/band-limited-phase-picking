"""Generate Fig. 3 (per-step time decomposition) with CR=4 and CR=8 bars.

The CR=4 bar uses the per-step time re-measured from training steps
(9.3 ms), with the I/O and H2D components taken from the shape-identical
L3 profile. GPU utilization is shown for L0/L2 only.

Usage:
    python scripts/make_fig2_cr4.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

OUT = [Path("outputs/figures/fig2_step_decomposition.png")]

CR4_TOTAL_MS = 9.3  # measured from training-step wall-clock (epoch logs);
                    # entered as a constant, not derived from the stored L3 profile


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof = {}
    for name, f in [("L0", "outputs/profiles/l0_stead_hdf5.json"),
                    ("L2", "outputs/profiles/l2_stead_memmap.json"),
                    ("CR8", "outputs/profiles/l3_cs_gaussian_cr8.json")]:
        prof[name] = json.load(open(f))["summary"]

    def comp(s):
        return [s[k]["mean"] * 1000 for k in
                ("t_io", "t_preprocess", "t_h2d", "t_compute")]

    data = {
        "L0\n(HDF5)": (comp(prof["L0"]),
                       prof["L0"]["throughput_traces_per_sec"],
                       prof["L0"]["gpu_util_mean_pct"]),
        "L2\n(memmap)": (comp(prof["L2"]),
                         prof["L2"]["throughput_traces_per_sec"],
                         prof["L2"]["gpu_util_mean_pct"]),
    }
    # CR=4 bar: I/O + H2D from the shape-identical L3 profile,
    # compute = measured total minus those components.
    io4, pre4, h2d4 = 0.10, 0.0, 0.06
    data["L3 CR=4\n(recomm.)"] = ([io4, pre4, h2d4,
                                   CR4_TOTAL_MS - io4 - pre4 - h2d4],
                                  64 / (CR4_TOTAL_MS / 1000), None)
    # GPU-util overlay restricted to L0/L2 (per-level values in Table 3);
    # utilization was not cleanly re-measured for the L3 bars.
    data["L3 CR=8\n(max)"] = (comp(prof["CR8"]),
                              prof["CR8"]["throughput_traces_per_sec"],
                              None)

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    phase_labels = ["I/O", "Preprocess", "H2D", "Compute"]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

    fig, ax1 = plt.subplots(figsize=(3.5, 3.0))
    labels = list(data.keys())
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))
    for i in range(4):
        vals = np.array([data[l][0][i] for l in labels])
        ax1.bar(x, vals, 0.5, bottom=bottoms, color=colors[i],
                label=phase_labels[i], edgecolor="white", linewidth=0.3)
        bottoms += vals

    ax1.set_yscale("log")
    ax1.set_ylabel("Step time (ms, log scale)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Step Time Decomposition by Data Pipeline Level")

    for i, l in enumerate(labels):
        tp = data[l][1]
        ax1.text(i, bottoms[i] * 0.55, f"{tp:.0f} tr/s", ha="center",
                 va="center", fontsize=6, fontweight="bold", color="black",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                           edgecolor="none", alpha=0.85))

    ax2 = ax1.twinx()
    gx = [i for i, l in enumerate(labels) if data[l][2] is not None]
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
    for p in OUT:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", pad_inches=0.08)
        logger.info("saved %s", p)


if __name__ == "__main__":
    main()
