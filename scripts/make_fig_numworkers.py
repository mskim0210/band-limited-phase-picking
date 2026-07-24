"""Regenerate fig_num_workers.png with clean (non-LaTeX-escaped) labels.

Reads outputs/profiles/exp3_num_workers_sweep.json.

Usage:
    python scripts/make_fig_numworkers.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = [Path("outputs/figures/fig_num_workers.png")]


def main() -> None:
    d = json.load(open("outputs/profiles/exp3_num_workers_sweep.json"))
    nws = d["worker_settings"]
    x = range(len(nws))

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for level, style, label in (("L2", "o-", "L2 (memmap)"),
                                ("L0", "s--", "L0 (HDF5)")):
        y = [d["results"][level][str(n)]["throughput_traces_per_sec"]
             for n in nws]
        ax.plot(x, y, style, markersize=4, linewidth=1.2, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(n) for n in nws])
    ax.set_xlabel("DataLoader workers")
    ax.set_ylabel("Training throughput (traces/s)")
    ax.set_title("Training throughput versus DataLoader worker count")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for p in OUT:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", pad_inches=0.05)
        print("saved", p)


if __name__ == "__main__":
    main()
