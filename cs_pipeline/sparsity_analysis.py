"""Sparsity analysis of STEAD waveforms in Wavelet and DCT domains.

Quantifies the compressibility of seismic waveforms in the wavelet and
DCT domains: sorted coefficient-magnitude decay and top-|c| energy
retention (Fig. 1).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pywt
except ImportError as e:
    raise ImportError(
        "PyWavelets is required for sparsity analysis. "
        "Install with: pip install PyWavelets"
    ) from e
from scipy.fft import dct

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core transform functions
# ---------------------------------------------------------------------------


def wavelet_coefficients(
    trace: np.ndarray,
    wavelet: str = "db4",
    level: int = 6,
) -> np.ndarray:
    """Wavelet transform followed by magnitude sort (descending).

    Args:
        trace: (N,) 1D waveform array.
        wavelet: Wavelet name for pywt. Default: 'db4'.
        level: Decomposition level. Default: 6.

    Returns:
        (K,) sorted magnitudes in descending order,
        where K is the total number of wavelet coefficients.
    """
    coeffs = pywt.wavedec(trace, wavelet, level=level, mode="symmetric")
    all_coeffs = np.concatenate([c.ravel() for c in coeffs])
    magnitudes = np.abs(all_coeffs)
    magnitudes.sort()
    return magnitudes[::-1].copy()


def dct_coefficients(trace: np.ndarray) -> np.ndarray:
    """DCT type-II transform followed by magnitude sort (descending).

    Args:
        trace: (N,) 1D waveform array.

    Returns:
        (N,) sorted magnitudes in descending order.
    """
    coeffs = dct(trace, type=2, norm="ortho")
    magnitudes = np.abs(coeffs)
    magnitudes.sort()
    return magnitudes[::-1].copy()


def energy_retention(
    sorted_magnitudes: np.ndarray,
    kept_ratios: list[float],
) -> list[float]:
    """Compute energy retention for given kept ratios.

    Energy is defined as sum of squared magnitudes (L2 norm squared).

    Args:
        sorted_magnitudes: (K,) magnitudes in descending order.
        kept_ratios: List of ratios in (0, 1], e.g. [0.05, 0.10, ...].

    Returns:
        List of retention percentages (0-100), same length as kept_ratios.
    """
    squared = sorted_magnitudes ** 2
    total_energy = squared.sum()
    if total_energy < 1e-30:
        return [0.0] * len(kept_ratios)

    cumulative = np.cumsum(squared)
    k_total = len(sorted_magnitudes)
    retentions = []
    for ratio in kept_ratios:
        k = max(1, int(k_total * ratio))
        k = min(k, k_total)
        retentions.append(float(cumulative[k - 1] / total_energy * 100.0))
    return retentions


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_sparsity(
    waveforms: np.ndarray,
    n_traces: int = 10_000,
    seed: int = 42,
    wavelet: str = "db4",
    wavelet_level: int = 6,
    kept_ratios: list[float] | None = None,
    compression_ratios: list[int] | None = None,
) -> dict[str, Any]:
    """Comprehensive sparsity analysis of STEAD waveforms.

    Args:
        waveforms: Memmap array, shape (N_total, 3, 6000).
        n_traces: Number of traces to sample. Default: 10000.
        seed: Random seed for reproducibility. Default: 42.
        wavelet: Wavelet name. Default: 'db4'.
        wavelet_level: Wavelet decomposition level. Default: 6.
        kept_ratios: Ratios for energy retention analysis.
            Default: [0.05, 0.10, 0.15, 0.20, 0.25, 0.50].
        compression_ratios: CRs for energy analysis.
            Default: [4, 8, 16, 32].

    Returns:
        Dictionary with keys: metadata, decay_curves, energy_retention,
        cr_retention, key_findings.
    """
    kept_ratios = kept_ratios or [0.05, 0.10, 0.15, 0.20, 0.25, 0.50]
    compression_ratios = compression_ratios or [4, 8, 16, 32]
    cr_kept_ratios = [1.0 / cr for cr in compression_ratios]

    n_total = len(waveforms)
    if n_traces > n_total:
        logger.warning(
            "n_traces=%d > total=%d, using all traces", n_traces, n_total
        )
        n_traces = n_total

    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, n_traces, replace=False)

    n_channels = 3
    n_samples = waveforms.shape[2] if waveforms.ndim == 3 else 6000

    # Storage for per-trace results
    # Wavelet coefficient count may differ from n_samples
    wav_test = wavelet_coefficients(np.zeros(n_samples), wavelet, wavelet_level)
    n_wav_coeffs = len(wav_test)

    wav_decay_all = np.zeros((n_traces * n_channels, n_wav_coeffs), dtype=np.float64)
    dct_decay_all = np.zeros((n_traces * n_channels, n_samples), dtype=np.float64)
    wav_retention_all = np.zeros((n_traces * n_channels, len(kept_ratios)), dtype=np.float64)
    dct_retention_all = np.zeros((n_traces * n_channels, len(kept_ratios)), dtype=np.float64)
    wav_cr_all = np.zeros((n_traces * n_channels, len(compression_ratios)), dtype=np.float64)
    dct_cr_all = np.zeros((n_traces * n_channels, len(compression_ratios)), dtype=np.float64)

    # Per-channel storage indices
    ch_indices = {c: [] for c in range(n_channels)}

    skipped = 0
    row = 0
    for i, idx in enumerate(indices):
        trace_data = waveforms[idx]  # (3, 6000)
        if isinstance(trace_data, np.memmap):
            trace_data = np.array(trace_data)

        for c in range(n_channels):
            ch_trace = trace_data[c].astype(np.float64)

            # Skip NaN/Inf or zero-energy traces
            if not np.isfinite(ch_trace).all() or np.abs(ch_trace).sum() < 1e-30:
                skipped += 1
                continue

            wav_mags = wavelet_coefficients(ch_trace, wavelet, wavelet_level)
            dct_mags = dct_coefficients(ch_trace)

            # Normalize decay by max for averaging
            wav_max = wav_mags[0] if wav_mags[0] > 0 else 1.0
            dct_max = dct_mags[0] if dct_mags[0] > 0 else 1.0

            wav_decay_all[row, :len(wav_mags)] = wav_mags / wav_max
            dct_decay_all[row, :len(dct_mags)] = dct_mags / dct_max

            wav_retention_all[row] = energy_retention(wav_mags, kept_ratios)
            dct_retention_all[row] = energy_retention(dct_mags, kept_ratios)
            wav_cr_all[row] = energy_retention(wav_mags, cr_kept_ratios)
            dct_cr_all[row] = energy_retention(dct_mags, cr_kept_ratios)

            ch_indices[c].append(row)
            row += 1

        if (i + 1) % 2000 == 0:
            logger.info("Processed %d/%d traces", i + 1, n_traces)

    if skipped > 0:
        logger.warning("Skipped %d invalid channels (NaN/Inf/zero-energy)", skipped)

    # Trim to actual rows
    valid = row
    wav_decay_all = wav_decay_all[:valid]
    dct_decay_all = dct_decay_all[:valid]
    wav_retention_all = wav_retention_all[:valid]
    dct_retention_all = dct_retention_all[:valid]
    wav_cr_all = wav_cr_all[:valid]
    dct_cr_all = dct_cr_all[:valid]

    logger.info("Analysis complete: %d valid channels from %d traces", valid, n_traces)

    # -- Build results dict --
    def _stats(arr: np.ndarray) -> dict[str, list[float]]:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
        }

    def _ch_stats(arr: np.ndarray, ch_idx: dict) -> dict[str, dict]:
        result = {}
        for c in range(n_channels):
            rows = ch_idx[c]
            if rows:
                result[f"ch{c}"] = _stats(arr[rows])
        return result

    results: dict[str, Any] = {
        "metadata": {
            "n_traces": n_traces,
            "n_valid_channels": valid,
            "n_skipped": skipped,
            "n_samples": n_samples,
            "n_channels": n_channels,
            "n_wav_coeffs": n_wav_coeffs,
            "wavelet": wavelet,
            "wavelet_level": wavelet_level,
            "dct_type": 2,
            "seed": seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "decay_curves": {
            "wavelet": {**_stats(wav_decay_all), "per_channel": _ch_stats(wav_decay_all, ch_indices)},
            "dct": {**_stats(dct_decay_all), "per_channel": _ch_stats(dct_decay_all, ch_indices)},
        },
        "energy_retention": {
            "kept_ratios": kept_ratios,
            "wavelet": {**_stats(wav_retention_all), "per_channel": _ch_stats(wav_retention_all, ch_indices)},
            "dct": {**_stats(dct_retention_all), "per_channel": _ch_stats(dct_retention_all, ch_indices)},
        },
        "cr_retention": {
            "compression_ratios": compression_ratios,
            "wavelet": _stats(wav_cr_all),
            "dct": _stats(dct_cr_all),
        },
        "key_findings": {},
    }

    # Key findings
    wav_ret_mean = wav_retention_all.mean(axis=0)
    dct_ret_mean = dct_retention_all.mean(axis=0)
    idx_20 = kept_ratios.index(0.20) if 0.20 in kept_ratios else -1

    wav_20 = float(wav_ret_mean[idx_20]) if idx_20 >= 0 else 0.0
    dct_20 = float(dct_ret_mean[idx_20]) if idx_20 >= 0 else 0.0

    wav_cr_mean = wav_cr_all.mean(axis=0)
    dct_cr_mean = dct_cr_all.mean(axis=0)
    idx_cr8 = compression_ratios.index(8) if 8 in compression_ratios else -1

    results["key_findings"] = {
        "wavelet_20pct_energy": round(wav_20, 2),
        "dct_20pct_energy": round(dct_20, 2),
        "cs_justified": bool(wav_20 >= 95.0 or dct_20 >= 95.0),
        "best_transform": "wavelet" if wav_20 >= dct_20 else "dct",
        "cr8_wavelet_energy": round(float(wav_cr_mean[idx_cr8]), 2) if idx_cr8 >= 0 else 0.0,
        "cr8_dct_energy": round(float(dct_cr_mean[idx_cr8]), 2) if idx_cr8 >= 0 else 0.0,
    }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sparsity(
    results: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Generate publication-quality sparsity figures.

    Produces:
        1. fig_decay.png — Sorted coefficient magnitude decay (log scale)
        2. fig_energy_retention.png — Energy retention vs kept ratio

    Args:
        results: Return value of analyze_sparsity().
        output_dir: Directory for figure output.

    Returns:
        List of saved figure paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # IEEE 2-column style
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    # -- Fig 1: Decay curve --
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    wav_mean = np.array(results["decay_curves"]["wavelet"]["mean"])
    wav_std = np.array(results["decay_curves"]["wavelet"]["std"])
    dct_mean = np.array(results["decay_curves"]["dct"]["mean"])
    dct_std = np.array(results["decay_curves"]["dct"]["std"])

    x_wav = np.linspace(0, 1, len(wav_mean))
    x_dct = np.linspace(0, 1, len(dct_mean))

    ax.semilogy(x_wav, wav_mean, "b-", linewidth=0.8, label="Wavelet (db4, L=6)")
    ax.fill_between(
        x_wav,
        np.maximum(wav_mean - wav_std, 1e-10),
        wav_mean + wav_std,
        alpha=0.2, color="blue",
    )
    ax.semilogy(x_dct, dct_mean, color="orange", linestyle="--", linewidth=0.8, label="DCT (type-II)")
    ax.fill_between(
        x_dct,
        np.maximum(dct_mean - dct_std, 1e-10),
        dct_mean + dct_std,
        alpha=0.2, color="orange",
    )

    ax.axvline(0.20, color="gray", linestyle=":", linewidth=0.6, label="20% mark")
    ax.axvline(1.0 / 8.0, color="red", linestyle=":", linewidth=0.6, label="CR=8 (12.5%)")

    ax.set_xlabel("Coefficient index (normalized)")
    ax.set_ylabel("Normalized magnitude")
    ax.set_title("STEAD Waveform Sparsity: Coefficient Magnitude Decay")
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=1e-6)
    ax.annotate("(a)", xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -30), textcoords="offset points",
                ha="center", va="top", fontsize=9,
                annotation_clip=False)

    path = output_dir / "fig_decay.png"
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)
    logger.info("Saved %s", path)

    # -- Fig 2: Energy retention --
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ratios = np.array(results["energy_retention"]["kept_ratios"]) * 100
    wav_ret = np.array(results["energy_retention"]["wavelet"]["mean"])
    wav_ret_std = np.array(results["energy_retention"]["wavelet"]["std"])
    dct_ret = np.array(results["energy_retention"]["dct"]["mean"])
    dct_ret_std = np.array(results["energy_retention"]["dct"]["std"])

    ax.plot(ratios, wav_ret, "bo-", markersize=3, linewidth=0.8, label="Wavelet (db4)")
    ax.plot(ratios, dct_ret, "s--", color="orange", markersize=3, linewidth=0.8,
            label="DCT (type-II)")

    ax.axhline(95, color="gray", linestyle=":", linewidth=0.6, label="95% threshold")

    # CR markers on the curves — add one to legend
    cr_list = results["cr_retention"]["compression_ratios"]
    wav_cr_ret = np.array(results["cr_retention"]["wavelet"]["mean"])
    dct_cr_ret = np.array(results["cr_retention"]["dct"]["mean"])
    cr_x = [100.0 / cr for cr in cr_list]
    ax.scatter(cr_x[:1], wav_cr_ret[:1], marker="D", s=20, color="blue", zorder=5,
               label="CR fractions (top-|c| energy)")
    ax.scatter(cr_x[1:], wav_cr_ret[1:], marker="D", s=20, color="blue", zorder=5)
    ax.scatter(cr_x, dct_cr_ret, marker="D", s=20, color="orange", zorder=5)

    # CR secondary axis (top) — only show CR=4,8,16 to avoid overlap
    ax2 = ax.twiny()
    cr_show = [cr for cr in cr_list if cr <= 16]
    cr_positions = [100.0 / cr for cr in cr_show]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(cr_positions)
    ax2.set_xticklabels([f"CR={cr}" for cr in cr_show])
    ax2.tick_params(axis="x", labelsize=6)

    ax.set_xlabel("Kept coefficients (%)")
    ax.set_ylabel("Retained energy (%)")
    ax.set_title("Energy Retention vs. Compression")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_ylim(48, 103)
    ax.annotate("(b)", xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -30), textcoords="offset points",
                ha="center", va="top", fontsize=9,
                annotation_clip=False)

    path = output_dir / "fig_energy_retention.png"
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)
    logger.info("Saved %s", path)

    return saved
