"""Evaluation metrics for seismic phase picking.

Implements Münchmeyer et al. (2022) evaluation framework.

Reference:
    Münchmeyer et al. (2022): "Which picker fits my data?", JGR 127.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


def pick_peaks(
    prob_array: np.ndarray,
    threshold: float = 0.3,
    min_distance: int = 100,
) -> dict[str, list[int]]:
    """Detect P and S phase peaks from model probability output.

    Args:
        prob_array: (3, T) array with channels [Noise, P, S].
        threshold: Minimum peak height.
        min_distance: Minimum samples between peaks.

    Returns:
        Dict with 'P' and 'S' keys, each a list of peak sample indices.
    """
    result = {}
    for ch, name in [(1, "P"), (2, "S")]:
        peaks, _ = find_peaks(
            prob_array[ch], height=threshold, distance=min_distance
        )
        result[name] = peaks.tolist()
    return result


def _match_peaks(
    pred_peaks: list[int],
    true_peaks: list[int],
    tolerance: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match predicted peaks to true peaks within tolerance (greedy nearest-first)."""
    if not pred_peaks or not true_peaks:
        return [], list(pred_peaks), list(true_peaks)

    pred_sorted = sorted(pred_peaks)
    true_remaining = sorted(true_peaks)
    matched = []
    unmatched_pred = []

    for p in pred_sorted:
        best_idx = -1
        best_dist = tolerance + 1
        for i, t in enumerate(true_remaining):
            dist = abs(p - t)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            matched.append((p, true_remaining.pop(best_idx)))
        else:
            unmatched_pred.append(p)

    return matched, unmatched_pred, list(true_remaining)


def compute_phase_metrics(
    pred_peaks: list[int],
    true_peaks: list[int],
    tolerance_samples: int = 10,
) -> dict[str, float]:
    """Compute detection metrics: TP, FP, FN, Precision, Recall, F1.

    Args:
        pred_peaks: Predicted peak sample indices.
        true_peaks: Ground truth peak sample indices.
        tolerance_samples: Max distance for match (10=0.1s@100Hz, 50=0.5s).

    Returns:
        Dict with tp, fp, fn, precision, recall, f1.
    """
    matched, unmatched_pred, unmatched_true = _match_peaks(
        pred_peaks, true_peaks, tolerance_samples
    )
    tp = len(matched)
    fp = len(unmatched_pred)
    fn = len(unmatched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def compute_pick_residuals(
    pred_peaks: list[int],
    true_peaks: list[int],
    tolerance_samples: int = 50,
    sample_rate: int = 100,
) -> dict[str, float]:
    """Compute onset timing residuals (seconds) for matched picks.

    Returns:
        Dict with mean_sec (μ), std_sec (σ), mae_sec, n_matched.
    """
    matched, _, _ = _match_peaks(pred_peaks, true_peaks, tolerance_samples)

    if not matched:
        return {"mean_sec": float("nan"), "std_sec": float("nan"),
                "mae_sec": float("nan"), "n_matched": 0}

    residuals = np.array([p - t for p, t in matched], dtype=np.float64)
    residuals_sec = residuals / sample_rate

    return {
        "mean_sec": float(residuals_sec.mean()),
        "std_sec": float(residuals_sec.std()),
        "mae_sec": float(np.abs(residuals_sec).mean()),
        "n_matched": len(matched),
    }
