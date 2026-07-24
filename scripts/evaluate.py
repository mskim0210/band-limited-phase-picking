"""Evaluate best checkpoints on STEAD test set.

Generates the test-set numbers for Tables 5 and 8.
Supports single checkpoint, batch-all (21 runs), and SNR-stratified evaluation.

Usage:
    # Single checkpoint
    python scripts/evaluate.py --checkpoint outputs/checkpoints/wavelet_cr4_seed42/best_model.pt

    # All 21 runs + summary
    python scripts/evaluate.py --batch-all

    # With SNR stratification
    python scripts/evaluate.py --batch-all --snr-stratified
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs_pipeline.cs_aware_model import CSPhaseNet
from cs_pipeline.cs_dataset import CompressedSTEADDataset
from cs_pipeline.metrics import compute_phase_metrics, compute_pick_residuals, pick_peaks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# INSTANCE cross-domain dataset
# ---------------------------------------------------------------------------


class InstanceCropDataset(torch.utils.data.Dataset):
    """INSTANCE dataset with P-arrival-centered crop to (3, 6000).

    Crops 120s traces to 60s centered on P-arrival.
    Optionally applies CS encoding (wavelet/DCT) for L3 evaluation.

    Args:
        split: 'test' (default for cross-domain eval).
        sigma: Gaussian label width in samples.
        cr: Compression ratio (None for baseline M=6000).
        matrix_type: 'wavelet' or 'dct' (None for baseline).
    """

    def __init__(
        self,
        split: str = "test",
        sigma: float = 10.0,
        cr: int | None = None,
        matrix_type: str | None = None,
    ) -> None:
        import seisbench.data as sbd

        self._instance = sbd.InstanceCounts()
        mask = self._instance.metadata["split"] == split
        self._indices = mask[mask].index.tolist()
        self._sigma = sigma
        self._target_len = 6000  # 60s @ 100Hz
        self._matrix = None

        if cr is not None and matrix_type is not None:
            from cs_pipeline.sensing_matrix import create_sensing_matrix
            self._matrix = create_sensing_matrix(matrix_type, N=6000, CR=cr)

        logger.info(
            "InstanceCropDataset(%s): %d traces, CR=%s, matrix=%s",
            split, len(self._indices), cr, matrix_type,
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = self._indices[idx]
        wf, meta = self._instance.get_sample(real_idx)
        wf = wf.astype(np.float32)
        total_len = wf.shape[1]

        # Get P-arrival for centering
        p_raw = meta.get("trace_P_arrival_sample", np.nan)
        try:
            p_sample = int(float(p_raw)) if not (isinstance(p_raw, float) and np.isnan(p_raw)) else -1
        except (ValueError, TypeError):
            p_sample = -1

        s_raw = meta.get("trace_S_arrival_sample", np.nan)
        try:
            s_sample = int(float(s_raw)) if not (isinstance(s_raw, float) and np.isnan(s_raw)) else -1
        except (ValueError, TypeError):
            s_sample = -1

        # Crop: P-arrival centered, or middle if no P
        half = self._target_len // 2
        if p_sample >= 0:
            center = p_sample
        else:
            center = total_len // 2

        start = max(0, center - half)
        end = start + self._target_len
        if end > total_len:
            end = total_len
            start = max(0, end - self._target_len)

        cropped = wf[:, start:end]

        # Pad if shorter (shouldn't happen for INSTANCE 12000 → 6000)
        if cropped.shape[1] < self._target_len:
            pad = np.zeros((3, self._target_len), dtype=np.float32)
            pad[:, :cropped.shape[1]] = cropped
            cropped = pad

        # Adjust arrival samples relative to crop
        if p_sample >= 0:
            p_sample = p_sample - start
        if s_sample >= 0:
            s_sample = s_sample - start

        # Std normalization
        std = cropped.std()
        if std > 1e-8:
            cropped /= std

        # CS encoding if applicable
        wf_tensor = torch.from_numpy(cropped)
        if self._matrix is not None:
            wf_tensor = self._matrix.encode(wf_tensor.unsqueeze(0)).squeeze(0)

        # Label at 6000 length (original domain)
        label = self._make_label(
            p_sample if 0 <= p_sample < self._target_len else None,
            s_sample if 0 <= s_sample < self._target_len else None,
        )

        return wf_tensor, torch.from_numpy(label)

    def _make_label(self, p, s, n=6000):
        label = np.zeros((3, n), dtype=np.float32)
        x = np.arange(n, dtype=np.float32)
        if p is not None:
            label[1] = np.exp(-((x - p) ** 2) / (2 * self._sigma ** 2))
        if s is not None:
            label[2] = np.exp(-((x - s) ** 2) / (2 * self._sigma ** 2))
        label[0] = np.clip(1.0 - label[1] - label[2], 0.0, 1.0)
        return label


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    cfg: object,
    tolerances: list[int] | None = None,
    snr_stratified: bool = False,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    """Evaluate a single checkpoint on STEAD test set.

    Args:
        checkpoint_path: Path to best_model.pt.
        cfg: OmegaConf config.
        tolerances: Tolerance in samples. Default: [10, 50] (0.1s, 0.5s).
        snr_stratified: Whether to compute SNR-stratified metrics.
        batch_size: Batch size for inference.
        num_workers: DataLoader workers.

    Returns:
        Evaluation results dict.
    """
    checkpoint_path = Path(checkpoint_path)
    tolerances = tolerances or [10, 50]

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    M = config["M"]
    N = config.get("N", 6000)
    matrix_type = config.get("matrix_type", "wavelet")
    cr = config.get("cr", 4)

    logger.info(
        "Evaluating: %s (M=%d, CR=%d, matrix=%s)",
        checkpoint_path.parent.name, M, cr, matrix_type,
    )

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("arch", "csphasenet") == "phasenet":
        from cs_pipeline.phasenet_wrapper import SeisBenchPhaseNet
        model = SeisBenchPhaseNet(M=M, N=N)
    else:
        model = CSPhaseNet(M=M, N=N, base_channels=config.get("base_channels", 16),
                           dilation=config.get("dilation", 1))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Build dataset
    dataset_name = getattr(cfg, "_dataset", "stead")
    # Discriminate the true baseline by matrix_type, NOT cr==1: the 'lowpass'
    # control has cr=1 but must be evaluated on its own LP-encoded data.
    is_baseline = (matrix_type == "baseline") or (matrix_type is None) or (cr is None)
    if dataset_name == "instance":
        test_ds = InstanceCropDataset(
            split="test", sigma=10.0, cr=None if is_baseline else cr,
            matrix_type=None if is_baseline else matrix_type,
        )
    elif is_baseline:
        from scripts.profile_baseline import STEADMemmapDataset
        test_ds = STEADMemmapDataset(cfg.data.memmap_dir, "test", sigma=10.0)
    else:
        cs_dir = Path(cfg.data.cs_dir) / f"{matrix_type}_cr{cr}"
        test_ds = CompressedSTEADDataset(cs_dir, "test", sigma=10.0)

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Run inference
    t_start = time.time()

    # Accumulators per tolerance
    tol_counters = {}
    for tol in tolerances:
        tol_counters[tol] = {
            "p_tp": 0, "p_fp": 0, "p_fn": 0,
            "s_tp": 0, "s_fp": 0, "s_fn": 0,
        }

    p_residuals = []
    s_residuals = []
    per_trace_rows = []  # for SNR stratification

    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.to(device, non_blocking=True)
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        labels = y.numpy()

        for i in range(probs.shape[0]):
            pred_picks = pick_peaks(probs[i], threshold=0.3)
            true_picks = pick_peaks(labels[i], threshold=0.5)

            for tol in tolerances:
                pm = compute_phase_metrics(pred_picks["P"], true_picks["P"], tol)
                sm = compute_phase_metrics(pred_picks["S"], true_picks["S"], tol)
                tc = tol_counters[tol]
                tc["p_tp"] += pm["tp"]; tc["p_fp"] += pm["fp"]; tc["p_fn"] += pm["fn"]
                tc["s_tp"] += sm["tp"]; tc["s_fp"] += sm["fp"]; tc["s_fn"] += sm["fn"]

            # Residuals (using largest tolerance for matching)
            max_tol = max(tolerances)
            pr = compute_pick_residuals(pred_picks["P"], true_picks["P"], max_tol)
            sr = compute_pick_residuals(pred_picks["S"], true_picks["S"], max_tol)

            if pr["n_matched"] > 0:
                p_residuals.append(pr["mean_sec"])
            if sr["n_matched"] > 0:
                s_residuals.append(sr["mean_sec"])

            # Per-trace for SNR stratification
            if snr_stratified:
                pm10 = compute_phase_metrics(pred_picks["P"], true_picks["P"], 10)
                sm10 = compute_phase_metrics(pred_picks["S"], true_picks["S"], 10)
                trace_idx = batch_idx * batch_size + i
                per_trace_rows.append({
                    "trace_idx": trace_idx,
                    "p_f1": pm10["f1"],
                    "s_f1": sm10["f1"],
                    "p_mae_sec": pr["mae_sec"],
                    "s_mae_sec": sr["mae_sec"],
                })

        if (batch_idx + 1) % 200 == 0:
            logger.info("  Batch %d/%d", batch_idx + 1, len(test_loader))

    eval_time = time.time() - t_start
    logger.info("  Evaluation done in %.1fs", eval_time)

    # Aggregate results
    result = {
        "config": checkpoint_path.parent.name,
        "checkpoint": str(checkpoint_path),
        "test_traces": len(test_ds),
        "eval_time_sec": round(eval_time, 1),
        "model": {"M": M, "N": N, "cr": cr, "matrix_type": matrix_type},
    }

    for tol in tolerances:
        tc = tol_counters[tol]
        tol_sec = tol / 100.0
        tol_key = f"tolerance_{tol_sec:.1f}s"

        def _f1(tp, fp, fn):
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

        result[tol_key] = {
            "P": _f1(tc["p_tp"], tc["p_fp"], tc["p_fn"]),
            "S": _f1(tc["s_tp"], tc["s_fp"], tc["s_fn"]),
        }

    # Timing residuals
    p_res = np.array(p_residuals) if p_residuals else np.array([])
    s_res = np.array(s_residuals) if s_residuals else np.array([])
    result["timing"] = {
        "P": {
            "mae_sec": round(float(np.abs(p_res).mean()), 4) if len(p_res) > 0 else None,
            "mean_residual_sec": round(float(p_res.mean()), 4) if len(p_res) > 0 else None,
            "std_residual_sec": round(float(p_res.std()), 4) if len(p_res) > 0 else None,
        },
        "S": {
            "mae_sec": round(float(np.abs(s_res).mean()), 4) if len(s_res) > 0 else None,
            "mean_residual_sec": round(float(s_res.mean()), 4) if len(s_res) > 0 else None,
            "std_residual_sec": round(float(s_res.std()), 4) if len(s_res) > 0 else None,
        },
    }

    # SNR stratification
    if snr_stratified and per_trace_rows:
        result["snr_stratified"] = _compute_snr_stratified(per_trace_rows, cfg)

    return result


def _parse_snr(snr_str: str) -> float:
    """Parse STEAD snr_db column: '[56.8 55.4 47.4]' → mean float."""
    try:
        if isinstance(snr_str, (int, float)):
            return float(snr_str)
        s = str(snr_str).strip("[] ")
        values = [float(v) for v in s.split()]
        return float(np.mean(values)) if values else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _compute_snr_stratified(per_trace_rows: list[dict], cfg: object) -> dict:
    """Compute SNR-stratified metrics using STEAD CSV metadata."""
    csv_path = cfg.data.stead_csv
    logger.info("Loading SNR from %s ...", csv_path)

    meta_snr = pd.read_csv(csv_path, usecols=["snr_db"], low_memory=False)

    # Get test set indices (same split as preprocessing)
    n_total = len(meta_snr)
    rng = np.random.RandomState(cfg.seed)
    indices = rng.permutation(n_total)
    n_train = int(n_total * cfg.data.split_ratio[0])
    n_val = int(n_total * cfg.data.split_ratio[1])
    test_indices = indices[n_train + n_val:]

    # Build per-trace DataFrame with SNR
    df = pd.DataFrame(per_trace_rows)
    snr_values = []
    for row in per_trace_rows:
        tidx = row["trace_idx"]
        if tidx < len(test_indices):
            csv_idx = test_indices[tidx]
            snr_values.append(_parse_snr(meta_snr.iloc[csv_idx]["snr_db"]))
        else:
            snr_values.append(float("nan"))

    df["snr_db"] = snr_values
    valid = df["snr_db"].notna().sum()
    logger.info("  SNR mapped: %d/%d traces have valid SNR", valid, len(df))

    # Bin and aggregate
    bins_def = [("<5", -999, 5), ("5-10", 5, 10), ("10-20", 10, 20), (">20", 20, 999)]
    result = {}
    for label, lo, hi in bins_def:
        mask = (df["snr_db"] >= lo) & (df["snr_db"] < hi)
        subset = df[mask]
        result[label] = {
            "count": int(mask.sum()),
            "p_f1_mean": round(float(subset["p_f1"].mean()), 4) if len(subset) > 0 else None,
            "s_f1_mean": round(float(subset["s_f1"].mean()), 4) if len(subset) > 0 else None,
            "p_mae_mean": round(float(subset["p_mae_sec"].mean()), 4) if len(subset) > 0 and subset["p_mae_sec"].notna().any() else None,
            "s_mae_mean": round(float(subset["s_mae_sec"].mean()), 4) if len(subset) > 0 and subset["s_mae_sec"].notna().any() else None,
        }

    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def batch_evaluate_all(cfg: object, snr_stratified: bool = False) -> dict:
    """Evaluate all checkpoints in outputs/checkpoints/ and compute summary.

    Returns:
        Dict with per_run results and config-level mean±std summary.
    """
    ckpt_dir = Path("outputs/checkpoints")
    all_results = {}
    configs = {}

    for run_dir in sorted(ckpt_dir.iterdir()):
        ckpt_path = run_dir / "best_model.pt"
        if not ckpt_path.exists():
            continue

        result = evaluate_checkpoint(
            ckpt_path, cfg, snr_stratified=snr_stratified
        )
        all_results[run_dir.name] = result

        # Group by config (strip _seed*)
        config_name = run_dir.name.rsplit("_seed", 1)[0]
        if config_name not in configs:
            configs[config_name] = []
        configs[config_name].append(result)

    # Compute summary per config
    summary = {}
    for config_name, runs in sorted(configs.items()):
        tol_key = "tolerance_0.1s"  # primary metric
        p_f1s = [r[tol_key]["P"]["f1"] for r in runs]
        s_f1s = [r[tol_key]["S"]["f1"] for r in runs]
        p_maes = [r["timing"]["P"]["mae_sec"] for r in runs if r["timing"]["P"]["mae_sec"] is not None]
        s_maes = [r["timing"]["S"]["mae_sec"] for r in runs if r["timing"]["S"]["mae_sec"] is not None]
        p_means = [r["timing"]["P"]["mean_residual_sec"] for r in runs if r["timing"]["P"]["mean_residual_sec"] is not None]
        s_means = [r["timing"]["S"]["mean_residual_sec"] for r in runs if r["timing"]["S"]["mean_residual_sec"] is not None]
        p_stds = [r["timing"]["P"]["std_residual_sec"] for r in runs if r["timing"]["P"]["std_residual_sec"] is not None]
        s_stds = [r["timing"]["S"]["std_residual_sec"] for r in runs if r["timing"]["S"]["std_residual_sec"] is not None]

        summary[config_name] = {
            "n_seeds": len(runs),
            "tolerance_0.1s": {
                "p_f1_mean": round(float(np.mean(p_f1s)), 4),
                "p_f1_std": round(float(np.std(p_f1s)), 4),
                "s_f1_mean": round(float(np.mean(s_f1s)), 4),
                "s_f1_std": round(float(np.std(s_f1s)), 4),
            },
            "timing": {
                "p_mae_mean": round(float(np.mean(p_maes)), 4) if p_maes else None,
                "p_residual_mean": round(float(np.mean(p_means)), 4) if p_means else None,
                "p_residual_std": round(float(np.mean(p_stds)), 4) if p_stds else None,
                "s_mae_mean": round(float(np.mean(s_maes)), 4) if s_maes else None,
                "s_residual_mean": round(float(np.mean(s_means)), 4) if s_means else None,
                "s_residual_std": round(float(np.mean(s_stds)), 4) if s_stds else None,
            },
        }

    return {"per_run": all_results, "summary": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CS phase picker")
    parser.add_argument("--config", default="configs/stead_experiment.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Single checkpoint path")
    parser.add_argument("--batch-all", action="store_true", help="Evaluate all checkpoints")
    parser.add_argument("--snr-stratified", action="store_true", help="SNR-stratified eval")
    parser.add_argument("--dataset", default="stead", choices=["stead", "instance"],
                        help="Evaluation dataset (default: stead)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg._dataset = args.dataset

    if args.batch_all:
        logger.info("Batch evaluating all checkpoints...")
        results = batch_evaluate_all(cfg, snr_stratified=args.snr_stratified)

        out_path = Path(args.output or "outputs/results/test_evaluation_summary.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Summary saved: %s", out_path)

        # Print table
        print("\n=== Test Set Evaluation (tolerance=0.1s) ===")
        print(f"{'Config':<22s} {'P-F1':>14s} {'S-F1':>14s} {'P-MAE':>8s} {'S-MAE':>8s}")
        print("-" * 70)
        for config, s in sorted(results["summary"].items()):
            t01 = s["tolerance_0.1s"]
            tm = s["timing"]
            p_str = f"{t01['p_f1_mean']:.4f}±{t01['p_f1_std']:.4f}"
            s_str = f"{t01['s_f1_mean']:.4f}±{t01['s_f1_std']:.4f}"
            pm = f"{tm['p_mae_mean']:.3f}s" if tm['p_mae_mean'] else "N/A"
            sm = f"{tm['s_mae_mean']:.3f}s" if tm['s_mae_mean'] else "N/A"
            print(f"{config:<22s} {p_str:>14s} {s_str:>14s} {pm:>8s} {sm:>8s}")

    elif args.checkpoint:
        result = evaluate_checkpoint(
            args.checkpoint, cfg, snr_stratified=args.snr_stratified
        )
        out_path = Path(args.output or f"outputs/results/{result['config']}_eval.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Result saved: %s", out_path)

        # Print key metrics
        t01 = result["tolerance_0.1s"]
        print(f"\n{result['config']}:")
        print(f"  P: F1={t01['P']['f1']:.4f}, Prec={t01['P']['precision']:.4f}, Rec={t01['P']['recall']:.4f}")
        print(f"  S: F1={t01['S']['f1']:.4f}, Prec={t01['S']['precision']:.4f}, Rec={t01['S']['recall']:.4f}")
        if result["timing"]["P"]["mae_sec"]:
            print(f"  P timing: MAE={result['timing']['P']['mae_sec']:.4f}s, μ={result['timing']['P']['mean_residual_sec']:.4f}s, σ={result['timing']['P']['std_residual_sec']:.4f}s")
        if result["timing"]["S"]["mae_sec"]:
            print(f"  S timing: MAE={result['timing']['S']['mae_sec']:.4f}s, μ={result['timing']['S']['mean_residual_sec']:.4f}s, σ={result['timing']['S']['std_residual_sec']:.4f}s")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
