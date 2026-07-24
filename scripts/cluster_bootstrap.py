"""Cluster bootstrap CI comparison: trace-level vs event-level.

Computes per-trace P/S picking results, then runs both trace-level
and event-cluster bootstrap to compare CI widths.

Usage:
    python scripts/cluster_bootstrap.py --config configs/stead_experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def compute_f1(tp: int, fp: int, fn: int) -> float:
    """Compute F1 from raw counts."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def get_per_trace_results(
    checkpoint_path: str,
    cfg: object,
    tolerance: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
) -> list[dict]:
    """Run inference and return per-trace tp/fp/fn for P and S.

    Returns:
        List of dicts with keys: p_tp, p_fp, p_fn, s_tp, s_fp, s_fn.
    """
    from cs_pipeline.cs_aware_model import CSPhaseNet
    from cs_pipeline.cs_dataset import CompressedSTEADDataset
    from cs_pipeline.metrics import compute_phase_metrics, pick_peaks
    from scripts.profile_baseline import STEADMemmapDataset

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    M = config["M"]
    N = config.get("N", 6000)
    cr = config.get("cr", 4)
    matrix_type = config.get("matrix_type", "wavelet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSPhaseNet(M=M, N=N, base_channels=config.get("base_channels", 16))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    if matrix_type == "baseline":  # NOT cr==1: lowpass control has cr=1 but its own data
        test_ds = STEADMemmapDataset(cfg.data.memmap_dir, "test", sigma=10.0)
    else:
        cs_dir = Path(cfg.data.cs_dir) / f"{matrix_type}_cr{cr}"
        test_ds = CompressedSTEADDataset(cs_dir, "test", sigma=10.0)

    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    results = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            labels = y.numpy()

            for i in range(probs.shape[0]):
                pred = pick_peaks(probs[i], threshold=0.3)
                true = pick_peaks(labels[i], threshold=0.5)
                pm = compute_phase_metrics(pred["P"], true["P"], tolerance)
                sm = compute_phase_metrics(pred["S"], true["S"], tolerance)
                results.append({
                    "p_tp": pm["tp"], "p_fp": pm["fp"], "p_fn": pm["fn"],
                    "s_tp": sm["tp"], "s_fp": sm["fp"], "s_fn": sm["fn"],
                })

    return results


def trace_bootstrap(
    results: list[dict], phase: str = "p", B: int = 10000, seed: int = 42,
) -> dict:
    """Trace-level bootstrap CI."""
    rng = np.random.RandomState(seed)
    n = len(results)
    tp_arr = np.array([r[f"{phase}_tp"] for r in results])
    fp_arr = np.array([r[f"{phase}_fp"] for r in results])
    fn_arr = np.array([r[f"{phase}_fn"] for r in results])

    f1_samples = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        f1_samples.append(compute_f1(
            tp_arr[idx].sum(), fp_arr[idx].sum(), fn_arr[idx].sum()
        ))

    f1_samples = np.array(f1_samples)
    return {
        "ci_low": round(float(np.percentile(f1_samples, 2.5)), 4),
        "ci_high": round(float(np.percentile(f1_samples, 97.5)), 4),
        "mean": round(float(f1_samples.mean()), 4),
        "std": round(float(f1_samples.std()), 4),
    }


def cluster_bootstrap(
    results: list[dict],
    cluster_ids: np.ndarray,
    phase: str = "p",
    B: int = 10000,
    seed: int = 42,
) -> dict:
    """Event-level cluster bootstrap CI (vectorized)."""
    rng = np.random.RandomState(seed)

    tp_arr = np.array([r[f"{phase}_tp"] for r in results])
    fp_arr = np.array([r[f"{phase}_fp"] for r in results])
    fn_arr = np.array([r[f"{phase}_fn"] for r in results])

    # Map cluster_ids to integer labels
    unique_clusters, cluster_labels = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)

    # Pre-aggregate per-cluster sums
    cluster_tp = np.zeros(n_clusters, dtype=np.int64)
    cluster_fp = np.zeros(n_clusters, dtype=np.int64)
    cluster_fn = np.zeros(n_clusters, dtype=np.int64)
    np.add.at(cluster_tp, cluster_labels, tp_arr)
    np.add.at(cluster_fp, cluster_labels, fp_arr)
    np.add.at(cluster_fn, cluster_labels, fn_arr)

    # Vectorized bootstrap: sample cluster indices, sum via fancy indexing
    f1_samples = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n_clusters, size=n_clusters)
        boot_tp = cluster_tp[idx].sum()
        boot_fp = cluster_fp[idx].sum()
        boot_fn = cluster_fn[idx].sum()
        f1_samples[b] = compute_f1(int(boot_tp), int(boot_fp), int(boot_fn))

    return {
        "ci_low": round(float(np.percentile(f1_samples, 2.5)), 4),
        "ci_high": round(float(np.percentile(f1_samples, 97.5)), 4),
        "mean": round(float(f1_samples.mean()), 4),
        "std": round(float(f1_samples.std()), 4),
        "n_clusters": int(n_clusters),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Cluster bootstrap CI comparison")
    parser.add_argument("--config", default="configs/stead_experiment.yaml")
    parser.add_argument("--B", type=int, default=10000)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    # Build test set cluster IDs
    logger.info("Loading metadata for cluster IDs...")
    meta = pd.read_csv(cfg.data.stead_csv, usecols=["source_id"], low_memory=False)
    rng = np.random.RandomState(cfg.seed)
    indices = rng.permutation(len(meta))
    n_train = int(len(meta) * cfg.data.split_ratio[0])
    n_val = int(len(meta) * cfg.data.split_ratio[1])
    test_indices = indices[n_train + n_val:]

    test_source_ids = meta.iloc[test_indices]["source_id"].values
    # Replace NaN with unique IDs
    for i in range(len(test_source_ids)):
        if pd.isna(test_source_ids[i]):
            test_source_ids[i] = f"_nan_{i}"

    # Configs to evaluate
    configs = [
        ("l2_baseline", "seed42", "outputs/checkpoints/l2_baseline_seed42/best_model.pt"),
        ("l2_baseline", "seed123", "outputs/checkpoints/l2_baseline_seed123/best_model.pt"),
        ("l2_baseline", "seed7", "outputs/checkpoints/l2_baseline_seed7/best_model.pt"),
        ("wavelet_cr4", "seed42", "outputs/checkpoints/wavelet_cr4_seed42/best_model.pt"),
        ("wavelet_cr4", "seed123", "outputs/checkpoints/wavelet_cr4_seed123/best_model.pt"),
        ("wavelet_cr4", "seed7", "outputs/checkpoints/wavelet_cr4_seed7/best_model.pt"),
    ]

    all_results = {}
    for config_name, seed_name, ckpt_path in configs:
        key = f"{config_name}_{seed_name}"
        logger.info("=== %s ===", key)

        per_trace = get_per_trace_results(ckpt_path, cfg)
        logger.info("  %d traces evaluated", len(per_trace))

        # Point estimate
        p_tp = sum(r["p_tp"] for r in per_trace)
        p_fp = sum(r["p_fp"] for r in per_trace)
        p_fn = sum(r["p_fn"] for r in per_trace)
        s_tp = sum(r["s_tp"] for r in per_trace)
        s_fp = sum(r["s_fp"] for r in per_trace)
        s_fn = sum(r["s_fn"] for r in per_trace)

        p_f1 = compute_f1(p_tp, p_fp, p_fn)
        s_f1 = compute_f1(s_tp, s_fp, s_fn)

        # Bootstrap
        p_trace = trace_bootstrap(per_trace, "p", B=args.B)
        p_cluster = cluster_bootstrap(per_trace, test_source_ids, "p", B=args.B)
        s_trace = trace_bootstrap(per_trace, "s", B=args.B)
        s_cluster = cluster_bootstrap(per_trace, test_source_ids, "s", B=args.B)

        result = {
            "P_F1_point": round(p_f1, 4),
            "S_F1_point": round(s_f1, 4),
            "P": {"trace_CI": p_trace, "event_cluster_CI": p_cluster},
            "S": {"trace_CI": s_trace, "event_cluster_CI": s_cluster},
        }

        logger.info("  P-F1=%.4f  trace=[%.4f,%.4f]  cluster=[%.4f,%.4f] (n=%d)",
                     p_f1, p_trace["ci_low"], p_trace["ci_high"],
                     p_cluster["ci_low"], p_cluster["ci_high"], p_cluster["n_clusters"])
        logger.info("  S-F1=%.4f  trace=[%.4f,%.4f]  cluster=[%.4f,%.4f]",
                     s_f1, s_trace["ci_low"], s_trace["ci_high"],
                     s_cluster["ci_low"], s_cluster["ci_high"])

        if config_name not in all_results:
            all_results[config_name] = {}
        all_results[config_name][seed_name] = result

    # Save
    out_path = Path("outputs/results/cluster_bootstrap_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved %s", out_path)

    # Summary
    print("\n=== Cluster Bootstrap Summary (P-F1) ===")
    for config_name in ["l2_baseline", "wavelet_cr4"]:
        for seed_name in ["seed42", "seed123", "seed7"]:
            r = all_results[config_name][seed_name]
            pt = r["P"]["trace_CI"]
            pc = r["P"]["event_cluster_CI"]
            print(f"  {config_name}_{seed_name}: "
                  f"point={r['P_F1_point']:.4f}  "
                  f"trace=[{pt['ci_low']:.4f},{pt['ci_high']:.4f}]  "
                  f"cluster=[{pc['ci_low']:.4f},{pc['ci_high']:.4f}]")


if __name__ == "__main__":
    main()
