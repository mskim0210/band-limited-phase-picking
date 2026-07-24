"""Train the compressed-domain phase picker on compressed STEAD data.

Records per-epoch metrics to CSV for the time-to-accuracy curve (Fig. 8):
    epoch, wall_clock_sec, train_loss, val_loss, val_p_f1, val_s_f1

Usage:
    # Smoke test (2 epochs)
    python scripts/train_cs_picker.py --config configs/stead_experiment.yaml \
        --matrix wavelet --cr 4 --seed 42 --max-epochs 2

    # Full training
    python scripts/train_cs_picker.py --config configs/stead_experiment.yaml \
        --matrix wavelet --cr 8 --seed 42

    # All 3 seeds
    for seed in 42 123 7; do
        python scripts/train_cs_picker.py --matrix wavelet --cr 8 --seed $seed
    done
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs_pipeline.cs_aware_model import CSPhaseNet
from cs_pipeline.cs_dataset import CompressedSTEADDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics (lightweight — no external dependency)
# ---------------------------------------------------------------------------


def pick_peaks(probs: np.ndarray, threshold: float = 0.3) -> int:
    """Find the index of the highest peak above threshold.

    Args:
        probs: (T,) probability array for a single phase.
        threshold: Minimum peak height.

    Returns:
        Peak sample index, or -1 if no peak found.
    """
    if probs.max() < threshold:
        return -1
    return int(np.argmax(probs))


def compute_phase_f1(
    pred_picks: list[int],
    true_picks: list[int],
    tolerance_samples: int = 10,
) -> dict[str, float]:
    """Compute F1, precision, recall for a list of phase picks.

    Args:
        pred_picks: Predicted pick indices (-1 = no pick).
        true_picks: True pick indices (-1 = no pick).
        tolerance_samples: Tolerance in samples (0.1s @ 100Hz = 10).

    Returns:
        Dict with f1, precision, recall, mae_samples.
    """
    tp = fp = fn = 0
    errors = []

    for pred, true in zip(pred_picks, true_picks):
        if true < 0 and pred < 0:
            continue  # true negative
        elif true < 0 and pred >= 0:
            fp += 1
        elif true >= 0 and pred < 0:
            fn += 1
        else:
            if abs(pred - true) <= tolerance_samples:
                tp += 1
                errors.append(abs(pred - true))
            else:
                fp += 1
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mae = float(np.mean(errors)) if errors else 0.0

    return {"f1": f1, "precision": precision, "recall": recall, "mae_samples": mae}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Validate and compute loss + phase F1 metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    p_preds: list[int] = []
    p_trues: list[int] = []
    s_preds: list[int] = []
    s_trues: list[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.item()
        n_batches += 1

        # Phase picking on softmax probabilities
        probs = torch.softmax(output, dim=1).cpu().numpy()  # (B, 3, 6000)
        labels = y.cpu().numpy()

        for i in range(probs.shape[0]):
            # P-phase (channel 1)
            p_preds.append(pick_peaks(probs[i, 1]))
            p_trues.append(pick_peaks(labels[i, 1], threshold=0.1))
            # S-phase (channel 2)
            s_preds.append(pick_peaks(probs[i, 2]))
            s_trues.append(pick_peaks(labels[i, 2], threshold=0.1))

    val_loss = total_loss / max(n_batches, 1)
    p_metrics = compute_phase_f1(p_preds, p_trues)
    s_metrics = compute_phase_f1(s_preds, s_trues)

    return {
        "val_loss": val_loss,
        "p_f1": p_metrics["f1"],
        "p_precision": p_metrics["precision"],
        "p_recall": p_metrics["recall"],
        "p_mae": p_metrics["mae_samples"],
        "s_f1": s_metrics["f1"],
        "s_precision": s_metrics["precision"],
        "s_recall": s_metrics["recall"],
        "s_mae": s_metrics["mae_samples"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_training(
    cfg: "DictConfig",
    matrix_type: str,
    cr: int,
    seed: int,
    max_epochs: int | None = None,
    output_dir: Path | None = None,
    dilation: int = 1,
    arch: str = "csphasenet",
) -> Path:
    """Run full training loop with per-epoch CSV logging.

    Args:
        cfg: OmegaConf config.
        matrix_type: Sensing matrix type.
        cr: Compression ratio.
        seed: Random seed.
        max_epochs: Override max epochs (for smoke test).
        output_dir: Override output directory.

    Returns:
        Path to output directory with checkpoints + CSV.
    """
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Paths & level detection.
    # NOTE: discriminate the true baseline by matrix_type, NOT by cr==1 —
    # the 'lowpass' control also has cr=1 but reads its own encoded
    # data from cs_encoded/lowpass_cr1 (M=6000, no downsample).
    is_baseline = (matrix_type == "baseline")
    if is_baseline:
        # L2 baseline: use original memmap with CSPhaseNet(M=6000)
        cr = 1
        M = 6000
        if output_dir is None:
            suffix = f"_dil{dilation}" if dilation > 1 else ""
            prefix = "phasenet_baseline" if arch == "phasenet" else f"l2_baseline{suffix}"
            output_dir = Path("outputs/checkpoints") / f"{prefix}_seed{seed}"
    else:
        M = 6000 // cr
        cs_dir = Path(cfg.data.cs_dir) / f"{matrix_type}_cr{cr}"
        if not cs_dir.exists():
            raise FileNotFoundError(
                f"CS encoded data not found: {cs_dir}. "
                f"Run: python scripts/preprocess_cs.py --cr {cr} --matrix {matrix_type}"
            )
        if output_dir is None:
            prefix = "phasenet_" if arch == "phasenet" else ""
            output_dir = Path("outputs/checkpoints") / f"{prefix}{matrix_type}_cr{cr}_seed{seed}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    patience = cfg.training.patience
    num_workers = cfg.training.num_workers
    sigma = cfg.training.label_sigma
    epochs = max_epochs or cfg.training.max_epochs

    level_name = "L2 baseline" if is_baseline else f"L3 {matrix_type} CR={cr}"
    logger.info(
        "Training: %s, M=%d, seed=%d, epochs=%d, batch=%d",
        level_name, M, seed, epochs, batch_size,
    )

    # Datasets
    if is_baseline:
        from scripts.profile_baseline import STEADMemmapDataset
        train_ds = STEADMemmapDataset(cfg.data.memmap_dir, "train", sigma=sigma)
        val_ds = STEADMemmapDataset(cfg.data.memmap_dir, "val", sigma=sigma)
    else:
        train_ds = CompressedSTEADDataset(cs_dir, "train", sigma=sigma)
        val_ds = CompressedSTEADDataset(cs_dir, "val", sigma=sigma)
        # Use the actual encoded length (wavelet M=1506≠6000//4; lowpass M=6000)
        if train_ds.M is not None:
            M = int(train_ds.M)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arch == "phasenet":
        from cs_pipeline.phasenet_wrapper import SeisBenchPhaseNet
        model = SeisBenchPhaseNet(M=M, N=6000).to(device)
    else:
        model = CSPhaseNet(M=M, N=6000, base_channels=16, dilation=dilation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # CSV logger for TTA curve
    csv_path = output_dir / "epoch_log.csv"
    csv_fields = [
        "epoch", "wall_clock_sec", "train_loss", "val_loss",
        "val_p_f1", "val_s_f1", "val_p_precision", "val_p_recall",
        "val_s_precision", "val_s_recall", "val_p_mae", "val_s_mae",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)

        wall_clock = time.time() - t_start
        epoch_time = time.time() - t_epoch_start

        # Log to CSV
        row = {
            "epoch": epoch,
            "wall_clock_sec": round(wall_clock, 1),
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_p_f1": round(val_metrics["p_f1"], 4),
            "val_s_f1": round(val_metrics["s_f1"], 4),
            "val_p_precision": round(val_metrics["p_precision"], 4),
            "val_p_recall": round(val_metrics["p_recall"], 4),
            "val_s_precision": round(val_metrics["s_precision"], 4),
            "val_s_recall": round(val_metrics["s_recall"], 4),
            "val_p_mae": round(val_metrics["p_mae"], 2),
            "val_s_mae": round(val_metrics["s_mae"], 2),
        }
        csv_writer.writerow(row)
        csv_file.flush()

        logger.info(
            "Epoch %d/%d (%.0fs) | train_loss=%.4f | val_loss=%.4f | "
            "P-F1=%.4f | S-F1=%.4f | wall=%.0fs",
            epoch, epochs, epoch_time, train_loss,
            val_metrics["val_loss"], val_metrics["p_f1"], val_metrics["s_f1"],
            wall_clock,
        )

        # Checkpoint best
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            patience_counter = 0
            ckpt_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_p_f1": val_metrics["p_f1"],
                "val_s_f1": val_metrics["s_f1"],
                "config": {
                    "matrix_type": matrix_type, "cr": cr, "seed": seed,
                    "M": M, "N": 6000, "base_channels": 16,
                    "dilation": dilation, "arch": arch,
                },
            }, ckpt_path)
            logger.info("  → Best model saved (val_loss=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    csv_file.close()
    total_time = time.time() - t_start
    logger.info(
        "Training complete: %d epochs, %.1f min, best_val_loss=%.4f",
        epoch, total_time / 60, best_val_loss,
    )
    logger.info("CSV log: %s", csv_path)
    logger.info("Best model: %s", output_dir / "best_model.pt")

    return output_dir


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train the compressed-domain phase picker")
    parser.add_argument("--config", default="configs/stead_experiment.yaml")
    parser.add_argument("--level", default="L3", choices=["L2", "L3"],
                        help="L2=baseline (no CS), L3=CS compressed")
    parser.add_argument("--matrix", default="wavelet", choices=["wavelet", "gaussian", "dct", "butterworth", "lowpass"])
    parser.add_argument("--cr", type=int, default=8, choices=[1, 2, 4, 8, 16])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dilation", type=int, default=1,
                        help="Conv dilation factor (receptive-field control)")
    parser.add_argument("--arch", default="csphasenet",
                        choices=["csphasenet", "phasenet"],
                        help="Model architecture (phasenet = SeisBench PhaseNet)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else None

    matrix = "baseline" if args.level == "L2" else args.matrix
    cr = 1 if args.level == "L2" else args.cr

    run_training(
        cfg=cfg,
        matrix_type=matrix,
        cr=cr,
        seed=args.seed,
        max_epochs=args.max_epochs,
        output_dir=out_dir,
        dilation=args.dilation,
        arch=args.arch,
    )


if __name__ == "__main__":
    main()
