"""Forward-pass inference throughput (Table 10).

Protocol: STEAD test windows (3, 6000), batch size 256, CUDA-synchronized,
mean over 30 batches after 5 warm-up batches, single GPU.  Three models are
timed: the full-band CSPhaseNet, the dilated full-band control (Section 3.4)
and the wavelet CR = 4 model.  Two numbers are reported per model, in traces
per second:

* ``*_forward_only``: GPU forward pass on input that is already standardized
  (and, for CR = 4, already wavelet-encoded) and resident on the device;
* ``baseline`` / ``dilated`` / ``cr4``: end to end, i.e. per-trace
  standardization on the CPU, for CR = 4 the per-window wavelet encoding on
  one CPU core (PyWavelets), and the host-to-device copy, followed by the
  forward pass.

The 2.9× forward-only advantage of CR = 4 therefore requires the encoding to
be amortized at the stream level or pre-encoded storage; with per-window
encoding the end-to-end rate falls below the baseline (Table 10 caption).

Usage:
    python scripts/inference_throughput.py [--batch 256 --n-batches 30 --warmup 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINTS = {
    "baseline": "outputs/checkpoints/l2_baseline_seed42/best_model.pt",
    "dilated": "outputs/checkpoints/l2_baseline_dil4_seed42/best_model.pt",
    "cr4": "outputs/checkpoints/wavelet_cr4_seed42/best_model.pt",
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def standardize(x: torch.Tensor) -> torch.Tensor:
    """Per-trace standard-deviation normalization, as in the training loaders."""
    std = x.std(dim=(1, 2), keepdim=True)
    return x / torch.where(std > 1e-8, std, torch.ones_like(std))


@torch.no_grad()
def measure_traces_per_sec(model: torch.nn.Module, x: torch.Tensor, n_batches: int,
                           warmup: int, device: torch.device,
                           encode: Callable[[torch.Tensor], torch.Tensor] | None = None) -> float:
    """Traces per second of ``model`` on batch ``x``.

    ``encode`` (optional) is applied to the CPU batch inside the timed loop so
    that end-to-end numbers include it; without it the batch is moved to the
    device once and only the forward pass is timed.  The first ``warmup``
    batches are not timed.
    """
    model = model.to(device).eval()
    x_dev = None if encode is not None else x.to(device)
    t0 = 0.0
    for i in range(warmup + n_batches):
        if i == warmup:
            _sync(device)
            t0 = time.perf_counter()
        xb = encode(x).to(device) if encode is not None else x_dev
        model(xb)
    _sync(device)
    dt = time.perf_counter() - t0
    return float(n_batches * x.shape[0] / dt)


def ratios(j: dict) -> dict[str, float]:
    """The three ratios quoted in the Table 10 caption and Section 3.5."""
    return {"cr4_forward_only_vs_baseline": round(j["cr4_forward_only"] / j["baseline_forward_only"], 2),
            "cr4_end_to_end_vs_baseline": round(j["cr4"] / j["baseline"], 2),
            "dilated_forward_only_vs_baseline": round(j["dilated_forward_only"] / j["baseline_forward_only"], 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Table 10 inference throughput")
    ap.add_argument("--config", default="configs/stead_experiment.yaml")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-batches", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="outputs/results/inference_throughput.json")
    args = ap.parse_args()

    from cs_pipeline.model_factory import build_model
    from cs_pipeline.sensing_matrix import create_sensing_matrix

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)  # one CPU core for the per-window encoding
    wf = np.load(Path(cfg.data.memmap_dir) / "test_waveforms.npy", mmap_mode="r")
    idx = np.sort(np.random.RandomState(args.seed).choice(len(wf), args.batch, replace=False))
    raw = torch.from_numpy(np.asarray(wf[idx], dtype=np.float32))  # (B, 3, 6000)
    mat = create_sensing_matrix("wavelet", N=6000, CR=4)

    models = {}
    for name, path in CHECKPOINTS.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = build_model(ck["config"])
        m.load_state_dict(ck["model_state_dict"])
        models[name] = m

    out: dict = {}
    # forward only: standardized (and encoded) input resident on the device
    x_full = standardize(raw)
    x_cr4 = standardize(mat.encode(x_full))
    for name in ("baseline", "dilated"):
        out[f"{name}_forward_only"] = measure_traces_per_sec(models[name], x_full, args.n_batches,
                                                             args.warmup, device)
    out["cr4_forward_only"] = measure_traces_per_sec(models["cr4"], x_cr4, args.n_batches,
                                                     args.warmup, device)
    # end to end: CPU standardization (+ per-window wavelet encoding) + H2D + forward
    for name in ("baseline", "dilated"):
        out[name] = measure_traces_per_sec(models[name], raw, args.n_batches, args.warmup, device,
                                           encode=standardize)
    out["cr4"] = measure_traces_per_sec(models["cr4"], raw, args.n_batches, args.warmup, device,
                                        encode=lambda xb: standardize(mat.encode(standardize(xb))))
    out = {k: int(round(v)) for k, v in out.items()}
    out["ratios"] = ratios(out)
    out["meta"] = {"batch": args.batch, "n_batches": args.n_batches, "warmup": args.warmup,
                   "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                   "cpu_threads": torch.get_num_threads(), "encoder": "wavelet db4 L=6, CR=4 (PyWavelets)",
                   "checkpoints": CHECKPOINTS, "unit": "traces per second"}
    Path(args.output).write_text(json.dumps(out, indent=1))
    logger.info("saved %s: %s", args.output, json.dumps({k: v for k, v in out.items() if k != "meta"}))


if __name__ == "__main__":
    main()
