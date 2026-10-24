"""GPU training-step microbenchmark per architecture and input length.

For CSPhaseNet, SeisBench PhaseNet and SeisBench EQTransformer at the full-band
length (M = 6000) and the wavelet CR = 4 length (M = 1506), measures:

* ``step_ms``: forward + backward + Adam step, CUDA-synchronized, over a batch
  size sweep (the training protocol uses batch 64);
* ``cpu_enqueue_ms``: the same step timed without synchronization, i.e. the
  time the CPU needs to launch the kernels (if this equals ``step_ms`` the step
  is launch-latency-bound, not arithmetic-bound);
* ``cuda_kernels``: number of CUDA kernels launched in one step (torch.profiler).

Usage:
    python scripts/gpu_step_microbench.py --batches 64 256 1024 \
        --output outputs/results/gpu_step_microbench_sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs_pipeline.model_factory import build_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

ARCHS = ("csphasenet", "phasenet", "eqtransformer")
LENGTHS = (6000, 1506)
N_OUT = 6000


def _batch(batch: int, M: int, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, 3, M, generator=gen).to(device)
    y = torch.softmax(torch.randn(batch, 3, N_OUT, generator=gen), dim=1).to(device)
    return x, y


def _step(model: nn.Module, opt: torch.optim.Optimizer, crit: nn.Module,
          x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    opt.zero_grad()
    loss = crit(model(x), y)
    loss.backward()
    opt.step()
    return loss


def time_step(arch: str, M: int, batch: int, device: torch.device, n_warm: int = 10,
              n_iter: int = 30, seed: int = 0) -> dict[str, float]:
    """Return mean synchronized step time and CPU enqueue time in ms."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = build_model({"arch": arch, "M": M, "N": N_OUT}).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    x, y = _batch(batch, M, device, gen)
    for _ in range(n_warm):
        _step(model, opt, crit, x, y)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _step(model, opt, crit, x, y).item()  # .item() syncs, as in train_one_epoch
    sync_ms = (time.perf_counter() - t0) / n_iter * 1e3
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _step(model, opt, crit, x, y)
    enqueue_ms = (time.perf_counter() - t0) / n_iter * 1e3  # no sync: CPU launch cost only
    torch.cuda.synchronize()
    out = {"step_ms": round(sync_ms, 2), "cpu_enqueue_ms": round(enqueue_ms, 2)}
    del model, opt, x, y
    torch.cuda.empty_cache()
    return out


def count_kernels(arch: str, M: int, batch: int, device: torch.device) -> int:
    """Number of CUDA kernels launched by one forward+backward+Adam step."""
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    model = build_model({"arch": arch, "M": M, "N": N_OUT}).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    x, y = _batch(batch, M, device, gen)
    for _ in range(3):
        _step(model, opt, crit, x, y)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _step(model, opt, crit, x, y)
        torch.cuda.synchronize()
    n = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    del model, opt, x, y
    torch.cuda.empty_cache()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU training-step microbenchmark")
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--archs", nargs="+", default=list(ARCHS))
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--output", default="outputs/results/gpu_step_microbench_sweep.json")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    out: dict = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                 "n_iter": args.n_iter, "lengths": list(LENGTHS), "by_batch": {},
                 "cuda_kernels_per_step_b64": {}}
    for arch in args.archs:
        for M in LENGTHS:
            k = count_kernels(arch, M, 64, device)
            out["cuda_kernels_per_step_b64"][f"{arch}_M{M}"] = k
            logger.info("%-13s M=%-4d kernels/step=%d", arch, M, k)
    for batch in args.batches:
        out["by_batch"][str(batch)] = {}
        for arch in args.archs:
            for M in LENGTHS:
                r = time_step(arch, M, batch, device, n_iter=args.n_iter)
                out["by_batch"][str(batch)][f"{arch}_M{M}"] = r
                logger.info("batch=%-5d %-13s M=%-4d step=%7.2f ms  cpu_enqueue=%7.2f ms",
                            batch, arch, M, r["step_ms"], r["cpu_enqueue_ms"])
        for arch in args.archs:
            a = out["by_batch"][str(batch)][f"{arch}_M6000"]["step_ms"]
            b = out["by_batch"][str(batch)][f"{arch}_M1506"]["step_ms"]
            out["by_batch"][str(batch)][f"{arch}_speedup"] = round(a / b, 2)
            logger.info("batch=%-5d %-13s speedup 6000/1506 = %.2fx", batch, arch, a / b)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    logger.info("saved %s", args.output)


if __name__ == "__main__":
    main()
