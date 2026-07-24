"""Measure the effective receptive field (ERF) of the trained pickers.

For each model (baseline, dilated 4x, wavelet CR=4), computes the gradient
of the P-channel output logit at the window centre with respect to the
input, averaged over real test traces, and reports the width of the input
interval containing 95% of the absolute-gradient mass -- converted to
SECONDS of signal so that models with different input lengths are
comparable. This turns the receptive-field mechanism claim from inferred
to directly measured.

Usage:
    python scripts/measure_erf.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs_pipeline.cs_aware_model import CSPhaseNet
from cs_pipeline.sensing_matrix import create_sensing_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

from omegaconf import OmegaConf
MEM = Path(OmegaConf.load("configs/stead_experiment.yaml").data.memmap_dir)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N, FS = 6000, 100.0
K = 64          # traces averaged
T_OUT = N // 2  # output position probed (window centre)


def load_model(ck: str) -> CSPhaseNet:
    d = torch.load(ck, map_location="cpu", weights_only=False)
    m = CSPhaseNet(M=d["config"]["M"], N=N,
                   base_channels=d["config"].get("base_channels", 16),
                   dilation=d["config"].get("dilation", 1))
    m.load_state_dict(d["model_state_dict"])
    return m.to(DEV).eval()


def erf_width_seconds(model, X: np.ndarray, dt: float) -> tuple[float, np.ndarray]:
    """95% |gradient|-mass width (s) of d logit_P(T_OUT) / d input."""
    prof = np.zeros(X.shape[-1], np.float64)
    for i in range(len(X)):
        x = torch.from_numpy(X[i:i + 1]).to(DEV).requires_grad_(True)
        out = model(x)                      # (1, 3, 6000) logits
        out[0, 1, T_OUT].backward()
        prof += x.grad.abs().sum(dim=1)[0].detach().cpu().numpy()
    prof /= prof.sum()
    csum = np.cumsum(prof)
    lo = int(np.searchsorted(csum, 0.025))
    hi = int(np.searchsorted(csum, 0.975))
    return (hi - lo) * dt, prof


def main() -> None:
    W = np.load(MEM / "test_waveforms.npy", mmap_mode="r")
    idx = np.linspace(0, len(W) - 1, K).astype(int)
    Xb = np.ascontiguousarray(W[idx]).astype(np.float32)
    Xb /= Xb.std(axis=(1, 2), keepdims=True) + 1e-8

    mat = create_sensing_matrix("wavelet", N=N, CR=4)
    X4 = mat.encode(torch.from_numpy(Xb)).numpy()
    X4 /= X4.std(axis=(1, 2), keepdims=True) + 1e-8
    X4 = X4.astype(np.float32)

    models = {
        "baseline": ("outputs/checkpoints/l2_baseline_seed42/best_model.pt",
                     Xb, 1.0 / FS),
        "dilated": ("outputs/checkpoints/l2_baseline_dil4_seed42/best_model.pt",
                    Xb, 1.0 / FS),
        "wavelet_cr4": ("outputs/checkpoints/wavelet_cr4_seed42/best_model.pt",
                        X4, (N / FS) / X4.shape[-1]),
    }

    res = {}
    for name, (ck, X, dt) in models.items():
        w, _ = erf_width_seconds(load_model(ck), X, dt)
        res[name] = round(w, 2)
        logger.info("%-12s ERF(95%% grad mass) = %.2f s", name, w)

    out = {"probe": "d logit_P(centre) / d input, |grad| summed over channels, "
                    f"averaged over {K} test traces; width holding 95% of mass",
           "erf_seconds": res}
    Path("outputs/results/erf_measurement.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
