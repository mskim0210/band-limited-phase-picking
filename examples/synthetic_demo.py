"""End-to-end synthetic mini-example (no external data required).

Runs the full pipeline of the paper on synthetic borehole-like waveforms:

  1. generate synthetic 3-component 60-s event windows (100 Hz) with known
     P and S arrivals plus pure-noise traces;
  2. wavelet band-level pre-encoding at CR = 4 (the paper's operating point);
  3. train a small CSPhaseNet on the compressed windows for a few epochs;
  4. evaluate peak-picking recall/precision against the known arrivals;
  5. demonstrate the label-free teacher-student distillation recipe
     (full-band teacher -> compressed student, no pick labels used).

Intended as the reproducible test case required by the journal's code
policy (the Korean borehole waveforms are available on request only); it
exercises the same code paths as the paper's experiments and finishes in
about a minute on CPU, seconds on GPU.

Usage:
    python examples/synthetic_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs_pipeline.cs_aware_model import CSPhaseNet
from cs_pipeline.sensing_matrix import create_sensing_matrix

FS, N = 100.0, 6000
RNG = np.random.default_rng(42)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ricker(t: np.ndarray, f0: float) -> np.ndarray:
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def make_event(p_sample: int, s_sample: int, snr: float) -> np.ndarray:
    """Synthetic 3-C window: band-limited noise + P (Z-dominant) + S (H-dominant)."""
    x = RNG.normal(0, 1.0, (3, N)).astype(np.float32)
    t = (np.arange(N) - p_sample) / FS
    p_wavelet = ricker(t, f0=5.0) * np.exp(-np.clip(t, 0, None) * 1.5)
    ts = (np.arange(N) - s_sample) / FS
    s_wavelet = ricker(ts, f0=3.0) * np.exp(-np.clip(ts, 0, None) * 1.0)
    x[2] += snr * p_wavelet + 0.4 * snr * s_wavelet          # Z
    x[0] += 0.3 * snr * p_wavelet + snr * s_wavelet          # E
    x[1] += 0.3 * snr * p_wavelet + 0.9 * snr * s_wavelet    # N
    return x


def make_dataset(n_events: int = 600, n_noise: int = 150):
    X, P, S = [], [], []
    for _ in range(n_events):
        p = int(RNG.integers(800, 2500))
        s = p + int(RNG.integers(300, 1500))
        X.append(make_event(p, s, snr=float(RNG.uniform(2.0, 6.0))))
        P.append(p)
        S.append(s)
    for _ in range(n_noise):
        X.append(RNG.normal(0, 1.0, (3, N)).astype(np.float32))
        P.append(-1)
        S.append(-1)
    X = np.stack(X)
    X /= X.std(axis=(1, 2), keepdims=True)
    P, S = np.array(P), np.array(S)
    order = RNG.permutation(len(X))  # interleave events and noise
    return X[order], P[order], S[order]


def make_labels(p, s, sigma=10.0):
    lab = np.zeros((len(p), 3, N), np.float32)
    x = np.arange(N, dtype=np.float32)
    for i in range(len(p)):
        if p[i] >= 0:
            lab[i, 1] = np.exp(-((x - p[i]) ** 2) / (2 * sigma**2))
        if s[i] >= 0:
            lab[i, 2] = np.exp(-((x - s[i]) ** 2) / (2 * sigma**2))
        lab[i, 0] = np.clip(1 - lab[i, 1] - lab[i, 2], 0, 1)
    return lab


def train(model, X, Y, epochs=30, lr=1e-3, bs=32, soft_targets=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([0.05, 0.40, 0.55]).to(DEV))
    model.train()
    idx = np.arange(len(X))
    for _ in range(epochs):
        RNG.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = torch.from_numpy(X[b]).to(DEV)
            yb = torch.from_numpy(Y[b]).to(DEV)
            opt.zero_grad()
            out = model(xb)
            loss = (-(yb * F.log_softmax(out, 1)).sum(1).mean()
                    if soft_targets else crit(out, yb))
            loss.backward()
            opt.step()
    model.eval()


@torch.no_grad()
def evaluate(model, X, gt, ch, thr=0.3, tol=50, bs=64):
    tp = fp = fn = 0
    for i in range(0, len(X), bs):
        probs = torch.softmax(model(torch.from_numpy(X[i:i + bs]).to(DEV)),
                              1)[:, ch].cpu().numpy()
        for j in range(len(probs)):
            g = gt[i + j]
            pks = find_peaks(probs[j], height=thr, distance=100)[0]
            if g < 0:
                fp += len(pks)
                continue
            if len(pks) == 0:
                fn += 1
                continue
            d = np.abs(pks - g)
            if d.min() <= tol:
                tp += 1
                fp += len(pks) - 1
            else:
                fn += 1
                fp += len(pks)
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return rec, prec


@torch.no_grad()
def soft_labels(model, X, bs=64):
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.softmax(
            model(torch.from_numpy(X[i:i + bs]).to(DEV)), 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    torch.manual_seed(42)
    print(f"device: {DEV}")

    print("[1/5] generating synthetic dataset (750 windows) ...")
    X, P, S = make_dataset()
    n_tr = 600
    Xtr, Xte, Ptr, Pte, Str_, Ste = X[:n_tr], X[n_tr:], P[:n_tr], P[n_tr:], \
        S[:n_tr], S[n_tr:]
    Ltr = make_labels(Ptr, Str_)

    print("[2/5] wavelet band-level pre-encoding at CR=4 ...")
    mat = create_sensing_matrix("wavelet", N=N, CR=4)
    enc = lambda A: (lambda z: (z / (z.std(axis=(1, 2), keepdims=True) + 1e-8)
                                ).astype(np.float32))(
        mat.encode(torch.from_numpy(A)).numpy())
    Ctr, Cte = enc(Xtr), enc(Xte)
    print(f"    {X.shape[-1]} -> {Ctr.shape[-1]} samples per channel")

    print("[3/5] training compressed CSPhaseNet (few epochs) ...")
    student = CSPhaseNet(M=Ctr.shape[-1], N=N, base_channels=8).to(DEV)
    train(student, Ctr, Ltr)
    rec, prec = evaluate(student, Cte, Pte, ch=1)
    print(f"    compressed model  P recall {rec:.2f}  precision {prec:.2f}")
    assert rec > 0.6, "compressed training failed to learn synthetic P onsets"

    print("[4/5] training full-band teacher ...")
    teacher = CSPhaseNet(M=N, N=N, base_channels=8).to(DEV)
    train(teacher, Xtr, Ltr)
    rec_t, prec_t = evaluate(teacher, Xte, Pte, ch=1)
    print(f"    full-band teacher P recall {rec_t:.2f}  precision {prec_t:.2f}")

    print("[5/5] label-free distillation (teacher soft output only) ...")
    T = soft_labels(teacher, Xtr)
    student2 = CSPhaseNet(M=Ctr.shape[-1], N=N, base_channels=8).to(DEV)
    train(student2, Ctr, T, soft_targets=True)
    rec_d, prec_d = evaluate(student2, Cte, Pte, ch=1)
    print(f"    distilled student P recall {rec_d:.2f}  precision {prec_d:.2f}"
          "  (no pick labels used)")
    assert rec_d > 0.5, "distillation failed on synthetic data"

    print("\nOK — full pipeline (encode -> train -> evaluate -> distill) "
          "reproduced on synthetic data.")


if __name__ == "__main__":
    main()
