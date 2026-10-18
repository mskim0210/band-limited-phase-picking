"""SeisBench EQTransformer wrapped for the CS training pipeline.

Adapts seisbench.models.EQTransformer (Mousavi et al., 2020; CNN + BiLSTM +
self-attention, i.e. a global receptive field) to the (B, 3, M) -> (B, 3, N)
logits interface of CSPhaseNet so the same soft-target cross-entropy training
and peak-picking evaluation code run unchanged.

The P and S head pre-activations (``logits=True``) are used as class logits
against a noise logit pinned at zero.  Because a softmax is invariant to a
common shift of its logits, this parameterization has exactly the expressive
power of a free 3-class softmax.  The detection head is left untrained, which
sidesteps the multi-task ambiguity noted by the reviewer.  EQTransformer
accepts ``in_samples=M`` natively, so no padding is needed; outputs are
linearly interpolated to N when M != N (mirroring CSPhaseNet).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SeisBenchEQTransformer(nn.Module):
    """seisbench.models.EQTransformer with zero-pinned-noise logits glue.

    Args:
        M: Input length (compressed or full).
        N: Output length. Default: 6000.
        in_channels: Input channels. Default: 3.
        drop_rate: Dropout inside EQTransformer. Default: 0.1 (SeisBench default).
    """

    def __init__(self, M: int = 6000, N: int = 6000, in_channels: int = 3,
                 drop_rate: float = 0.1) -> None:
        super().__init__()
        import seisbench.models as sbm  # lazy: optional dependency

        self.M = M
        self.N = N
        self.net = sbm.EQTransformer(
            in_channels=in_channels, in_samples=M, classes=2, phases="PS",
            lstm_blocks=3, drop_rate=drop_rate, sampling_rate=100, norm="std",
        )
        n_params = sum(p.numel() for p in self.parameters())
        logger.info("SeisBenchEQTransformer: M=%d → N=%d, params=%.2fM",
                    M, N, n_params / 1e6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, M) → (B, 3, N) logits ordered [noise=0, P, S]."""
        _det, lp, ls = self.net(x, logits=True)  # each (B, M)
        out = torch.stack([torch.zeros_like(lp), lp, ls], dim=1)  # (B, 3, M)
        if self.M != self.N:
            out = F.interpolate(out, size=self.N, mode="linear",
                                align_corners=False)
        return out
