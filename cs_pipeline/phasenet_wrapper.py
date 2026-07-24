"""SeisBench PhaseNet wrapped for the CS training pipeline.

Adapts seisbench.models.PhaseNet (Zhu & Beroza, 2019; stride-4 U-Net with a
receptive field spanning thousands of samples -- i.e., receptive-field
adequate for 60-s windows) to the (B, 3, M) -> (B, 3, N) logits interface of
CSPhaseNet, so the same training/evaluation code runs unchanged.

The wrapper right-pads the input to the nearest length the PhaseNet
encoder/decoder accepts (found by trial at construction), trims the output
back to M, and linearly interpolates to N when M != N (mirroring CSPhaseNet's
output upsampling for compressed inputs).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SeisBenchPhaseNet(nn.Module):
    """seisbench.models.PhaseNet with padding/trim/upsample glue.

    Args:
        M: Input length (compressed or full).
        N: Output length. Default: 6000.
        in_channels: Input channels. Default: 3.
        num_classes: Output classes. Default: 3 (Noise, P, S).
    """

    def __init__(self, M: int = 6000, N: int = 6000, in_channels: int = 3,
                 num_classes: int = 3) -> None:
        super().__init__()
        import seisbench.models as sbm  # lazy: optional dependency

        self.M = M
        self.N = N
        self.net = sbm.PhaseNet(in_channels=in_channels, classes=num_classes)

        # Find the smallest acceptable padded length >= M by trial.
        self.pad = None
        with torch.no_grad():
            for extra in range(0, 64):
                try:
                    self.net(torch.zeros(1, in_channels, M + extra))
                    self.pad = extra
                    break
                except RuntimeError:
                    continue
        if self.pad is None:
            raise ValueError(f"No acceptable PhaseNet input length near M={M}")

        n_params = sum(p.numel() for p in self.parameters())
        logger.info("SeisBenchPhaseNet: M=%d (+%d pad) → N=%d, params=%.2fM",
                    M, self.pad, N, n_params / 1e6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, M) → (B, 3, N) logits."""
        if self.pad:
            x = F.pad(x, (0, self.pad))
        out = self.net(x, logits=True)
        if self.pad:
            out = out[..., :self.M]
        if self.M != self.N:
            out = F.interpolate(out, size=self.N, mode="linear",
                                align_corners=False)
        return out
