"""CS-aware PhaseNet model for compressed-domain phase picking.

A lightweight U-Net that takes compressed (3, M) waveforms and produces
(3, 6000) phase probability outputs matching original-length labels.

The architecture adapts to any M = N/CR automatically via the stem
and output layers, while the core encoder/decoder operates at feature level.

References:
    - PhaseNet (Zhu & Beroza, 2019): original architecture
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CSPhaseNet(nn.Module):
    """CS-aware PhaseNet for compressed waveform input.

    Architecture:
        Stem: Conv1d(3 → base_ch) on compressed input (M samples)
        Encoder: 4 downsampling blocks with residual connections
        Decoder: 4 upsampling blocks with skip connections
        Upsample: Interpolate from encoder output length to 6000
        Head: Conv1d → 3 classes (Noise, P, S)

    Args:
        M: Compressed input length (= N // CR).
        N: Original output length. Default: 6000.
        in_channels: Input channels. Default: 3.
        num_classes: Output channels. Default: 3 (Noise, P, S).
        base_channels: Base feature channels. Default: 16.
        dilation: Dilation factor applied to every convolution. Enlarges the
            effective receptive field by the same factor without changing the
            parameter count (receptive-field control experiment). Default: 1.

    Example::

        model = CSPhaseNet(M=750, N=6000)
        x = torch.randn(4, 3, 750)
        y = model(x)  # (4, 3, 6000)
    """

    def __init__(
        self,
        M: int = 750,
        N: int = 6000,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 16,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.dilation = dilation
        bc = base_channels
        d = dilation

        # Stem on compressed domain
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, bc, 7, padding=3 * d, dilation=d),
            nn.BatchNorm1d(bc),
            nn.ReLU(inplace=True),
        )

        # Encoder (4 levels)
        self.enc1 = self._enc_block(bc, bc * 2, d)
        self.enc2 = self._enc_block(bc * 2, bc * 4, d)
        self.enc3 = self._enc_block(bc * 4, bc * 8, d)
        self.enc4 = self._enc_block(bc * 8, bc * 16, d)

        # Decoder (4 levels)
        self.dec4 = self._dec_block(bc * 16, bc * 8, d)
        self.dec3 = self._dec_block(bc * 8 * 2, bc * 4, d)  # *2 for skip
        self.dec2 = self._dec_block(bc * 4 * 2, bc * 2, d)
        self.dec1 = self._dec_block(bc * 2 * 2, bc, d)

        # Output head
        self.head = nn.Sequential(
            nn.Conv1d(bc * 2, bc, 3, padding=1 * d, dilation=d),
            nn.BatchNorm1d(bc),
            nn.ReLU(inplace=True),
            nn.Conv1d(bc, num_classes, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "CSPhaseNet: M=%d → N=%d, base_ch=%d, dilation=%d, params=%.2fM",
            M, N, bc, d, n_params / 1e6,
        )

    @staticmethod
    def _enc_block(in_ch: int, out_ch: int, d: int = 1) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 7, stride=2, padding=3 * d, dilation=d),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 7, padding=3 * d, dilation=d),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _dec_block(in_ch: int, out_ch: int, d: int = 1) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 7, padding=3 * d, dilation=d),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 7, padding=3 * d, dilation=d),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, 3, M) → (B, 3, 6000).

        Args:
            x: (B, in_channels, M) compressed waveform.

        Returns:
            (B, num_classes, N) phase probabilities.
        """
        # Stem
        s = self.stem(x)  # (B, bc, M)

        # Encoder
        e1 = self.enc1(s)    # (B, bc*2, M//2)
        e2 = self.enc2(e1)   # (B, bc*4, M//4)
        e3 = self.enc3(e2)   # (B, bc*8, M//8)
        e4 = self.enc4(e3)   # (B, bc*16, M//16)

        # Decoder with skip connections
        d4 = self.dec4(e4)                                         # (B, bc*8, M//16)
        d4 = F.interpolate(d4, size=e3.shape[2], mode="linear", align_corners=False)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))                 # (B, bc*4, M//8)
        d3 = F.interpolate(d3, size=e2.shape[2], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))                 # (B, bc*2, M//4)
        d2 = F.interpolate(d2, size=e1.shape[2], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))                 # (B, bc, M//2)
        d1 = F.interpolate(d1, size=s.shape[2], mode="linear", align_corners=False)

        out = self.head(torch.cat([d1, s], dim=1))  # (B, num_classes, M)

        # Upsample to original length N
        out = F.interpolate(out, size=self.N, mode="linear", align_corners=False)
        return out
