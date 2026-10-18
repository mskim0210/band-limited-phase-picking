"""Single place that maps a checkpoint/CLI config to a picker model."""

from __future__ import annotations

import torch.nn as nn

from cs_pipeline.cs_aware_model import CSPhaseNet

ARCHS = ("csphasenet", "phasenet", "eqtransformer")


def build_model(config: dict) -> nn.Module:
    """Build a picker from a config dict (as stored in ``ckpt["config"]``).

    Args:
        config: Keys ``arch`` (default "csphasenet"), ``M``, ``N`` (default
            6000), ``base_channels`` (default 16), ``dilation`` (default 1).

    Returns:
        Un-initialized model exposing the (B, 3, M) -> (B, 3, N) logits
        interface.

    Raises:
        ValueError: If ``arch`` is not one of ``ARCHS``.
    """
    arch = config.get("arch", "csphasenet")
    M, N = int(config["M"]), int(config.get("N", 6000))
    if arch == "phasenet":
        from cs_pipeline.phasenet_wrapper import SeisBenchPhaseNet
        return SeisBenchPhaseNet(M=M, N=N)
    if arch == "eqtransformer":
        from cs_pipeline.eqt_wrapper import SeisBenchEQTransformer
        return SeisBenchEQTransformer(M=M, N=N)
    if arch == "csphasenet":
        return CSPhaseNet(M=M, N=N, base_channels=config.get("base_channels", 16),
                          dilation=config.get("dilation", 1))
    raise ValueError(f"Unknown arch {arch!r}; expected one of {ARCHS}")
