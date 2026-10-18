"""Unit tests for cs_pipeline.model_factory."""

from __future__ import annotations

import pytest
import torch

from cs_pipeline.model_factory import build_model


def test_factory_dispatch() -> None:
    cfgs = [
        {"arch": "csphasenet", "M": 1506, "N": 6000, "base_channels": 16, "dilation": 1},
        {"arch": "phasenet", "M": 1506, "N": 6000},
        {"arch": "eqtransformer", "M": 1506, "N": 6000},
    ]
    x = torch.randn(1, 3, 1506)
    names = []
    for cfg in cfgs:
        m = build_model(cfg).eval()
        with torch.no_grad():
            assert m(x).shape == (1, 3, 6000)
        names.append(type(m).__name__)
    assert names == ["CSPhaseNet", "SeisBenchPhaseNet", "SeisBenchEQTransformer"]


def test_factory_default_is_csphasenet() -> None:
    m = build_model({"M": 6000})  # legacy checkpoints lack 'arch'
    assert type(m).__name__ == "CSPhaseNet"


def test_factory_rejects_unknown_arch() -> None:
    with pytest.raises(ValueError):
        build_model({"arch": "mamba", "M": 6000})
