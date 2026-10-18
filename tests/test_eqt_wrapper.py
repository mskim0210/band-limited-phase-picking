"""Unit tests for cs_pipeline.eqt_wrapper (SeisBench EQTransformer glue)."""

from __future__ import annotations

import pytest
import torch

from cs_pipeline.eqt_wrapper import SeisBenchEQTransformer


@pytest.mark.parametrize("M", [6000, 1506])
def test_output_shape_and_grad(M: int) -> None:
    model = SeisBenchEQTransformer(M=M, N=6000)
    x = torch.randn(2, 3, M, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 3, 6000)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_logits_are_pre_sigmoid_and_noise_pinned() -> None:
    model = SeisBenchEQTransformer(M=6000, N=6000).eval()
    x = torch.randn(2, 3, 6000)
    with torch.no_grad():
        out = model(x)
        _det, p_prob, s_prob = model.net(x)  # SeisBench default: sigmoid outputs
    assert torch.all(out[:, 0] == 0.0)  # noise logit pinned at 0
    assert torch.allclose(torch.sigmoid(out[:, 1]), p_prob, atol=1e-5)
    assert torch.allclose(torch.sigmoid(out[:, 2]), s_prob, atol=1e-5)
    probs = torch.softmax(out, dim=1)
    assert torch.allclose(probs.sum(1), torch.ones(2, 6000), atol=1e-5)


def test_param_count_matches_seisbench() -> None:
    model = SeisBenchEQTransformer(M=6000)
    n = sum(p.numel() for p in model.parameters())
    assert 370_000 < n < 385_000
