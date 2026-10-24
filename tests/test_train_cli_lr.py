"""Tests for the ``--lr`` override of scripts/train_cs_picker.py.

EQTransformer diverges (training loss rises after epoch ~5) at the shared
lr = 1e-3, so the CLI must allow a per-run learning-rate override that is
also recorded in the checkpoint config.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from scripts.train_cs_picker import build_parser, resolve_lr


def test_parser_accepts_lr_override_and_defaults_to_none() -> None:
    parser = build_parser()
    assert parser.parse_args([]).lr is None
    assert parser.parse_args(["--lr", "1e-4"]).lr == 1e-4


def test_resolve_lr_prefers_override_over_config() -> None:
    cfg = OmegaConf.create({"training": {"lr": 1e-3}})
    assert resolve_lr(cfg, None) == 1e-3
    assert resolve_lr(cfg, 1e-4) == 1e-4
