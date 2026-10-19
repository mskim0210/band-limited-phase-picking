"""cluster_bootstrap must refuse event-split checkpoints (trace-level test set overlaps their training rows)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_bootstrap_refuses_event_split_checkpoint(tmp_path: Path) -> None:
    from scripts.cluster_bootstrap import get_per_trace_results
    ck = tmp_path / "best_model.pt"
    torch.save({"config": {"M": 6000, "N": 6000, "cr": 1, "matrix_type": "baseline", "split": "event"},
                "model_state_dict": {}}, ck)
    with pytest.raises(ValueError, match="event"):
        get_per_trace_results(str(ck), cfg=None)
