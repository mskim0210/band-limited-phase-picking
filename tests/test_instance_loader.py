"""Zero-shot STEAD→INSTANCE loader must match the STEAD training protocol.

STEAD stores each trace as E, N, Z and the memmap keeps that order, so the
INSTANCE reader has to request the same component order explicitly (the
SeisBench default is Z, N, E).  Training standardizes each trace in its input
domain, i.e. after encoding for compressed models, so the INSTANCE window
must be standardized after encoding as well.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.evaluate import STEAD_COMPONENT_ORDER, InstanceCropDataset


class FakeInstance:
    metadata = pd.DataFrame({"split": ["test"]})

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def get_sample(self, i):
        t = np.arange(12000) / 100.0
        wf = np.stack([3 * np.sin(2 * np.pi * 2 * t),
                       2 * np.sin(2 * np.pi * 2 * t),
                       np.sin(2 * np.pi * 2 * t)]).astype("float32")
        return wf, {"trace_P_arrival_sample": 6000, "trace_S_arrival_sample": 6500}


def test_instance_reader_requests_stead_component_order() -> None:
    assert STEAD_COMPONENT_ORDER == "ENZ"
    with patch("seisbench.data.InstanceCounts", side_effect=lambda **kw: FakeInstance(**kw)) as ctor:
        InstanceCropDataset()
    assert ctor.call_args.kwargs["component_order"] == "ENZ"


def test_compressed_window_is_standardized_after_encoding() -> None:
    with patch("seisbench.data.InstanceCounts", return_value=FakeInstance()):
        x, _ = InstanceCropDataset()[0]
        y, _ = InstanceCropDataset(cr=4, matrix_type="wavelet")[0]
    assert x.shape == (3, 6000) and abs(float(x.std()) - 1.0) < 1e-3
    assert y.shape == (3, 1506)
    assert abs(float(y.std()) - 1.0) < 1e-3  # training normalizes in the compressed domain
