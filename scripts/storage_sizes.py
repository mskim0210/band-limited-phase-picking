"""Storage of the pre-encoded training archives (Table 5, Section 3.3, Section 4).

Storage is the float32 memmap payload of all three splits:
``traces × 3 components × M samples × 4 bytes``.  GB means 10⁹ bytes and the
saving is computed from bytes, never from rounded gigabytes.  When the
memmap files are present their on-disk sizes are recorded next to the
theoretical payload (they differ only by the 128-byte NPY headers).

Usage:
    python scripts/storage_sizes.py [--config configs/stead_experiment.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

N_TRACES = 1_265_657                # STEAD traces (85/5/10 split, all kept)
M_BY_CR = {1: 6000, 2: 3004, 4: 1506, 8: 756, 16: 382}
CS_DIR = {1: None, 2: "wavelet_cr2", 4: "wavelet_cr4", 8: "wavelet_cr8", 16: "wavelet_cr16"}


def payload_bytes(n_traces: int, m: int) -> int:
    """float32 payload of ``n_traces`` three-component traces of length ``m``."""
    return n_traces * 3 * m * 4


def saving_pct(m: int, n_full: int = 6000) -> float:
    """Storage saving relative to the full-length archive, in percent (1 decimal)."""
    return round(100.0 * (1.0 - m / n_full), 1)


def storage_table(n_traces: int, m_by_cr: dict[int, int]) -> list[dict]:
    rows = []
    for cr, m in sorted(m_by_cr.items()):
        b = payload_bytes(n_traces, m)
        pct = saving_pct(m, m_by_cr[1])
        rows.append({"cr": cr, "M": m, "bytes": b, "gb": round(b / 1e9, 3),
                     "gb_str": f"{b / 1e9:.1f} GB", "saving_pct": pct,
                     "saving_str": f"{pct:.0f}%"})
    return rows


def measured_bytes(cfg, cr: int) -> int | None:
    """On-disk size of the three ``*_waveforms.npy`` files of one archive."""
    if cr == 1:
        d = Path(cfg.data.memmap_dir)
    else:
        d = Path(cfg.data.cs_dir) / CS_DIR[cr]
    files = [d / f"{s}_waveforms.npy" for s in ("train", "val", "test")]
    if not all(f.exists() for f in files):
        return None
    return sum(f.stat().st_size for f in files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stead_experiment.yaml")
    ap.add_argument("--output", default="outputs/results/storage_sizes.json")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    rows = storage_table(N_TRACES, M_BY_CR)
    for r in rows:
        r["measured_bytes"] = measured_bytes(cfg, r["cr"])
    out = {"meta": {"n_traces": N_TRACES, "dtype": "float32", "components": 3,
                    "unit": "GB = 1e9 bytes", "definition": "traces x 3 x M x 4 bytes, all splits"},
           "rows": rows}
    Path(args.output).write_text(json.dumps(out, indent=1))
    for r in rows:
        logger.info("CR=%2d M=%4d %s (%s saving) measured=%s", r["cr"], r["M"], r["gb_str"],
                    r["saving_str"], r["measured_bytes"])
    logger.info("saved %s", args.output)


if __name__ == "__main__":
    main()
