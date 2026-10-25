"""Fig. 3 (per-step time decomposition) with the CR = 4 and CR = 8 bars.

Thin wrapper kept for the documented command.  The single implementation is
``scripts.generate_figures.generate_fig3_step_decomposition`` (the function
``generate_figures.py`` calls as well), so both commands write the same
figure; the CR = 4 per-step time and its I/O/H2D components are the constants
documented there.

Usage:
    python scripts/make_fig2_cr4.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_figures import generate_fig3_step_decomposition, load_profiles  # noqa: E402

logger = logging.getLogger(__name__)

build = generate_fig3_step_decomposition
PROFILE_DIR = Path("outputs/profiles")
OUT_DIRS = (Path("outputs/figures"),)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    profiles = load_profiles(PROFILE_DIR)
    for out_dir in OUT_DIRS:
        logger.info("saved %s", build(profiles, out_dir))


if __name__ == "__main__":
    main()
