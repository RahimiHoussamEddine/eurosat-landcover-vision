"""Build the benchmark using project-local cache directories."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[1]
    for variable, relative in {
        "MPLCONFIGDIR": ".cache/matplotlib",
        "XDG_CACHE_HOME": ".cache",
        "TMPDIR": ".cache/tmp",
    }.items():
        value = root / relative
        value.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(value))

    from eurosat_landcover_vision.pipeline import build

    summary = build()
    print(
        "Built EuroSAT benchmark: "
        f"CNN test accuracy={summary['cnn_test_metrics']['accuracy']:.4f}, "
        f"macro-F1={summary['cnn_test_metrics']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
