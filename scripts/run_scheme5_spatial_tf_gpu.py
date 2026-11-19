#!/usr/bin/env python3
"""GPU training: scheme5 only, all seven EarthquakeNPP catalogs.

Scheme 5 uses a Spatial Transformer with Temporal and Kappa left as FCNN 5x32.
Protocol is identical to the four-scheme GPU runs: official jitter catalog and
config.py splits, H=128, batch 16, lr=1e-4, seed 1, 50-epoch cap, patience 0
(run all 50, always restore the best-validation snapshot), mu initialized from
the train event rate/area and updated analytically each epoch, boundary_angles=128.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import run_four_schemes_gpu as four  # noqa: E402
import run_six_datasets_gpu as six  # noqa: E402

SCHEME5 = {
    "id": "scheme5",
    "name": "Spatial Transformer only; Temporal and Kappa FCNN",
    "flags": {
        "temporal_backbone": "fcnn",
        "spatial_backbone": "transformer",
        "kappa_backbone": "fcnn",
    },
}
four.SCHEMES["scheme5"] = SCHEME5

six.DATASETS = (
    "SCEDC_30",
    "SanJac",
    "WHITE",
    "SCEDC_25",
    "SaltonSea",
    "ComCat",
    "SCEDC_20",
)
six.CATALOGS = dict(six.CATALOGS)
six.CATALOGS["SanJac"] = "Datasets/QTM/SanJac_catalog_jitter.csv"
six.WINDOWS = dict(six.WINDOWS)
six.WINDOWS["SanJac"] = (
    "aux 2008-01-01, train 2009-01-01, val 2014-01-01, "
    "test 2016-01-01–2018-01-01, m0=1.0"
)


if __name__ == "__main__":
    if "--schemes" not in sys.argv:
        sys.argv.extend(["--schemes", "scheme5"])
    six.main()
