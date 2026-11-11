#!/usr/bin/env python
"""Compare two SPIReS reflectance LUTs on the same grid (e.g. DISORT vs delta-Eddington).

    python examples/compare_engines.py A.nc B.nc

Prints mean bias (B - A), RMSE and max |B - A| overall, per spectral region (VIS 400-700,
NIR 700-1400, SWIR 1400-2500 nm) and per solar angle. Exits 2 if the coordinate grids differ.
Because both engines share every optical constant (spires_lut.optics), the numbers isolate the
solver approximation — this is the DISORT validation the second engine buys us (bead spires-dvc).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from spires_lut.io import read_lut

REGIONS_NM = {
    "VIS 400-700": (400.0, 700.0),
    "NIR 700-1400": (700.0, 1400.0),
    "SWIR 1400-2500": (1400.0, 2500.0),
}
COORDS = ("band", "solar_angle", "lap_concentration", "sqrt_grain_radius")


def _stats(delta: xr.DataArray) -> dict:
    d = delta.values.astype(np.float64)
    return {
        "bias": float(np.mean(d)),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "max_abs": float(np.max(np.abs(d))),
    }


def same_grid(a: xr.Dataset, b: xr.Dataset) -> bool:
    return all(a[c].size == b[c].size and np.allclose(a[c].values, b[c].values) for c in COORDS)


def compare(a: xr.Dataset, b: xr.Dataset) -> dict:
    """Statistics of ``b - a`` for two LUTs on the same grid; see module docstring for keys."""
    delta = b["reflectance"] - a["reflectance"]
    band = a["band"].values.astype(np.float64)
    out = {"overall": _stats(delta), "region": {}, "solar_angle": {}}
    for name, (lo, hi) in REGIONS_NM.items():
        sel = (band >= lo) & (band <= hi)
        out["region"][name] = _stats(delta.isel(band=np.flatnonzero(sel)))
    for i, angle in enumerate(a["solar_angle"].values):
        out["solar_angle"][float(angle)] = _stats(delta.isel(solar_angle=i))
    return out


def _print(stats: dict, label_a: str, label_b: str) -> None:
    print(f"B - A   where A = {label_a}\n              B = {label_b}\n")
    print(f"{'subset':<22} {'bias':>10} {'rmse':>10} {'max|d|':>10}")
    rows = [("overall", stats["overall"])]
    rows += list(stats["region"].items())
    rows += [(f"solar_angle={angle:g}", s) for angle, s in stats["solar_angle"].items()]
    for name, s in rows:
        print(f"{name:<22} {s['bias']:>+10.5f} {s['rmse']:>10.5f} {s['max_abs']:>10.5f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="reference reflectance LUT (e.g. DISORT)")
    ap.add_argument("b", type=Path, help="candidate reflectance LUT (e.g. delta-Eddington)")
    args = ap.parse_args()

    a, b = read_lut(args.a), read_lut(args.b)
    if not same_grid(a, b):
        print("grids differ:", {c: (a[c].size, b[c].size) for c in COORDS}, file=sys.stderr)
        sys.exit(2)
    _print(compare(a, b), f"{args.a} [{a.attrs.get('surface_model', '?')}]",
           f"{args.b} [{b.attrs.get('surface_model', '?')}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
