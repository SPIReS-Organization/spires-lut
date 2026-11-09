#!/usr/bin/env python
"""Produce the SPIReS sensor reflectance LUTs for the AGU26 poster sprint (bead spires-kor).

Runs DISORT once per LAP type on the production grid (expensive; hours on one core,
use --cores), then band-averages the hyperspectral result onto every requested sensor
and writes one contract-validated netCDF per sensor:

    <out>/hyperspectral/<lap_type>/spires_reflectance_lut.nc      (+ spires_albedo_lut.nc)
    <out>/spires_reflectance_lut_<sensor>_<lap_type>.nc

The albedo LUT is broadband and sensor-independent; it is left where simulate.py put it.
Pass --hyperspectral <dir> to skip DISORT and reuse an existing run.

Example:
    python examples/make_sensor_luts.py --atm-lut /path/to/atm.nc --out /data/spires/luts --cores 16
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from spires_lut.io import read_lut, write_lut
from spires_lut.lut import build_sensor_lut, sensor_lut_filename
from spires_lut.sensors import get_sensor, list_sensors

DEFAULT_SENSORS = ["OLI", "OLI2", "MSI", "HLSL30", "HLSS30", "HLSS30-msi", "MODIS", "LANDSAT10"]
log = logging.getLogger("make_sensor_luts")


def run_disort(out_dir: Path, atm_lut: Path, lap_type: str, cores: int) -> Path:
    from spires_lut.simulate import create_spires_lut   # imports the compiled disort extension

    hs_dir = out_dir / "hyperspectral" / lap_type
    hs_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log.info("DISORT %s -> %s (cores=%d)", lap_type, hs_dir, cores)
    create_spires_lut(output_dir=hs_dir, atm_lut=atm_lut, cpu_cores=cores, lap_type=lap_type)
    log.info("DISORT done in %.0f s", time.perf_counter() - t0)
    return hs_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--atm-lut", type=Path, help="MODTRAN-like atmospheric LUT (required unless --hyperspectral)")
    ap.add_argument("--hyperspectral", type=Path, help="existing dir holding spires_reflectance_lut.nc; skips DISORT")
    ap.add_argument("--lap-type", default="dust_skiles_size3", help="DISORT lap_type stem (default: dust_skiles_size3)")
    ap.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS, choices=list_sensors(), metavar="SENSOR")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(message)s")

    if a.hyperspectral is None:
        if a.atm_lut is None:
            ap.error("--atm-lut is required unless --hyperspectral is given")
        hs_dir = run_disort(a.out, a.atm_lut, a.lap_type, a.cores)
    else:
        hs_dir = a.hyperspectral

    hs_path = hs_dir / "spires_reflectance_lut.nc"
    hs = read_lut(hs_path)
    lap_type = hs["lap_concentration"].attrs["lap_type"]
    log.info("hyperspectral LUT %s: %s", hs_path, dict(hs.sizes))

    for name in a.sensors:
        t0 = time.perf_counter()
        ds = build_sensor_lut(hs, get_sensor(name), source=str(hs_path))
        path = write_lut(ds, a.out / sensor_lut_filename(name, lap_type))
        log.info("%-11s %2d bands -> %s (%.2f s)", name, ds.sizes["band"], path.name, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
