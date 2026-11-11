#!/usr/bin/env python
"""Produce the SPIReS sensor reflectance LUTs for the AGU26 poster sprint (bead spires-kor).

Runs the surface RT engine once per LAP type on the production grid, then band-averages the
hyperspectral result onto every requested sensor and writes one contract-validated netCDF per
sensor:

    <out>/hyperspectral/<lap_type>/spires_reflectance_lut.nc                 (--engine disort; + spires_albedo_lut.nc with --atm-lut)
    <out>/hyperspectral/<lap_type>-delta-eddington/spires_reflectance_lut.nc (--engine delta-eddington)
    <out>/spires_reflectance_lut_<sensor>_<lap_type>.nc

DISORT is expensive (hours on one core, use --cores); delta-Eddington runs the whole grid in
seconds (~0.15 ms per point) and serves as a stand-in and as a DISORT cross-check
(examples/compare_engines.py). Sensor LUT
filenames do not encode the engine: to build both sets, use two different --out directories
(the refuse-to-overwrite guard stops silent mixing).

The albedo LUT is broadband and sensor-independent; it is left where simulate.py put it.
Pass --hyperspectral <dir> to skip the engine run and reuse an existing one.

Example:
    python examples/make_sensor_luts.py --out /data/spires/luts --cores 16
    python examples/make_sensor_luts.py --atm-lut /path/to/atm.nc --out /data/spires/luts --cores 16  # + albedo LUT
    python examples/make_sensor_luts.py --engine delta-eddington --out /data/spires/luts-de
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


ENGINES = ("disort", "delta-eddington")


def hyperspectral_dir(out_dir: Path, lap_type: str, engine: str) -> Path:
    """DISORT keeps the historical <out>/hyperspectral/<lap_type>; other engines get a suffix."""
    return out_dir / "hyperspectral" / (lap_type if engine == "disort" else f"{lap_type}-{engine}")


def run_hyperspectral(out_dir: Path, atm_lut: Path | None, lap_type: str, cores: int, engine: str) -> Path:
    from spires_lut.simulate import create_spires_lut   # imports the compiled disort extension

    hs_dir = hyperspectral_dir(out_dir, lap_type, engine)
    hs_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log.info("%s %s -> %s (cores=%d)", engine, lap_type, hs_dir, cores)
    create_spires_lut(output_dir=hs_dir, atm_lut=atm_lut, cpu_cores=cores, lap_type=lap_type,
                      engine=engine.replace("-", "_"))
    log.info("%s done in %.0f s", engine, time.perf_counter() - t0)
    return hs_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--atm-lut", type=Path,
                    help="MODTRAN-like atmospheric LUT; without it the broadband albedo LUT is skipped and the "
                         "reflectance LUT is the black-sky DHR (what simulate.py stores anyway)")
    ap.add_argument("--hyperspectral", type=Path, help="existing dir holding spires_reflectance_lut.nc; skips DISORT")
    ap.add_argument("--lap-type", default="dust_skiles_size3",
                    help="DISORT lap optical-property stem; also names the output files (default: dust_skiles_size3)")
    ap.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS, choices=list_sensors(), metavar="SENSOR")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--engine", choices=ENGINES, default="disort",
                    help="surface RT engine for the hyperspectral LUT (default: disort); delta-eddington is a "
                         "closed form that runs the grid in seconds, reflectance only, cannot be combined with --atm-lut")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(message)s")
    if a.engine != "disort" and a.atm_lut is not None:
        ap.error("--atm-lut needs --engine disort (the albedo LUT is DISORT-only)")

    if a.hyperspectral is None:
        hs_dir = run_hyperspectral(a.out, a.atm_lut, a.lap_type, a.cores, a.engine)
    else:
        hs_dir = a.hyperspectral

    hs_path = hs_dir / "spires_reflectance_lut.nc"
    hs = read_lut(hs_path)
    log.info("hyperspectral LUT %s: %s", hs_path, dict(hs.sizes))

    # simulate.py stamps every LUT's lap_type attr as the contract family "dust", so the
    # file name has to come from the optical-property stem the user actually ran.
    targets = {name: a.out / sensor_lut_filename(name, a.lap_type) for name in a.sensors}
    existing = [p for p in targets.values() if p.exists()]
    if existing:
        ap.error("refusing to overwrite: " + ", ".join(str(p) for p in existing))

    for name, out_path in targets.items():
        t0 = time.perf_counter()
        ds = build_sensor_lut(hs, get_sensor(name), source=str(hs_path))
        path = write_lut(ds, out_path)
        log.info("%-11s %2d bands -> %s (%.2f s)", name, ds.sizes["band"], path.name, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
