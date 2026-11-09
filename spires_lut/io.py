"""Read and write SPIReS reflectance LUTs as netCDF, validating on both sides."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import xarray as xr
from spires_contract.lut import validate_reflectance_lut

PathLike = Union[str, Path]


def write_lut(ds: xr.Dataset, path: PathLike) -> Path:
    """Validate ``ds`` against the reflectance-LUT contract, then write it. Returns ``path``."""
    validate_reflectance_lut(ds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {"band": {"dtype": "str"}} if ds["band"].dtype.kind in ("U", "O") else {}
    ds.to_netcdf(path, encoding=encoding)
    return path


def read_lut(path: PathLike) -> xr.Dataset:
    """Load a reflectance LUT fully into memory and validate it."""
    with xr.open_dataset(path) as handle:
        ds = handle.load()
    if ds["band"].dtype.kind == "O":
        ds = ds.assign_coords(band=ds["band"].astype(str))
    ds.encoding["source"] = str(path)
    validate_reflectance_lut(ds)
    return ds
