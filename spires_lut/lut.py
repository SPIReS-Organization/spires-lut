"""Assemble contract-validated sensor LUTs from the hyperspectral DISORT LUT.

This is the seam between ``simulate.py`` (expensive physics, run once per LAP type)
and ``sensors`` (cheap band averaging, run once per sensor). It is the only place,
besides ``io.py``, that imports ``spires_contract`` validators.
"""
from __future__ import annotations

import datetime as dt
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import xarray as xr
from spires_contract import conventions as c
from spires_contract.lut import validate_reflectance_lut

from spires_lut.sensors.convolve import band_average
from spires_lut.sensors.model import Sensor

try:
    _VERSION = version("spires-lut")
except PackageNotFoundError:  # running from a checkout
    _VERSION = "0+unknown"

_WAVELENGTH_DIM = "band"   # simulate.py stores wavelength (nm) on the 'band' axis


def sensor_lut_filename(sensor_name: str, lap_type: str) -> str:
    return f"spires_reflectance_lut_{sensor_name.lower()}_{lap_type.lower()}.nc"


def _check_hyperspectral(ds: xr.Dataset) -> xr.DataArray:
    if c.REFLECTANCE_LUT_VARIABLE not in ds.data_vars:
        raise ValueError(f"hyperspectral LUT lacks variable {c.REFLECTANCE_LUT_VARIABLE!r}")
    da = ds[c.REFLECTANCE_LUT_VARIABLE]
    if tuple(da.dims) != tuple(c.REFLECTANCE_LUT_DIMS):
        raise ValueError(f"hyperspectral LUT dims {da.dims} != required {c.REFLECTANCE_LUT_DIMS}")
    if _WAVELENGTH_DIM not in da.coords:
        raise ValueError(f"hyperspectral LUT lacks the {_WAVELENGTH_DIM!r} wavelength coordinate")
    wl = da[_WAVELENGTH_DIM]
    if not np.issubdtype(wl.dtype, np.number):
        raise ValueError(
            f"{_WAVELENGTH_DIM!r} coordinate must be numeric wavelength in nm, got dtype {wl.dtype}; "
            "this looks like an already band-averaged LUT"
        )
    if "lap_concentration" not in da.coords or not da["lap_concentration"].attrs.get(c.LUT_LAP_TYPE_ATTR):
        # let the contract phrase this one so the message matches other callers
        validate_reflectance_lut(ds)
    return da


def build_sensor_lut(hyperspectral: xr.Dataset, sensor: Sensor, source: str = "") -> xr.Dataset:
    """Band-average a hyperspectral reflectance LUT onto ``sensor`` and validate the result.

    ``hyperspectral`` is what ``simulate.create_spires_lut`` writes (or ``io.read_lut`` of
    it): variable ``reflectance`` on ``(band, solar_angle, lap_concentration,
    sqrt_grain_radius)`` with ``band`` holding wavelength in nm. The returned Dataset has
    the same dims with ``band`` replaced by the sensor's band names (string coordinate),
    auxiliary ``center_wavelength_nm`` / ``fwhm_nm`` / ``long_name`` coords, float32
    data, axis ``units`` and ``lap_type`` carried over, and provenance in ``attrs``.
    """
    da = _check_hyperspectral(hyperspectral)
    averaged = band_average(da, sensor, wavelength_dim=_WAVELENGTH_DIM).astype(np.float32)
    kinds = sorted({b.response for b in sensor.bands})
    lap_type = da["lap_concentration"].attrs[c.LUT_LAP_TYPE_ATTR]
    out = xr.Dataset({c.REFLECTANCE_LUT_VARIABLE: averaged})
    out.attrs = {
        "sensor": sensor.name,
        "platform": sensor.platform,
        "srf_source": sensor.srf_source,
        "srf_response_kind": kinds[0] if len(kinds) == 1 else ",".join(kinds),
        "lap_type": lap_type,
        "spires_lut_version": _VERSION,
        "created": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source_hyperspectral_lut": source or hyperspectral.encoding.get("source", ""),
    }
    validate_reflectance_lut(out)
    return out
