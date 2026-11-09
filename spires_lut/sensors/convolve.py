"""Band-average hyperspectral reflectance onto sensor bands.

Band reflectance is R_b = sum_i R(lambda_i) w_bi with w_bi the band response
integrated over the i-th bin of the input wavelength grid and normalised to sum to 1.
Responses are resampled *down* to the grid (bin edges midway between neighbouring
centres, i.e. +/-5 nm on the 10 nm DISORT grid); reflectance is never interpolated
up to 1 nm, because that would invent spectral structure DISORT never computed.
Measured curves are integrated with the trapezoid rule; ``tophat`` and ``gaussian``
responses are integrated analytically (error function for the Gaussian).

Known limitations
-----------------
* Narrow bands (OLI B1 ~16 nm, MSI 60 m bands 15-20 nm) sit on 2-3 bins of the
  10 nm single-scattering-property grid. This is a limitation of the SSP grid, not of
  the convolution; the legacy Zenodo LUTs have the same limitation.
* Solar-weighted (at-sensor) averaging is intentionally omitted: the inversion
  compares against surface-reflectance products, for which plain response weighting
  is the correct quantity.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf

from spires_lut.sensors.model import Band, Sensor

# Mirrors disort_wrapper.py:103-105 (np.arange(205, 4995.0001, 10) sliced from index 5).
DISORT_GRID_NM = np.arange(255.0, 4995.0001, 10.0)

_SIGMA_PER_FWHM = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def bin_edges(grid_nm) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(grid_nm, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("wavelength grid must be 1-D and strictly increasing")
    mid = 0.5 * (grid[1:] + grid[:-1])
    lo = np.concatenate([[grid[0] - (mid[0] - grid[0])], mid])
    hi = np.concatenate([mid, [grid[-1] + (grid[-1] - mid[-1])]])
    return lo, hi


@lru_cache(maxsize=None)
def load_srf(csv_name: str) -> pd.DataFrame:
    path = resources.files("spires_lut.sensors.data").joinpath(csv_name)
    with path.open("r") as fh:
        df = pd.read_csv(fh, comment="#", dtype={"band": str})
    expected = {"band", "wavelength_nm", "response"}
    if set(df.columns) != expected:
        raise ValueError(f"{csv_name}: expected columns {sorted(expected)}, got {list(df.columns)}")
    return df


def srf_curve(band: Band) -> Tuple[np.ndarray, np.ndarray]:
    df = load_srf(band.srf_file)
    sel = df[df["band"] == band.srf_band].sort_values("wavelength_nm")
    if sel.empty:
        raise ValueError(f"band {band.name!r}: no curve {band.srf_band!r} in {band.srf_file}")
    return sel["wavelength_nm"].to_numpy(dtype=float), sel["response"].to_numpy(dtype=float)


def _integrate_curve(wl, resp, lo, hi):
    cum = np.concatenate([[0.0], cumulative_trapezoid(resp, wl)])
    at = lambda x: np.interp(x, wl, cum, left=0.0, right=cum[-1])  # noqa: E731
    return at(hi) - at(lo)


def _integrate_tophat(lower, upper, lo, hi):
    return np.clip(np.minimum(hi, upper) - np.maximum(lo, lower), 0.0, None)


def _integrate_gaussian(center, fwhm, lo, hi):
    sigma = fwhm * _SIGMA_PER_FWHM
    z = lambda x: (x - center) / (sigma * np.sqrt(2.0))  # noqa: E731
    return 0.5 * (erf(z(hi)) - erf(z(lo)))


def band_weights(band: Band, grid_nm) -> np.ndarray:
    grid = np.asarray(grid_nm, dtype=float)
    lo, hi = bin_edges(grid)
    if band.response == "srf":
        wl, resp = srf_curve(band)
        raw = _integrate_curve(wl, resp, lo, hi)
    elif band.response == "tophat":
        raw = _integrate_tophat(band.lower_nm, band.upper_nm, lo, hi)
    else:
        raw = _integrate_gaussian(band.center_nm, band.fwhm_nm, lo, hi)
    total = raw.sum()
    if not total > 0:
        src = f" from {band.srf_file}" if band.response == "srf" else ""
        raise ValueError(
            f"band {band.name!r}{src}: response has no overlap with the "
            f"{grid[0]:.0f}-{grid[-1]:.0f} nm wavelength grid"
        )
    return raw / total


def sensor_weights(sensor: Sensor, grid_nm) -> np.ndarray:
    """(band, grid) weight matrix; raises ValueError naming any band without coverage."""
    return np.stack([band_weights(b, grid_nm) for b in sensor.bands])


def band_average(spectral: xr.DataArray, sensor: Sensor, wavelength_dim: str = "band") -> xr.DataArray:
    """Average ``spectral`` (any dims, one of them the wavelength axis in nm) onto ``sensor`` bands.

    The wavelength dim is replaced by a ``band`` dim holding the sensor's band names, with
    auxiliary coords ``center_wavelength_nm``, ``fwhm_nm`` and ``long_name``. Other dims,
    coords and attrs pass through unchanged.
    """
    if wavelength_dim not in spectral.dims:
        raise ValueError(f"spectral array has no {wavelength_dim!r} dim; dims are {spectral.dims}")
    grid = spectral[wavelength_dim].to_numpy().astype(float)
    weights = sensor_weights(sensor, grid)
    moved = spectral.transpose(wavelength_dim, ...)
    values = np.tensordot(weights, moved.to_numpy(), axes=([1], [0]))
    other = tuple(d for d in moved.dims if d != wavelength_dim)
    coords = {d: moved.coords[d] for d in other if d in moved.coords}
    coords["band"] = ("band", list(sensor.band_names))
    coords["center_wavelength_nm"] = ("band", [b.center_nm for b in sensor.bands], {"units": "nm"})
    coords["fwhm_nm"] = ("band", [b.fwhm_nm for b in sensor.bands], {"units": "nm"})
    coords["long_name"] = ("band", [b.long_name for b in sensor.bands])
    return xr.DataArray(values, dims=("band",) + other, coords=coords, attrs=dict(spectral.attrs))
