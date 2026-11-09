import numpy as np
import pytest
import xarray as xr
from spires_contract import conventions as c

from spires_lut.sensors.convolve import DISORT_GRID_NM


@pytest.fixture
def hyperspectral_lut() -> xr.Dataset:
    """Shape-faithful stand-in for what simulate.create_spires_lut writes (3 grain x 2 LAP x 2 angle)."""
    grid = DISORT_GRID_NM
    sqrt_r = np.array([10.0, 20.0, 30.0])
    lap = np.array([0.0, 1000.0])
    angle = np.array([0.0, 30.0])
    # smooth, physically-shaped spectrum: high in the visible, dropping through the SWIR
    spectrum = 0.95 * np.exp(-((grid - 500.0) / 1500.0) ** 2)
    refl = (
        spectrum[:, None, None, None]
        * (1.0 - 0.002 * angle[None, :, None, None])
        * (1.0 - 1e-5 * lap[None, None, :, None])
        * (1.0 - 0.005 * sqrt_r[None, None, None, :])
    ).astype(np.float32)
    return xr.Dataset(
        {c.REFLECTANCE_LUT_VARIABLE: (("band", "solar_angle", "lap_concentration", "sqrt_grain_radius"), refl)},
        coords={
            "band": ("band", grid, {"units": "nm"}),
            "solar_angle": ("solar_angle", angle, {"units": c.LUT_AXIS_UNITS["solar_angle"]}),
            "lap_concentration": ("lap_concentration", lap,
                                  {"units": c.LUT_AXIS_UNITS["lap_concentration"], "lap_type": "dust"}),
            "sqrt_grain_radius": ("sqrt_grain_radius", sqrt_r, {"units": c.LUT_AXIS_UNITS["sqrt_grain_radius"]}),
        },
    )
