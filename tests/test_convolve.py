import numpy as np
import pytest
import xarray as xr

from spires_lut.sensors.convolve import (
    DISORT_GRID_NM,
    band_average,
    band_weights,
    bin_edges,
)
from spires_lut.sensors.model import Band, Sensor

GRID = np.arange(255.0, 4995.0001, 10.0)


def test_disort_grid_constant_matches_wrapper():
    # disort_wrapper.py:103-105 -> 475 centres, 255..4995 nm, 10 nm step
    assert DISORT_GRID_NM.shape == (475,)
    assert DISORT_GRID_NM[0] == 255.0 and DISORT_GRID_NM[-1] == 4995.0


def test_bin_edges_are_plus_minus_half_step():
    lo, hi = bin_edges(GRID)
    np.testing.assert_allclose(lo, GRID - 5.0)
    np.testing.assert_allclose(hi, GRID + 5.0)


def test_bin_edges_reject_unsorted():
    with pytest.raises(ValueError, match="increasing"):
        bin_edges(np.array([300.0, 200.0, 400.0]))


def test_tophat_weights_split_evenly_across_two_bins():
    b = Band(name="T", center_nm=1000.0, fwhm_nm=20.0, response="tophat", lower_nm=990.0, upper_nm=1010.0)
    w = band_weights(b, GRID)
    assert w.sum() == pytest.approx(1.0)
    nz = np.flatnonzero(w)
    assert GRID[nz].tolist() == [995.0, 1005.0]
    np.testing.assert_allclose(w[nz], [0.5, 0.5])


def test_gaussian_weights_sum_to_one_and_are_symmetric():
    b = Band(name="G", center_nm=1000.0, fwhm_nm=20.0, response="gaussian")
    w = band_weights(b, GRID)
    assert w.sum() == pytest.approx(1.0)
    i = np.argmin(np.abs(GRID - 995.0))
    assert w[i] == pytest.approx(w[i + 1])  # bins 995 and 1005 straddle the centre


def test_response_outside_grid_raises_naming_band():
    b = Band(name="TIR1", center_nm=8300.0, fwhm_nm=250.0, response="gaussian")
    with pytest.raises(ValueError, match="TIR1"):
        band_weights(b, GRID)


def _sensor(*bands):
    return Sensor(name="X", platform="test", bands=bands, srf_source="n/a")


def test_flat_spectrum_averages_to_itself_for_tophat_and_gaussian():
    spectral = xr.DataArray(
        np.full((GRID.size, 2), 0.5, dtype=np.float32),
        dims=("band", "solar_angle"),
        coords={"band": ("band", GRID, {"units": "nm"}), "solar_angle": [0.0, 30.0]},
    )
    s = _sensor(
        Band(name="T", center_nm=1000.0, fwhm_nm=20.0, response="tophat", lower_nm=990.0, upper_nm=1010.0),
        Band(name="G", center_nm=1600.0, fwhm_nm=90.0, response="gaussian"),
    )
    out = band_average(spectral, s)
    assert out.dims == ("band", "solar_angle")
    assert list(out["band"].values) == ["T", "G"]
    np.testing.assert_allclose(out.values, 0.5, rtol=1e-6)
    np.testing.assert_allclose(out["center_wavelength_nm"].values, [1000.0, 1600.0])
    np.testing.assert_allclose(out["fwhm_nm"].values, [20.0, 90.0])


def test_step_spectrum_gives_analytic_half():
    step = (GRID >= 1000.0).astype(np.float32)  # 0 below 1000 nm, 1 at and above
    spectral = xr.DataArray(step, dims=("band",), coords={"band": ("band", GRID, {"units": "nm"})})
    s = _sensor(
        Band(name="T", center_nm=1000.0, fwhm_nm=20.0, response="tophat", lower_nm=990.0, upper_nm=1010.0),
        Band(name="G", center_nm=1000.0, fwhm_nm=30.0, response="gaussian"),
    )
    out = band_average(spectral, s)
    np.testing.assert_allclose(out.values, [0.5, 0.5], atol=1e-6)


def test_band_average_requires_wavelength_dim():
    spectral = xr.DataArray(np.zeros(3), dims=("x",))
    with pytest.raises(ValueError, match="band"):
        band_average(spectral, _sensor(Band(name="G", center_nm=1000.0, fwhm_nm=30.0, response="gaussian")))
