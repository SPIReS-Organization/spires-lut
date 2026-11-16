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


def test_fully_in_grid_tophat_and_gaussian_sum_to_one_within_tolerance():
    top = Band(name="T", center_nm=1000.0, fwhm_nm=20.0, response="tophat", lower_nm=990.0, upper_nm=1010.0)
    gau = Band(name="G", center_nm=1600.0, fwhm_nm=90.0, response="gaussian")
    assert band_weights(top, GRID).sum() == pytest.approx(1.0, abs=1e-12)
    assert band_weights(gau, GRID).sum() == pytest.approx(1.0, abs=1e-6)


def test_tophat_past_grid_edge_is_normalised_by_full_width():
    # Grid spans 250-5000 nm; the band is 4000-5010 nm, so 1000 of its 1010 nm are on the grid.
    b = Band(name="E", center_nm=4505.0, fwhm_nm=1010.0, response="tophat", lower_nm=4000.0, upper_nm=5010.0)
    w = band_weights(b, GRID)
    assert w.sum() == pytest.approx(1000.0 / 1010.0, rel=1e-12)
    i = np.argmin(np.abs(GRID - 4995.0))
    assert w[i] == pytest.approx(10.0 / 1010.0, rel=1e-12)  # not 10/1000 (in-grid renormalisation)


def test_tophat_with_less_than_99_percent_in_grid_raises_naming_band_and_fraction():
    # 1000 of 1015 nm on the grid = 98.5 %, just under the 99 % minimum.
    b = Band(name="E", center_nm=4507.5, fwhm_nm=1015.0, response="tophat", lower_nm=4000.0, upper_nm=5015.0)
    with pytest.raises(ValueError, match=r"'E'.*98\.52%"):
        band_weights(b, GRID)


def test_fraction_in_message_keeps_two_decimals_just_under_minimum():
    # 0.9899 is under the 0.99 minimum; one decimal would print "99.0%", which reads as passing.
    upper = 4000.0 + 1000.0 / 0.9899
    b = Band(name="E", center_nm=4500.0, fwhm_nm=upper - 4000.0, response="tophat", lower_nm=4000.0, upper_nm=upper)
    with pytest.raises(ValueError, match=r"98\.99% of its response"):
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


def test_band_average_rejects_wavelength_dim_without_coord():
    # positions 0..n-1 must not be silently treated as nm
    spectral = xr.DataArray(np.zeros((3, 2)), dims=("band", "solar_angle"))
    with pytest.raises(ValueError, match="no coordinate"):
        band_average(spectral, _sensor(Band(name="G", center_nm=1000.0, fwhm_nm=30.0, response="gaussian")))


from pathlib import Path
from unittest import mock

import spires_lut.sensors.convolve as convolve_mod

TEST_DATA = Path(__file__).parent / "data"


def _patch_srf_dir(monkeypatch):
    """Point load_srf at tests/data instead of the bundled package data."""
    convolve_mod.load_srf.cache_clear()
    monkeypatch.setattr(
        convolve_mod.resources, "files", lambda _pkg: TEST_DATA, raising=True
    )


def test_srf_flat_curve_matches_tophat_with_same_edges(monkeypatch):
    _patch_srf_dir(monkeypatch)
    srf = Band(name="F", center_nm=860.0, fwhm_nm=40.0, response="srf", srf_file="flat_srf.csv", srf_band="F")
    top = Band(name="T", center_nm=860.0, fwhm_nm=40.0, response="tophat", lower_nm=840.0, upper_nm=880.0)
    np.testing.assert_allclose(band_weights(srf, GRID), band_weights(top, GRID), atol=1e-12)
    convolve_mod.load_srf.cache_clear()


def test_srf_missing_column_names_band_and_file(monkeypatch):
    _patch_srf_dir(monkeypatch)
    bad = Band(name="Z", center_nm=860.0, fwhm_nm=40.0, response="srf", srf_file="flat_srf.csv", srf_band="nope")
    with pytest.raises(ValueError, match="flat_srf.csv"):
        band_weights(bad, GRID)
    convolve_mod.load_srf.cache_clear()


def test_single_row_srf_raises_value_error_naming_band(monkeypatch, tmp_path):
    # One sample has no integral; must hit the band-naming ValueError, not an IndexError.
    (tmp_path / "one_row_srf.csv").write_text("band,wavelength_nm,response\nS,1000,1.0\n")
    convolve_mod.load_srf.cache_clear()
    monkeypatch.setattr(convolve_mod.resources, "files", lambda _pkg: tmp_path, raising=True)
    b = Band(name="S", center_nm=1000.0, fwhm_nm=20.0, response="srf", srf_file="one_row_srf.csv", srf_band="S")
    try:
        with pytest.raises(ValueError, match=r"'S'.*no overlap"):
            band_weights(b, GRID)
    finally:
        convolve_mod.load_srf.cache_clear()


def test_srf_past_grid_edge_is_normalised_by_full_curve(monkeypatch, tmp_path):
    # Flat measured curve 4000-5010 nm: 1010 nm of response, 1000 nm of it inside the grid (ends 5000 nm).
    rows = ["band,wavelength_nm,response"] + [f"E,{w},1.0" for w in range(4000, 5011, 10)]
    (tmp_path / "edge_srf.csv").write_text("\n".join(rows) + "\n")
    convolve_mod.load_srf.cache_clear()
    monkeypatch.setattr(convolve_mod.resources, "files", lambda _pkg: tmp_path, raising=True)
    b = Band(name="E", center_nm=4505.0, fwhm_nm=1010.0, response="srf", srf_file="edge_srf.csv", srf_band="E")
    try:
        w = band_weights(b, GRID)
    finally:
        convolve_mod.load_srf.cache_clear()
    assert w.sum() == pytest.approx(1000.0 / 1010.0, rel=1e-9)
    assert w[np.argmin(np.abs(GRID - 4995.0))] == pytest.approx(10.0 / 1010.0, rel=1e-9)
