"""create_spires_lut without an atmospheric LUT writes the black-sky reflectance LUT only."""
import numpy as np
import pytest

from spires_lut import simulate
from spires_lut.io import read_lut

WL_NM = np.arange(255.0, 4995.0001, 10.0)  # what run_point returns after the wrapper drops 5 bands


def _stub_run_point(calls):
    def run_point(sza=45.0, grain_radius=(300.0,), lap_concentration=(1000.0,), white_sky=True, **_):
        calls.append(white_sky)
        # smooth spectrum shaped by the inputs so each grid cell is distinguishable
        spec = 0.9 * np.exp(-((WL_NM - 500.0) / 1500.0) ** 2)
        spec = spec * (1 - 1e-3 * sza) * (1 - 1e-4 * grain_radius[0]) * (1 - 1e-5 * lap_concentration[0])
        white = None if not white_sky else spec * 0.99
        return spec * 0.98, white, spec, white, WL_NM
    return run_point


@pytest.fixture
def small_grid_kwargs():
    return dict(lap_points=2, illumination_angle_points=2, grain_radius_max=40.0)


def test_without_atm_writes_reflectance_only(tmp_path, monkeypatch, small_grid_kwargs):
    calls = []
    monkeypatch.setattr(simulate, "run_point", _stub_run_point(calls))

    simulate.create_spires_lut(tmp_path, atm_lut=None, **small_grid_kwargs)

    assert not (tmp_path / "spires_albedo_lut.nc").exists()
    ds = read_lut(tmp_path / "spires_reflectance_lut.nc")
    assert dict(ds.sizes) == {"band": 475, "solar_angle": 2, "lap_concentration": 2, "sqrt_grain_radius": 2}
    assert ds["lap_concentration"].attrs["lap_type"] == "dust"
    # the first call is the wavelength probe (white_sky default); every grid call must skip the white-sky run
    assert calls[1:] and not any(calls[1:])


def test_without_atm_stores_black_sky_dhr(tmp_path, monkeypatch, small_grid_kwargs):
    monkeypatch.setattr(simulate, "run_point", _stub_run_point([]))
    simulate.create_spires_lut(tmp_path, atm_lut=None, **small_grid_kwargs)
    ds = read_lut(tmp_path / "spires_reflectance_lut.nc")

    sza = float(ds["solar_angle"][1])
    r = float(np.round(ds["sqrt_grain_radius"][1] ** 2))
    lap = float(ds["lap_concentration"][1])
    expected = 0.9 * np.exp(-((WL_NM - 500.0) / 1500.0) ** 2) * (1 - 1e-3 * sza) * (1 - 1e-4 * r) * (1 - 1e-5 * lap)
    np.testing.assert_allclose(ds["reflectance"][:, 1, 1, 1].values, expected.astype(np.float32), rtol=1e-6)


def test_grain_grid_defaults_unchanged():
    g = simulate.grain_radius_grid()
    assert g[0] == 30.0 and g[-1] == 1500.0 and len(g) == 148
