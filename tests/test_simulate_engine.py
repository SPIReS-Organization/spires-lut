"""create_spires_lut dispatches the point function by `engine`; grid/file layout is shared."""
import numpy as np
import pytest

from spires_lut import simulate, two_stream
from spires_lut.io import read_lut

WL_NM = np.arange(255.0, 4995.0001, 10.0)


def _stub_disort_run_point(calls):
    def run_point(sza=45.0, grain_radius=(300.0,), lap_concentration=(1000.0,), white_sky=True, **_):
        calls.append("disort")
        spec = 0.9 * np.exp(-((WL_NM - 500.0) / 1500.0) ** 2) * (1 - 1e-3 * sza)
        white = None if not white_sky else spec * 0.99
        return spec * 0.98, white, spec, white, WL_NM
    return run_point


def _stub_two_stream_run_point(calls):
    def run_point(sza=45.0, grain_radius=(300.0,), lap_concentration=(0.0,), white_sky=False, **_):
        calls.append("delta_eddington")
        spec = 0.8 * np.exp(-((WL_NM - 500.0) / 1500.0) ** 2) * (1 - 1e-3 * sza) * (1 - 1e-4 * grain_radius[0])
        return None, None, spec, None, WL_NM
    return run_point


@pytest.fixture
def small_grid_kwargs():
    return dict(lap_points=2, illumination_angle_points=2, grain_radius_max=40.0)


def test_delta_eddington_engine_writes_validating_reflectance_lut(tmp_path, monkeypatch, small_grid_kwargs):
    calls = []
    monkeypatch.setattr(two_stream, "run_point", _stub_two_stream_run_point(calls))
    monkeypatch.setattr(simulate, "run_point", _stub_disort_run_point(calls))

    simulate.create_spires_lut(tmp_path, engine="delta_eddington", **small_grid_kwargs)

    assert not (tmp_path / "spires_albedo_lut.nc").exists()
    ds = read_lut(tmp_path / "spires_reflectance_lut.nc")          # read_lut validates the contract
    assert dict(ds.sizes) == {"band": 475, "solar_angle": 2, "lap_concentration": 2, "sqrt_grain_radius": 2}
    assert ds.attrs["surface_model"] == "delta_eddington"
    assert calls and set(calls) == {"delta_eddington"}             # probe + every grid point, never DISORT

    sza = float(ds["solar_angle"][1])
    r = float(np.round(ds["sqrt_grain_radius"][1] ** 2))
    expected = 0.8 * np.exp(-((WL_NM - 500.0) / 1500.0) ** 2) * (1 - 1e-3 * sza) * (1 - 1e-4 * r)
    np.testing.assert_allclose(ds["reflectance"][:, 1, 0, 1].values, expected.astype(np.float32), rtol=1e-6)


def test_default_engine_is_disort_and_stamps_attr(tmp_path, monkeypatch, small_grid_kwargs):
    calls = []
    monkeypatch.setattr(two_stream, "run_point", _stub_two_stream_run_point(calls))
    monkeypatch.setattr(simulate, "run_point", _stub_disort_run_point(calls))

    simulate.create_spires_lut(tmp_path, atm_lut=None, **small_grid_kwargs)

    ds = read_lut(tmp_path / "spires_reflectance_lut.nc")
    assert ds.attrs["surface_model"] == "disort"
    assert set(calls) == {"disort"}


def test_unknown_engine_raises_before_any_work(tmp_path, monkeypatch, small_grid_kwargs):
    monkeypatch.setattr(simulate, "run_point", _stub_disort_run_point([]))
    with pytest.raises(ValueError, match="engine"):
        simulate.create_spires_lut(tmp_path, engine="snicar", **small_grid_kwargs)
    assert not any(tmp_path.iterdir())


def test_delta_eddington_with_atm_lut_raises(tmp_path, monkeypatch, small_grid_kwargs):
    monkeypatch.setattr(two_stream, "run_point", _stub_two_stream_run_point([]))
    with pytest.raises(ValueError, match="atm_lut"):
        simulate.create_spires_lut(tmp_path, engine="delta_eddington", atm_lut=tmp_path / "atm.nc",
                                   **small_grid_kwargs)
    assert not any(tmp_path.iterdir())
