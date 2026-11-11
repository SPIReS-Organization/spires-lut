"""snow_lap_mixture: shapes, ranges and the physics every engine depends on. Needs the OP data."""
import dataclasses

import numpy as np
import pytest

from spires_lut import optics

pytestmark = pytest.mark.skipif(not optics.OP_DIR.exists(), reason="biosnicar optical-property data not downloaded")


def _mix(rds=300.0, lap=0.0, **kw):
    kwargs = dict(rds_snw=[rds], rho_snw=[300.0], dz=[100.0], lap_type="dust_skiles_size3",
                  mss_cnc_lap=[lap], mss_cnc_snw_alg=[0.0], lw_frac=[0.0], grain_shape=1)
    kwargs.update(kw)
    return optics.snow_lap_mixture(**kwargs)


def test_shapes_and_wavelength_grid():
    m = _mix()
    assert m.wavelength_um.shape == (475,)
    assert m.wavelength_um[0] == pytest.approx(0.255) and m.wavelength_um[-1] == pytest.approx(4.995)
    for name in ("tau", "ssa", "g", "ssa_snw", "g_snw", "mac_snw", "tau_snw"):
        assert getattr(m, name).shape == (1, 475), name
    assert m.L_snw.shape == (1,)
    assert m.ssa_lap.shape == m.mac_lap.shape == m.g_lap.shape == (2, 475)
    assert m.MSSlap.shape == m.L_lap.shape == (1, 2)
    assert m.tau_lap.shape == (1, 2, 475)


def test_ssa_and_g_strictly_inside_unit_interval():
    m = _mix()
    assert np.all((m.ssa > 0) & (m.ssa < 1))
    assert np.all((m.g > 0) & (m.g < 1))
    assert np.all(np.isfinite(m.tau)) and np.all(m.tau > 0)


def test_lap_lowers_visible_ssa():
    clean, dusty = _mix(lap=0.0), _mix(lap=1000.0)
    vis = (clean.wavelength_um >= 0.4) & (clean.wavelength_um <= 0.7)
    assert np.all(dusty.ssa[0, vis] < clean.ssa[0, vis])


def test_larger_grain_lowers_nir_ssa():
    small, large = _mix(rds=100.0), _mix(rds=1000.0)
    nir = (small.wavelength_um >= 0.8) & (small.wavelength_um <= 1.4)
    assert np.all(large.ssa[0, nir] < small.ssa[0, nir])


def test_mixture_is_frozen():
    m = _mix()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.tau = None  # type: ignore[misc]


def test_invalid_grain_radius_raises():
    with pytest.raises(ValueError, match="Grain size"):
        _mix(rds=300.5)


def test_two_stream_run_point_spectrum():
    from spires_lut import two_stream

    rho_dir_dir, rho_dif_dir, rho_dir_dif, rho_dif_dif, wl_nm = two_stream.run_point(
        sza=30.0, grain_radius=[300.0], lap_concentration=[0.0], white_sky=False)
    assert rho_dir_dir is None and rho_dif_dir is None and rho_dif_dif is None
    assert wl_nm.shape == (475,) and wl_nm[0] == pytest.approx(255.0) and wl_nm[-1] == pytest.approx(4995.0)
    assert rho_dir_dif.shape == (475,)
    assert np.all((rho_dir_dif > 0) & (rho_dir_dif <= 1))
    vis = (wl_nm >= 400) & (wl_nm <= 700)
    swir = (wl_nm >= 1500) & (wl_nm <= 2500)
    assert rho_dir_dif[vis].mean() > 0.9 > rho_dir_dif[swir].mean()


def test_two_stream_run_point_ignores_view_geometry_and_does_not_mutate_inputs():
    from spires_lut import two_stream

    lwc = [10.0]
    a = two_stream.run_point(sza=45.0, vza=15.0, raa=90.0, lwc=lwc)[2]
    b = two_stream.run_point(sza=45.0, vza=0.0, raa=0.0, lwc=lwc)[2]
    np.testing.assert_array_equal(a, b)
    assert lwc == [10.0]


def test_two_stream_run_point_guards():
    from spires_lut import two_stream

    with pytest.raises(ValueError, match="n_layers"):
        two_stream.run_point(n_layers=2, grain_radius=[300.0, 300.0])
    with pytest.raises(ValueError, match="dust"):
        two_stream.run_point(lap_type="snow_algae_empirical_Chevrollier2023")
