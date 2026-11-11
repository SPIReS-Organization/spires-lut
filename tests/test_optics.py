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
