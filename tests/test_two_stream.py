"""Closed-form delta-Eddington direct-beam albedo: limits and monotonicity (no data, no extension)."""
import numpy as np
import pytest

from spires_lut.two_stream import delta_eddington_albedo


@pytest.mark.parametrize("g", [0.0, 0.5, 0.89, 0.99])
@pytest.mark.parametrize("mu0", [0.05, 0.5, 1.0])
def test_conservative_scattering_gives_unit_albedo(g, mu0):
    assert delta_eddington_albedo(1.0, g, mu0) == pytest.approx(1.0, abs=1e-12)


def test_hand_computed_value():
    # omega=0.99, g=0, mu0=1: a*=1, xi=sqrt(0.03), P=2xi/3, R = 0.99/(1+P)/(1+xi)
    xi = np.sqrt(0.03)
    expected = 0.99 / (1 + 2 * xi / 3) / (1 + xi)
    assert expected == pytest.approx(0.7565, abs=5e-4)          # the number quoted in the spec
    assert delta_eddington_albedo(0.99, 0.0, 1.0) == pytest.approx(expected, rel=1e-12)


def test_monotonic_decreasing_in_mu0():
    mu0 = np.linspace(0.02, 1.0, 50)
    r = delta_eddington_albedo(0.95, 0.85, mu0)
    assert np.all(np.diff(r) < 0)


def test_monotonic_increasing_in_ssa():
    ssa = np.linspace(0.5, 1.0, 51)
    r = delta_eddington_albedo(ssa, 0.85, 0.7)
    assert np.all(np.diff(r) > 0)


def test_broadcasting_over_spectrum_and_angles():
    ssa = np.linspace(0.6, 0.9999, 475)        # a spectrum
    g = np.full(475, 0.88)
    mu0 = np.cos(np.deg2rad(np.array([0.0, 45.0, 75.0])))[:, None]
    r = delta_eddington_albedo(ssa, g, mu0)
    assert r.shape == (3, 475)
    assert np.all((r > 0) & (r <= 1))


def test_out_of_range_inputs_are_clipped_not_nan():
    r = delta_eddington_albedo(np.array([-0.1, 1.2]), np.array([-0.5, 1.5]), 0.5)
    assert np.all(np.isfinite(r))
    assert r[1] == pytest.approx(1.0, abs=1e-12)
