"""Tests for the interpolation helpers in spires_lut.pyrt.aerosol."""
import sys
import types

try:
    import disort  # noqa: F401  compiled extension, not built in spipy14
except ImportError:
    # spires_lut/pyrt/__init__.py eagerly imports surface.py, which does
    # `from disort import disobrdf`. aerosol.py never touches disort, so a
    # placeholder module lets the test import aerosol without the extension.
    _stub = types.ModuleType("disort")
    _stub.disobrdf = None
    sys.modules["disort"] = _stub

import numpy as np  # noqa: E402
from spires_lut.pyrt.aerosol import _Interpolator  # noqa: E402


def test_linear_matches_nearest_neighbor_on_grid_points():
    rng = np.random.default_rng(0)
    pgrid = np.array([0.1, 1.0, 10.0])
    wgrid = np.array([0.4, 0.6, 0.8, 1.0])
    coeff = rng.random((2, pgrid.size, wgrid.size))

    interp = _Interpolator(coeff, pgrid, wgrid, pgrid.copy(), wgrid.copy())

    np.testing.assert_allclose(interp.linear(), interp.nearest_neighbor())


def test_linear_is_bilinear_and_clamps_outside_grid():
    # f(p, w) = 2p + w on the unit square. Bilinear interpolation of a linear
    # function is exact, so the expected values are known in closed form.
    pgrid = np.array([0.0, 1.0])
    wgrid = np.array([0.0, 1.0])
    coeff = np.array([[[0.0, 1.0],
                       [2.0, 3.0]]])  # coeff[moment, p_idx, w_idx]

    particle_size = np.array([0.5, 2.0])  # 2.0 lies outside the grid
    wavelength = np.array([0.25])

    out = _Interpolator(coeff, pgrid, wgrid, particle_size, wavelength).linear()

    # Inside: f(0.5, 0.25) = 1.25. Outside: p=2.0 clamps to the nearest edge
    # p=1.0, giving f(1.0, 0.25) = 2.25. Linear extrapolation would give 4.25.
    np.testing.assert_allclose(out[0], np.array([[1.25], [2.25]]))
