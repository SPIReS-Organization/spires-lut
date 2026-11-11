"""Delta-Eddington direct-beam albedo of a semi-infinite snow layer (Wiscombe & Warren 1980).

Second spires-lut reflectance engine. It evaluates a closed form on the same snow + LAP
optical mixture that DISORT uses (``spires_lut.optics``), so differences between the two
LUTs isolate the solver approximation. Runs the production grid in seconds on one core.

Known limitation: the delta-Eddington direct-beam albedo degrades at grazing illumination
(SZA >~ 75 deg). The production grid reaches 89 deg; the DISORT cross-check
(``examples/compare_engines.py``) quantifies the error there rather than hiding it.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def delta_eddington_albedo(ssa: npt.ArrayLike, g: npt.ArrayLike, mu0: npt.ArrayLike) -> np.ndarray:
    """Direct-beam albedo R(mu0) of a semi-infinite layer, WW80 eqs. 3-4 after delta-scaling.

    ``ssa`` (single-scattering albedo), ``g`` (asymmetry parameter) and ``mu0`` (cosine of the
    solar zenith angle) broadcast against each other. The optical depth drops out for a
    semi-infinite layer. ``ssa`` is clipped to [0, 1] and ``g`` to [0, 0.99] (the ceiling
    ``spires_lut.optics`` applies), so out-of-range ``ssa``/``g`` yield a finite value rather than NaN. ``mu0`` is not
    checked; callers pass cos(SZA) for SZA in [0, 90).
    ssa == 1 returns exactly 1.
    """
    w = np.clip(np.asarray(ssa, dtype=float), 0.0, 1.0)
    g = np.clip(np.asarray(g, dtype=float), 0.0, 0.99)  # same ceiling the mixture code applies; g == 1 would be 0/0
    mu0 = np.asarray(mu0, dtype=float)

    # delta-scaling (forward peak removed)
    g2 = g * g
    w_star = (1.0 - g2) * w / (1.0 - w * g2)
    g_star = g / (1.0 + g)

    a_star = 1.0 - w_star * g_star
    b_star = g_star / a_star
    xi = np.sqrt(3.0 * a_star * (1.0 - w_star))
    p = 2.0 * xi / (3.0 * a_star)

    return w_star / (1.0 + p) * (1.0 - b_star * xi * mu0) / (1.0 + xi * mu0)
