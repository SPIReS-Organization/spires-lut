"""Delta-Eddington direct-beam albedo of a semi-infinite snow layer (Wiscombe & Warren 1980).

Second spires-lut reflectance engine. It evaluates a closed form on the same snow + LAP
optical mixture that DISORT uses (``spires_lut.optics``), so differences between the two
LUTs isolate the solver approximation. Runs the production grid in seconds on one core.

Known limitation: the delta-Eddington direct-beam albedo degrades at grazing illumination
(SZA >~ 75 deg). The production grid reaches 89 deg; the DISORT cross-check
(``examples/compare_engines.py``) quantifies the error there rather than hiding it.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from spires_lut import optics


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


def run_point(
    sza: float = 45.0,
    vza: float = 0.0,
    raa: float = 0.0,
    n_layers: int = 1,
    grain_shape: int = 1,
    grain_radius: List[float] = [300.0],
    lap_concentration: List[float] = [0.0],
    algae_concentration: List[float] = [0.0],
    lwc: List[float] = [0.0],
    snow_density: List[float] = [300.0],
    dz: List[float] = [100.0],
    lap_type: str = "dust_skiles_size3",
    white_sky: bool = False,
    **_ignored,
) -> Tuple[None, None, np.ndarray, None, np.ndarray]:
    """Delta-Eddington counterpart of ``simulate.run_point`` with the same return arity.

    Returns ``(rho_dir_dir, rho_dif_dir, rho_dir_dif, rho_dif_dif, wavelength_nm)`` where only
    ``rho_dir_dif`` — the black-sky directional-hemispherical reflectance of layer 0 — is
    computed; the other three are ``None`` (no BRF, no white-sky: the albedo LUT stays
    DISORT + atmosphere). ``vza``, ``raa`` and ``white_sky`` are accepted and ignored so
    ``create_spires_lut`` can call either engine with identical keyword arguments.
    Only a single, optically deep layer is supported.
    """
    if n_layers != 1 or len(grain_radius) != 1:
        raise ValueError(f"two_stream.run_point supports n_layers=1 only, got n_layers={n_layers}")
    if "dust" not in lap_type:
        raise ValueError(
            "For now, only support we only support dust for lap_type. Check spires-contract."
        )

    mix = optics.snow_lap_mixture(
        rds_snw=list(grain_radius),
        rho_snw=list(snow_density),
        dz=list(dz),
        lap_type=lap_type.lower(),
        mss_cnc_lap=list(lap_concentration),
        mss_cnc_snw_alg=list(algae_concentration),
        lw_frac=list(lwc),  # copy: snow_lap_mixture converts percent -> fraction in place
        grain_shape=grain_shape,
    )
    mu0 = np.cos(np.deg2rad(sza))
    rho_dir_dif = delta_eddington_albedo(mix.ssa[0], mix.g[0], mu0)
    return None, None, rho_dir_dif, None, mix.wavelength_um * 1000.0
