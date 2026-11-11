"""Snow + LAP optical mixture shared by every spires-lut radiative-transfer engine.

Pure numpy/pandas/pooch: no solver dependency. ``snow_lap_mixture`` is the exact body that
``DISORT.__init__`` used to run inline, so the DISORT and delta-Eddington engines see
identical single-scattering albedo, asymmetry parameter and optical depth per layer.
Optical constants are biosnicar-py 3.0's 480-band Mie tables (ice spheres, Picard 2016
constants; water spheres; LAPs), downloaded once into ``OP_DIR``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pooch

OP_DIR = (
    Path.home() / ".spires" / "data" / "3.0.zip.unzip" / "biosnicar-py-3.0" / "data" / "OP_data" / "480band"
)
WVL_THRESHOLD = 5  # drop the first 5 of biosnicar's 480 bands (205-245 nm)
OP_WAVELENGTH_UM = (np.arange(205.0, 4995.0001, 10.0) / 1000.0)[WVL_THRESHOLD:]  # 475 bands, 255-4995 nm

# Fixed second impurity: snow algae are always present as a (possibly zero-concentration) LAP.
_ALGAE_STEM = "snow_algae_empirical_Chevrollier2023"


def get_optical_property_data() -> None:
    # Download biosnicar data if not already present
    # TODO we could also create our own OPs and host them at some point in the future
    pooch.create(
        path=Path.home() / ".spires" / "data",
        base_url="https://github.com/jmcook1186/biosnicar-py/archive/refs/tags/",
        registry={"3.0.zip": "sha256:1a25a07a57d622614635ff650ccf53a6a81b1534dc53b17ba1acd3a57609838e"},
        env="LUT_DATA_DIR").fetch("3.0.zip", processor=pooch.Unzip())
    return


@dataclass(frozen=True)
class Mixture:
    """Per-layer optical properties of the snow + LAP mixture on the 475-band grid.

    ``tau``, ``ssa``, ``g`` are what a solver consumes; the remaining fields are the
    component arrays ``DISORT`` exposes as attributes (kept for equivalence/debugging).
    """
    wavelength_um: np.ndarray  # (475,)
    tau: np.ndarray            # (nbr_lyr, 475) mixture optical depth
    ssa: np.ndarray            # (nbr_lyr, 475) mixture single-scattering albedo
    g: np.ndarray              # (nbr_lyr, 475) mixture asymmetry parameter
    ssa_snw: np.ndarray        # (nbr_lyr, 475)
    g_snw: np.ndarray          # (nbr_lyr, 475)
    mac_snw: np.ndarray        # (nbr_lyr, 475) mass extinction coefficient
    tau_snw: np.ndarray        # (nbr_lyr, 475)
    L_snw: np.ndarray          # (nbr_lyr,) snow mass per unit area after LAP removal
    ssa_lap: np.ndarray        # (nbr_lap, 475)
    mac_lap: np.ndarray        # (nbr_lap, 475)
    g_lap: np.ndarray          # (nbr_lap, 475)
    MSSlap: np.ndarray         # (nbr_lyr, nbr_lap) mass mixing ratio kg/kg
    L_lap: np.ndarray          # (nbr_lyr, nbr_lap)
    tau_lap: np.ndarray        # (nbr_lyr, nbr_lap, 475)


def snow_lap_mixture(
    rds_snw: list,
    rho_snw: list,
    dz: list,
    lap_type: str,
    mss_cnc_lap: list,
    mss_cnc_snw_alg: list,
    lw_frac: list,
    grain_shape: int = 1,
) -> Mixture:
    """Combine ice (+ liquid-water coating), one LAP and snow algae into per-layer tau/ssa/g.

    Args:
        rds_snw:         snow grain radius of each layer (um); must match a tabulated radius exactly
        rho_snw:         density of each layer (kg m-3)
        dz:              thickness of each layer (m)
        lap_type:        stem of the LAP optical-property arrays in lap.npz, e.g. "dust_skiles_size3"
        mss_cnc_lap:     LAP concentration per layer (micro g per g of snow)
        mss_cnc_snw_alg: snow algae concentration per layer (cells/mL)
        lw_frac:         liquid water fraction per layer (percent). NOTE: converted to a fraction
                         IN PLACE, as the original DISORT code did; pass a copy if you reuse the list.
        grain_shape:     1=sphere; 2=spheroid; 3=hexagonal plate; 4=koch snowflake (He et al. 2017)

    The number of layers is ``len(rds_snw)``.
    """
    nbr_lyr = len(rds_snw)

    # If dir doesn't exist get the data
    if not OP_DIR.exists():
        get_optical_property_data()

    # Load OPs
    ssps_ice = np.load(OP_DIR / "luts" / "ice_sphere_Pic16.npz")
    ssps_water = np.load(OP_DIR / "luts" / "water_sphere.npz")
    ssps_lap = np.load(OP_DIR / "lap.npz")

    wavelength_um = OP_WAVELENGTH_UM.copy()  # frozen dataclass does not freeze the array; don't alias the module constant
    nbr_wvl = len(wavelength_um)

    # set up empty arrays
    SSA_snw = np.empty([nbr_lyr, nbr_wvl])
    MAC_snw = np.empty([nbr_lyr, nbr_wvl])
    g_snw = np.empty([nbr_lyr, nbr_wvl])

    # Similar to SNICAR: He et al. 2017 parameterization
    g_wvl = np.array(
        [0.25, 0.70, 1.41, 1.90, 2.50, 3.50, 4.00, 5.00]
    )  # wavelength (um) division point
    g_wvl_center = (
        g_wvl[1:8] / 2 + g_wvl[0:7] / 2
    )  # % center point for wavelength band
    g_b0 = np.array(
        [
            9.76029e-01,
            9.67798e-01,
            1.00111e00,
            1.00224e00,
            9.64295e-01,
            9.97475e-01,
            9.97475e-01,
        ]
    )
    g_b1 = np.array(
        [
            5.21042e-01,
            4.96181e-01,
            1.83711e-01,
            1.37082e-01,
            5.50598e-02,
            8.48743e-02,
            8.48743e-02,
        ]
    )
    g_b2 = np.array(
        [
            -2.66792e-04,
            1.14088e-03,
            2.37011e-04,
            -2.35905e-04,
            8.40449e-04,
            -4.71484e-04,
            -4.71484e-04,
        ]
    )

    # Tables 1 & 2 and Eqs. 3.1-3.4 from Fu, 2007
    g_F07_c2 = np.array(
        [1.349959e-1, 1.115697e-1, 9.853958e-2, 5.557793e-2, -1.233493e-1, 0.0, 0.0]
    )
    g_F07_c1 = np.array(
        [
            -3.987320e-1,
            -3.723287e-1,
            -3.924784e-1,
            -3.259404e-1,
            4.429054e-2,
            -1.726586e-1,
            -1.726586e-1,
        ]
    )
    g_F07_c0 = np.array(
        [
            7.938904e-1,
            8.030084e-1,
            8.513932e-1,
            8.692241e-1,
            7.085850e-1,
            6.412701e-1,
            6.412701e-1,
        ]
    )
    g_F07_p2 = np.array(
        [
            3.165543e-3,
            2.014810e-3,
            1.780838e-3,
            6.987734e-4,
            -1.882932e-2,
            -2.277872e-2,
            -2.277872e-2,
        ]
    )
    g_F07_p1 = np.array(
        [
            1.140557e-1,
            1.143152e-1,
            1.143814e-1,
            1.071238e-1,
            1.353873e-1,
            1.914431e-1,
            1.914431e-1,
        ]
    )
    g_F07_p0 = np.array(
        [
            5.292852e-1,
            5.425909e-1,
            5.601598e-1,
            6.023407e-1,
            6.473899e-1,
            4.634944e-1,
            4.634944e-1,
        ]
    )

    # read in single scattering albedo, mass extinction coefficients (MAC), and asymmetry parameter (g) for ice
    # crystals in each layer, optional with coated liquid water spheres
    for i in np.arange(0, nbr_lyr, 1):

        ice_diff = np.abs(ssps_ice["radii"] - rds_snw[i])
        water_diff = np.abs(ssps_water["radii"] - rds_snw[i])

        if ice_diff.min() > 1e-5 or water_diff.min() > 1e-5:
            radii = ssps_ice["radii"]
            raise ValueError(
                f"Grain size {rds_snw[i]} is invalid.\n"
                f"Available range: {radii.min()} to {radii.max()}.\n"
                "Note: Spacing is 1.0 (from 30 to 1500) and 20.0 (from 1500 to 5000).\n"
                "Please choose an exact value matching the pre-computed optical property data."
            )
        ice_idx = ice_diff.argmin()
        water_idx = water_diff.argmin()

        # NOTE: this is the updated code chunk for settings g, ssa, and MAC.
        # apply LWC to OP , weighted volume average
        lw_frac[i] = lw_frac[i] / 100
        # neglecting air mass:
        # Air has refractive index of approx. 1.
        vlm_frac_ice = 1 - lw_frac[i]
        # vlm_frac_ice = (rho_snw[i] - lw_frac[i] * 1000) / 917
        # vlm_frac_air = 1 - lw_frac[i] - vlm_frac_ice

        w_asm = ssps_water["asm_prm"][water_idx, WVL_THRESHOLD:]
        i_asm = ssps_ice["asm_prm"][ice_idx, WVL_THRESHOLD:]

        w_ext = ssps_water["ext_cff_vlm"][water_idx, WVL_THRESHOLD:]
        i_ext = ssps_ice["ext_cff_vlm"][ice_idx, WVL_THRESHOLD:]

        w_sca = ssps_water["sca_cff_vlm"][water_idx, WVL_THRESHOLD:]
        i_sca = ssps_ice["sca_cff_vlm"][ice_idx, WVL_THRESHOLD:]

        # Asymmetry Parameter (g)
        g = (w_asm * lw_frac[i] + i_asm * vlm_frac_ice) / (
            lw_frac[i] + vlm_frac_ice
        )
        g[g <= 0] = 0.00001
        g[g > 0.99] = 0.99
        g_snw[i, :] = g

        # Mass Extinction Coefficient (MAC)
        ext_cff_mss = (w_ext * lw_frac[i] + i_ext * vlm_frac_ice) / 917
        MAC_snw[i, :] = ext_cff_mss

        # Single Scattering Albedo (SSA)
        ssa = ((w_sca * lw_frac[i] + i_sca * vlm_frac_ice) / 917) / MAC_snw[i, :]
        ssa[ssa <= 0] = 0.00000001
        ssa[ssa >= 1] = 0.99999999
        SSA_snw[i, :] = ssa

        # adjust single scattering albedo and g to match updated grain shape
        if grain_shape != 1:

            params = {
                2: (0.929, 0.5, False), # Spheroid
                3: (0.788, 2.5, True),  # Hex Plate
                4: (0.712, 2.5, True)   # Koch
            }
            fs_val, ar_val, use_log = params[grain_shape]

            # Diameter scaling
            diam_ice = 2.0 * rds_snw[i]
            if grain_shape == 4:
                diam_ice /= 0.544

            # Compute g_ice_Cg_tmp (Eq.7)
            g_ice_Cg_tmp = g_b0 * (fs_val / 0.788)**g_b1 * diam_ice**g_b2

            # Compute gg_ice_F07_tmp (Eq. 3.1 or 3.3)
            if not use_log:
                # Spheroid: Quadratic in AR
                gg_ice_F07_tmp = g_F07_c0 + g_F07_c1 * ar_val + g_F07_c2 * ar_val**2
            else:
                # Hex/Koch: Polynomial in log(AR)
                log_ar = np.log(ar_val)
                gg_ice_F07_tmp = g_F07_p0 + g_F07_p1 * log_ar + g_F07_p2 * log_ar**2

            # Resample
            g_ice_Cg_tmp = wvl_resample(g_ice_Cg_tmp, g_wvl_center, wavelength_um, np.full_like(wavelength_um, 0.11))
            gg_ice_F07_tmp = wvl_resample(gg_ice_F07_tmp, g_wvl_center, wavelength_um, np.full_like(wavelength_um, 0.11))

            g_ice_F07 = gg_ice_F07_tmp + (1.0 - gg_ice_F07_tmp) / SSA_snw[i, :] / 2

            g_snw[i, :] = g_ice_F07 * g_ice_Cg_tmp
            g_snw[i, :][g_snw[i, :] < 0.1] = np.nan
            g_snw[i, :][g_snw[i, :] > 1.0] = np.nan
            g_snw[i, :] = pd.Series(g_snw[i, :]).interpolate(method="linear")

    # define total number of different LAPs in model. For now, only one type of lap and snow algae
    # NOTE snow algae is fixed for now
    lap_stems = [lap_type, _ALGAE_STEM]
    nbr_lap = len(lap_stems)

    # load mass concentrations per layer into numpy array (one row per layer, one column per impurity) and
    # convert to kg/kg unit
    MSSlap = np.zeros([nbr_lyr, nbr_lap])
    # lap in micro g / g
    MSSlap[0:nbr_lyr, 0] = np.asarray(mss_cnc_lap) * 1e-6  # lap
    # snow algae given as concentration in units of cells/mL
    MSSlap[0:nbr_lyr, 1] = (
        np.asarray(mss_cnc_snw_alg) / 917 * 1e-3
    )  # snow algae

    # read in LAP optical properties (single scattering albedo, mass extinction coefficient, asymmetry parameter)
    SSA_lap = np.zeros([nbr_lap, nbr_wvl])
    MAC_lap = np.zeros([nbr_lap, nbr_wvl])
    g_lap = np.zeros([nbr_lap, nbr_wvl])

    # drop the same leading WVL_THRESHOLD bands as for the ice optical properties
    for ii, stem in enumerate(lap_stems):
        SSA_lap[ii, :] = ssps_lap[f"{stem}__ss_alb"][WVL_THRESHOLD:]
        MAC_lap[ii, :] = ssps_lap[f"{stem}__ext_cff_mss"][WVL_THRESHOLD:]
        g_lap[ii, :] = ssps_lap[f"{stem}__asm_prm"][WVL_THRESHOLD:]

    # initialize arrays
    L_snw = np.zeros(nbr_lyr)
    tau_snw = np.zeros([nbr_lyr, nbr_wvl])
    L_lap = np.zeros([nbr_lyr, nbr_lap])
    tau_lap = np.zeros([nbr_lyr, nbr_lap, nbr_wvl])
    tau = np.zeros([nbr_lyr, nbr_wvl])
    SSA = np.zeros([nbr_lyr, nbr_wvl])
    g_mix = np.zeros([nbr_lyr, nbr_wvl])

    # calculate effective tau (optical depth), SSA (single scattering albedo) and g (asymmetry parameter) for the
    # snow + LAP mixture. SSA and g for the individual components have been calculated using Mie theory and
    # stored in a netcdf file. Here, these values are combined to give an overall SSA and g for the snow + LAP
    # mixture. For each layer, the layer mass (L) is density * layer thickness. For each layer, the optical depth is
    # the layer mass * the mass extinction coefficient
    for i in range(nbr_lyr):
        # Snow layer mass
        L_snw[i] = dz[i] * rho_snw[i]

        # Reset sums for this layer
        tau_sum_layer = np.zeros(nbr_wvl)
        SSA_sum_layer = np.zeros(nbr_wvl)
        g_sum_layer = np.zeros(nbr_wvl)

        for j in range(nbr_lap):
            # LAP mass per unit area
            L_lap[i, j] = L_snw[i] * MSSlap[i, j]

            # Optical depth of LAP
            tau_lap[i, j, :] = L_lap[i, j] * MAC_lap[j, :]

            # Accumulate layer sums
            tau_sum_layer += tau_lap[i, j, :]
            SSA_sum_layer += tau_lap[i, j, :] * SSA_lap[j, :]
            g_sum_layer += (
                tau_lap[i, j, :] * SSA_lap[j, :] * g_lap[j, :]
            )

            # Reduce snow mass to account for LAP
            if j == 0:
                L_snw[i] -= L_lap[i, j]
            else:
                L_snw[i] -= L_lap[i, j] * 1e-12

        # Snow optical depth
        tau_snw[i, :] = L_snw[i] * MAC_snw[i, :]

        # Total optical depth
        tau[i, :] = tau_sum_layer + tau_snw[i, :]

        # Effective SSA
        SSA[i, :] = (
            SSA_sum_layer + SSA_snw[i, :] * tau_snw[i, :]
        ) / tau[i, :]

        # Effective asymmetry parameter g
        g_mix[i, :] = (
            g_sum_layer + g_snw[i, :] * SSA_snw[i, :] * tau_snw[i, :]
        ) / (tau[i, :] * SSA[i, :])

    return Mixture(
        wavelength_um=wavelength_um,
        tau=tau,
        ssa=SSA,
        g=g_mix,
        ssa_snw=SSA_snw,
        g_snw=g_snw,
        mac_snw=MAC_snw,
        tau_snw=tau_snw,
        L_snw=L_snw,
        ssa_lap=SSA_lap,
        mac_lap=MAC_lap,
        g_lap=g_lap,
        MSSlap=MSSlap,
        L_lap=L_lap,
        tau_lap=tau_lap,
    )


def generate_filter(wvl_m, wvl, wl_resol):
    """
    Generate norm array for wavelength resampling.

    :param wvl_m:    start wavelength grid
    :param wvl:      final wavelength grid
    :param wl_resol: resolution of final wavelength grid
    :return:         resampling norm
    """
    num_wvl_m = len(wvl_m)
    num_wvl = len(wvl)

    s_norm_m = np.zeros((num_wvl_m, num_wvl))
    exp_max = 2.
    exp_min = 2.
    exp_arr = exp_max + (exp_min - exp_max) * np.arange(0, 2100, 1) / (num_wvl - 1)
    c_arr = (1 / (2 ** exp_arr * np.log(2))) ** (1 / exp_arr)

    for bd in range(num_wvl):
        li1 = np.logical_and(wvl_m >= (wvl[bd] - 2. * wl_resol[bd]), wvl_m <= (wvl[bd] + 2. * wl_resol[bd]))
        li1 = np.where(li1)
        cnt = len(li1)

        if cnt > 0:
            tmp = np.abs(wvl[bd] - wvl_m[li1]) / (wl_resol[bd] * c_arr[bd])
            s = np.exp(-(tmp ** exp_arr[bd]))
            s_norm_m[li1, bd] = s / np.sum(s)

    return s_norm_m


def wvl_resample(specs, wvl_m, wvl, wl_resol):
    """
    Resample given spectra to desired wavelength grid.

    :param specs:    input spectra, can be single spectrum or array of spectra
    :param wvl_m:    input center wavelengths
    :param wvl:      output center wavelengths
    :param wl_resol: output fwhm
    :return:         input spectra resampled to desired wavelength grid
    """
    s_norm = generate_filter(wvl_m=wvl_m, wvl=wvl, wl_resol=wl_resol)
    specs_resample = specs @ s_norm

    return specs_resample
