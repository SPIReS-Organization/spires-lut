import numpy as np
from pathlib import Path
import pandas as pd

from spires_lut.pyrt.aerosol import HenyeyGreenstein, TabularLegendreCoefficients
from spires_lut.pyrt.controller import ComputationalParameters, ModelBehavior
from spires_lut.pyrt.eos import Hydrostatic
from spires_lut.pyrt.observation import Spectral
from spires_lut.pyrt.output import OutputArrays, OutputBehavior, UserLevel
from spires_lut.pyrt.radiation import IncidentFlux, ThermalEmission
from spires_lut.pyrt.surface import Surface
from spires_lut.common import *

import disort


class DISORT(object):
    """Class that holds all necessary attributes and functions to apply the multistream radiative transfer solver
    DISORT to given Mie calculated snow optical properties. In particular, this class supports the inclusion of liquid
    water coatings as well as lap and snow algae.
    """

    def __init__(
        self,
        sza: float,
        saa: float,
        vza: np.array,
        vaa: np.array,
        e_dir: np.ndarray,
        e_dif: np.ndarray,
        wvl: np.ndarray,
        fwhm: np.ndarray,
        nbr_lyr: int,
        rho_snw: list,
        dz: list,
        lap_type: str,
        mss_cnc_lap: list,
        mss_cnc_snw_alg: list,
        rds_snw: list = None,
        lw_frac: list = None,
        grain_shape: int = 1,
    ):
        """Instance of Disort.

        Args:
            sza:             solar zenith angle (degrees)
            saa:             solar azimuth angle (degrees)
            vza:             view zenith angle (degrees)
            vaa:             view azimuth angle (degrees)
            pa:              phase angle (degrees)
            ea:              emission angle (degrees)
            e_dir:           normalized direct incident flux (values between 0 and 1), per instrument wavelength
            e_dif:           normalized diffuse incident flux (must be 1 - edir), per instrument wavelength
            wvl:             instrument wavelengths
            fwhm:            instrument full-width-half-max
            cryo_type:       surface type, options are "snow" and "ice"
            nbr_lyr:         number of horizontal snow surface layers
            rho_snw:         density of each layer (unit = kg m-3), list must have len(nbr_lyr)
            dz:              thickness of each layer (unit = m), list must have len(nbr_lyr)
            lap_type:        file name of lap optical properties
            mss_cnc_lap:     lap in micro g per g of snow
            mss_cnc_snw_alg: snow algae concentration of each layer in units of cells/mL, list must have len(nbr_lyr)
            rds_snw:         snow grain radius of each layer, list must have len(nbr_lyr), only if cryo_type = 'snow'
            lw_frac:         liquid water fraction
            grain_shape:     1=sphere; 2=spheroid; 3=hexagonal plate; 4=koch snowflake
        """
        # get all needed angles
        self.UMU = np.cos(np.deg2rad(vza))
        self.UMU0 = np.cos(np.deg2rad(sza))
        self.PHI = vaa
        self.PHI0 = saa

        # set up VAA convention for SPIReS
        self.PHI = 180 - self.PHI

        # set base directory for auxiliary files
        self.dir_base = (
            Path.home()
            / ".spires"
            / "data"
            / "3.0.zip.unzip"
            / "biosnicar-py-3.0"
            / "data"
            / "OP_data"
            / "480band"
        )
        self.wvl_threshold = 10

        # Load OPs
        ssps_ice = np.load(self.dir_base / "luts" / "ice_sphere_Pic16.npz")
        ssps_water = np.load(self.dir_base / "luts" / "water_sphere.npz")
        ssps_lap = np.load(self.dir_base / "lap.npz")

        self.wvl_op = np.arange(205.0, 4995.0001, 10.0) / 1000.0  # 480 band biosnicar
        self.wvl_op = self.wvl_op[self.wvl_threshold :]
        nbr_wvl_op = len(self.wvl_op)

        # convert wavelengths to wavenumbers
        wvl = wvl / 1000  # to microns
        fwhm = fwhm / 1000  # to microns
        self.wvl = wvl
        self.fwhm = fwhm
        self.nbr_wvl = len(self.wvl)

        short_wvl = self.wvl - fwhm / 2
        long_wvl = self.wvl + fwhm / 2

        spectral = Spectral(short_wavelength=short_wvl, long_wavelength=long_wvl)

        self.WVNMHI = spectral.high_wavenumber
        self.WVNMLO = spectral.low_wavenumber

        # set up empty arrays
        SSA_snw = np.empty([nbr_lyr, nbr_wvl_op])
        MAC_snw = np.empty([nbr_lyr, nbr_wvl_op])
        g_snw = np.empty([nbr_lyr, nbr_wvl_op])

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

            w_asm = ssps_water["asm_prm"][water_idx, self.wvl_threshold :]
            i_asm = ssps_ice["asm_prm"][ice_idx, self.wvl_threshold :]

            w_ext = ssps_water["ext_cff_vlm"][water_idx, self.wvl_threshold :]
            i_ext = ssps_ice["ext_cff_vlm"][ice_idx, self.wvl_threshold :]

            w_sca = ssps_water["sca_cff_vlm"][water_idx, self.wvl_threshold :]
            i_sca = ssps_ice["sca_cff_vlm"][ice_idx, self.wvl_threshold :]

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
                g_ice_Cg_tmp = wvl_resample(g_ice_Cg_tmp, g_wvl_center, self.wvl_op, np.full_like(self.wvl_op, 0.1))
                gg_ice_F07_tmp = wvl_resample(gg_ice_F07_tmp, g_wvl_center, self.wvl_op, np.full_like(self.wvl_op, 0.1))
                
                g_ice_F07 = gg_ice_F07_tmp + (1.0 - gg_ice_F07_tmp) / SSA_snw[i, :] / 2
                
                g_snw[i, :] = g_ice_F07 * g_ice_Cg_tmp
                g_snw[i, :][g_snw[i, :] < 0.1] = np.nan
                g_snw[i, :][g_snw[i, :] > 1.0] = np.nan
                g_snw[i, :] = pd.Series(g_snw[i, :]).interpolate(method="linear")


        # resample wavelength grid of snow optical properties to given instrument wavelengths
        self.SSA_snw = wvl_resample(SSA_snw, self.wvl_op, self.wvl, self.fwhm)
        self.MAC_snw = wvl_resample(MAC_snw, self.wvl_op, self.wvl, self.fwhm)
        self.g_snw = wvl_resample(g_snw, self.wvl_op, self.wvl, self.fwhm)

        # define total number of different LAPs in model. For now, only one type of lap and snow algae
        # NOTE snow algae is fixed for now
        lap_stems = [lap_type, "snow_algae_empirical_Chevrollier2023"]
        nbr_lap = len(lap_stems)

        # load mass concentrations per layer into numpy array (one row per layer, one column per impurity) and
        # convert to kg/kg unit
        self.MSSlap = np.zeros([nbr_lyr, nbr_lap])
        # lap in micro g / g
        self.MSSlap[0:nbr_lyr, 0] = np.asarray(mss_cnc_lap) * 1e-6  # lap
        # snow algae given as concentration in units of cells/mL
        self.MSSlap[0:nbr_lyr, 1] = (
            np.asarray(mss_cnc_snw_alg) / 917 * 1e-3
        )  # snow algae

        # read in LAP optical properties (single scattering albedo, mass extinction coefficient, asymmetry parameter)
        SSA_lap = np.zeros([nbr_lap, nbr_wvl_op])
        MAC_lap = np.zeros([nbr_lap, nbr_wvl_op])
        g_lap = np.zeros([nbr_lap, nbr_wvl_op])

        # skip the first ten wavelengths for now as we have different shapes for the ice optical properties
        for ii, stem in enumerate(lap_stems):
            SSA_lap[ii, :] = ssps_lap[f"{stem}__ss_alb"][self.wvl_threshold :]
            MAC_lap[ii, :] = ssps_lap[f"{stem}__ext_cff_mss"][self.wvl_threshold :]
            g_lap[ii, :] = ssps_lap[f"{stem}__asm_prm"][self.wvl_threshold :]

        # resample wavelength grid of LAP optical properties to given instrument wavelengths
        self.SSA_lap = wvl_resample(SSA_lap, self.wvl_op, self.wvl, self.fwhm)
        self.MAC_lap = wvl_resample(MAC_lap, self.wvl_op, self.wvl, self.fwhm)
        self.g_lap = wvl_resample(g_lap, self.wvl_op, self.wvl, self.fwhm)

        # initialize arrays
        self.L_snw = np.zeros(nbr_lyr)
        self.tau_snw = np.zeros([nbr_lyr, self.nbr_wvl])
        self.L_lap = np.zeros([nbr_lyr, nbr_lap])
        self.tau_lap = np.zeros([nbr_lyr, nbr_lap, self.nbr_wvl])
        self.tau_sum = np.zeros([nbr_lyr, self.nbr_wvl])
        self.SSA_sum = np.zeros([nbr_lyr, self.nbr_wvl])
        self.g_sum = np.zeros([nbr_lyr, self.nbr_wvl])
        self.tau = np.zeros([nbr_lyr, self.nbr_wvl])
        self.SSA = np.zeros([nbr_lyr, self.nbr_wvl])
        self.g = np.zeros([nbr_lyr, self.nbr_wvl])

        # calculate effective tau (optical depth), SSA (single scattering albedo) and g (asymmetry parameter) for the
        # snow + LAP mixture. SSA and g for the individual components have been calculated using Mie theory and
        # stored in a netcdf file. Here, these values are combined to give an overall SSA and g for the snow + LAP
        # mixture. For each layer, the layer mass (L) is density * layer thickness. For each layer, the optical depth is
        # the layer mass * the mass extinction coefficient
        for i in range(nbr_lyr):
            # Snow layer mass
            self.L_snw[i] = dz[i] * rho_snw[i]

            # Reset sums for this layer
            tau_sum_layer = np.zeros(self.nbr_wvl)
            SSA_sum_layer = np.zeros(self.nbr_wvl)
            g_sum_layer = np.zeros(self.nbr_wvl)

            for j in range(nbr_lap):
                # LAP mass per unit area
                self.L_lap[i, j] = self.L_snw[i] * self.MSSlap[i, j]

                # Optical depth of LAP
                self.tau_lap[i, j, :] = self.L_lap[i, j] * self.MAC_lap[j, :]

                # Accumulate layer sums
                tau_sum_layer += self.tau_lap[i, j, :]
                SSA_sum_layer += self.tau_lap[i, j, :] * self.SSA_lap[j, :]
                g_sum_layer += (
                    self.tau_lap[i, j, :] * self.SSA_lap[j, :] * self.g_lap[j, :]
                )

                # Reduce snow mass to account for LAP
                if j == 0:
                    self.L_snw[i] -= self.L_lap[i, j]
                else:
                    self.L_snw[i] -= self.L_lap[i, j] * 1e-12

            # Snow optical depth
            self.tau_snw[i, :] = self.L_snw[i] * self.MAC_snw[i, :]

            # Total optical depth
            self.tau[i, :] = tau_sum_layer + self.tau_snw[i, :]

            # Effective SSA
            self.SSA[i, :] = (
                SSA_sum_layer + self.SSA_snw[i, :] * self.tau_snw[i, :]
            ) / self.tau[i, :]

            # Effective asymmetry parameter g
            self.g[i, :] = (
                g_sum_layer + self.g_snw[i, :] * self.SSA_snw[i, :] * self.tau_snw[i, :]
            ) / (self.tau[i, :] * self.SSA[i, :])

        # for multi-stream models, we don't need to apply the delta-scaling as we do for two-stream approximations:
        # tau_star = (1 - (self.SSA * (self.g**2))) * self.tau
        # SSA_star = ((1 - (self.g**2)) * self.SSA) / (1 - (self.SSA * (self.g**2)))
        # g_star = self.g / (1 + self.g)
        g_star = self.g
        SSA_star = self.SSA
        tau_star = self.tau

        # calculate total optical depth of entire column, i.e., tau_clm = total optical depth from lower boundary of
        # the first layer to upper boundary of layer n. This is therefore a cumulative quantity - subsequently, lower
        # layers contain the sum of the optical depth of all overlying layers.
        tau_clm = np.zeros([nbr_lyr, self.nbr_wvl])

        for i in np.arange(1, nbr_lyr, 1):
            # start loop from 2nd layer, i.e., index = 1
            tau_clm[i, :] = tau_clm[i - 1, :] + tau_star[i - 1, :]

        # turn the asymmetry parameter into Legendre coefficients and calculate the phase function of the snow + LAP
        # mixture, using 20 Legendre moments according to Painter & Dozier (2004a)
        hg = HenyeyGreenstein(asymmetry=g_star)
        lg_coeff = hg.legendre_decomposition(n_moments=20)
        pf = TabularLegendreCoefficients(
            coefficients=lg_coeff,
            particle_size_grid=np.array(rho_snw),
            wavelength_grid=wvl,
            particle_size_profile=np.array(rho_snw),
            wavelengths=wvl,
        )

        pf.make_nn_phase_function()
        snow_pf = pf.phase_function

        self.DTAUC = tau_star
        self.SSALB = SSA_star
        self.PMOM = snow_pf

        # temperature and height at each layer boundary. In fact, only needed for atmospheric simulations, but required
        # as input to DISORT, so just an arbitrarily selected placeholder in the snow case
        altitude_grid = np.linspace(100, 0, num=51)
        pressure_profile = 500 * np.exp(-altitude_grid / 10)
        temperature_profile = np.linspace(150, 250, num=51)
        z_grid = np.linspace(100, 0, num=15)
        mass = 7.3 * 10**-26
        gravity = 3.7

        hydro = Hydrostatic(
            altitude_grid=altitude_grid,
            pressure_grid=pressure_profile,
            temperature_grid=temperature_profile,
            altitude_boundaries=z_grid,
            particle_mass=mass,
            gravity=gravity,
        )

        TEMPER = hydro.temperature
        H_LYR = hydro.scale_height

        self.TEMPER = TEMPER[: nbr_lyr + 1]
        self.H_LYR = H_LYR[: nbr_lyr + 1]

        # set a few optional computational and model parameters
        cp = ComputationalParameters(
            n_layers=nbr_lyr,
            n_moments=20,
            n_streams=16,
            n_azimuth=1,
            n_polar=1,
            n_user_levels=nbr_lyr + 1,
        )

        mb = ModelBehavior(
            accuracy=0.0,
            delta_m_plus=True,
            do_pseudo_sphere=False,
            header="",
            print_variables=None,
            radius=6371.0,
        )

        self.ACCUR = mb.accuracy
        self.DELTAMPLUS = mb.delta_m_plus
        self.DO_PSEUDO_SPHERE = mb.do_pseudo_sphere
        self.HEADER = mb.header
        self.PRNT = mb.print_variables
        self.EARTH_RADIUS = mb.radius

        ob = OutputBehavior(
            incidence_beam_conditions=False,
            only_fluxes=False,
            user_angles=True,
            user_optical_depths=False,
        )

        self.IBCND = ob.incidence_beam_conditions
        self.ONLYFL = ob.only_fluxes
        self.USRANG = ob.user_angles
        self.USRTAU = ob.user_optical_depths

        ulv = UserLevel(n_user_levels=cp.n_user_levels, optical_depth_output=None)

        self.UTAU = ulv.optical_depth_output

        # define the beam flux and the isotropic flux at the top of the atmosphere.
        # as the corresponding incident flux is umu0 * beam_flux and pi * isotropic_flux, we set fbeam to pi and divide
        # by umu0 in order to normalize direct and diffuse irradiance
        self.FBEAM = np.zeros(self.nbr_wvl)
        self.FISOT = np.zeros(self.nbr_wvl)

        for wl in range(self.nbr_wvl):
            beam_flux = (np.pi / self.UMU0) * e_dir[wl]
            isotropic_flux = e_dif[wl]
            flux = IncidentFlux(beam_flux=beam_flux, isotropic_flux=isotropic_flux)
            self.FBEAM[wl] = flux.beam_flux
            self.FISOT[wl] = flux.isotropic_flux

        # define whether thermal emission is used in the model
        te = ThermalEmission(
            thermal_emission=False,
            bottom_temperature=0.0,
            top_temperature=0.0,
            top_emissivity=1.0,
        )

        self.PLANK = te.thermal_emission
        self.BTEMP = te.bottom_temperature
        self.TTEMP = te.top_temperature
        self.TEMIS = te.top_emissivity

        # initializing output arrays
        oa = OutputArrays(
            n_polar=cp.n_polar, n_user_levels=cp.n_user_levels, n_azimuth=cp.n_azimuth
        )

        self.ALBMED = oa.albedo_medium
        self.FLUP = oa.diffuse_up_flux
        self.RFLDN = oa.diffuse_down_flux
        self.RFLDIR = oa.direct_beam_flux
        self.DFDT = oa.flux_divergence
        self.UU = oa.intensity
        self.UAVG = oa.mean_intensity
        self.TRNMED = oa.transmissivity_medium

        # settings for underlying surface reflectance, which is 0.0 and lambertian in the snow case
        sfc = Surface(
            albedo=0.0,
            n_streams=cp.n_streams,
            n_polar=cp.n_polar,
            n_azimuth=cp.n_azimuth,
            user_angles=ob.user_angles,
            only_fluxes=ob.only_fluxes,
            n_mug=200,
        )

        sfc.make_lambertian()

        self.ALBEDO = sfc.albedo
        self.LAMBER = sfc.lambertian
        self.RHOU = sfc.rhou
        self.RHOQ = sfc.rhoq
        self.BEMST = sfc.bemst
        self.EMUST = sfc.emust
        self.RHO_ACCURATE = sfc.rho_accurate

    def run(self):
        """Run DISORT for given optical properties, illumination and observation angles as well as incident beam
        intensities and return specific spectral snow reflectance factor:

           pure direct illumination conditions (e_dir = 1.0):
           - bidirectional reflectance factor (BRF)
           - directional-hemispherical reflectance (DHR)

           for direct + diffuse or diffuse only illumination conditions:
           - hemispherical-directional reflectance factor (HDRF)
           - bihemispherical reflectance (BHR or albedo)

        Returns: snow reflectance factor (BRF and DHR for pure direct illumination conditions (e_dir = 1.0),
                 and HDRF and BHR for the remaining), and associated wavelengths
        """

        hdrf = np.zeros(self.nbr_wvl)
        albedo = np.zeros(self.nbr_wvl)

        for ind in range(self.nbr_wvl):
            rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = disort.disort(
                self.USRANG,
                self.USRTAU,
                self.IBCND,
                self.ONLYFL,
                self.PRNT,
                self.PLANK,
                self.LAMBER,
                self.DELTAMPLUS,
                self.DO_PSEUDO_SPHERE,
                self.DTAUC[:, ind],
                self.SSALB[:, ind],
                self.PMOM[:, :, ind],
                self.TEMPER,
                self.WVNMLO,
                self.WVNMHI,
                self.UTAU,
                self.UMU0,
                self.PHI0,
                self.UMU,
                self.PHI,
                self.FBEAM[ind],
                self.FISOT[ind],
                self.ALBEDO,
                self.BTEMP,
                self.TTEMP,
                self.TEMIS,
                self.EARTH_RADIUS,
                self.H_LYR,
                self.RHOQ,
                self.RHOU,
                self.RHO_ACCURATE,
                self.BEMST,
                self.EMUST,
                self.ACCUR,
                self.HEADER,
                self.RFLDIR,
                self.RFLDN,
                self.FLUP,
                self.DFDT,
                self.UAVG,
                self.UU,
                self.ALBMED,
                self.TRNMED,
            )

            hdrf[ind] = uu[0, 0, 0]
            albedo[ind] = flup[0] / np.pi

        return hdrf, albedo, self.wvl
