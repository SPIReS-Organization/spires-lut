import numpy as np

from spires_lut.optics import (  # noqa: F401  (get_optical_property_data, generate_filter, wvl_resample re-exported for backward compatibility)
    OP_DIR,
    WVL_THRESHOLD,
    generate_filter,
    get_optical_property_data,
    snow_lap_mixture,
    wvl_resample,
)
from spires_lut.pyrt.aerosol import HenyeyGreenstein, TabularLegendreCoefficients
from spires_lut.pyrt.controller import ComputationalParameters, ModelBehavior
from spires_lut.pyrt.eos import Hydrostatic
from spires_lut.pyrt.observation import Spectral
from spires_lut.pyrt.output import OutputArrays, OutputBehavior, UserLevel
from spires_lut.pyrt.radiation import IncidentFlux, ThermalEmission
from spires_lut.pyrt.surface import Surface

import disort


class DISORT(object):
    """Class that holds all necessary attributes and functions to apply the multistream radiative transfer solver
    DISORT to given Mie calculated snow optical properties. In particular, this class supports the inclusion of liquid
    water coatings as well as lap and snow algae. Adapted code from Niklas Bohn.
    """

    def __init__(
        self,
        sza: float,
        saa: float,
        vza: np.array,
        vaa: np.array,
        e_dir: np.ndarray,
        e_dif: np.ndarray,
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

        # set base directory for auxiliary files (kept as attributes for callers that read them)
        self.dir_base = OP_DIR
        self.wvl_threshold = WVL_THRESHOLD

        # snow_lap_mixture sizes its layers from rds_snw; a mismatch would reach disort as a wrong-length DTAUC
        if len(rds_snw) != nbr_lyr:
            raise ValueError(f"nbr_lyr={nbr_lyr} but len(rds_snw)={len(rds_snw)}")

        mix = snow_lap_mixture(
            rds_snw=rds_snw,
            rho_snw=rho_snw,
            dz=dz,
            lap_type=lap_type,
            mss_cnc_lap=mss_cnc_lap,
            mss_cnc_snw_alg=mss_cnc_snw_alg,
            lw_frac=lw_frac,
            grain_shape=grain_shape,
        )

        self.wvl_op = mix.wavelength_um
        self.fwhm = np.full_like(self.wvl_op, fill_value=10.0) / 1000.0

        # convert wavelengths to wavenumbers
        self.wvl = self.wvl_op
        self.nbr_wvl = len(self.wvl)
        short_wvl = self.wvl - self.fwhm / 2
        long_wvl = self.wvl + self.fwhm / 2

        spectral = Spectral(short_wavelength=short_wvl, long_wavelength=long_wvl)

        self.WVNMHI = spectral.high_wavenumber
        self.WVNMLO = spectral.low_wavenumber

        self.SSA_snw, self.MAC_snw, self.g_snw = mix.ssa_snw, mix.mac_snw, mix.g_snw
        self.MSSlap = mix.MSSlap
        self.SSA_lap, self.MAC_lap, self.g_lap = mix.ssa_lap, mix.mac_lap, mix.g_lap
        self.L_snw, self.tau_snw = mix.L_snw, mix.tau_snw
        self.L_lap, self.tau_lap = mix.L_lap, mix.tau_lap
        self.tau, self.SSA, self.g = mix.tau, mix.ssa, mix.g

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
            wavelength_grid=self.wvl,
            particle_size_profile=np.array(rho_snw),
            wavelengths=self.wvl,
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

        e_dir = e_dir * np.ones_like(self.wvl)
        e_dif = e_dif * np.ones_like(self.wvl)

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
