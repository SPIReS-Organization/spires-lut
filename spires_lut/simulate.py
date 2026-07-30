import numpy as np
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed
import xarray as xr
from typing import List, Tuple, Optional, Union
import numpy.typing as npt

from spires_lut.common import VectorInterpolator
from spires_lut.disort_wrapper import DISORT
from spires_contract import conventions as c


def create_spires_lut(
    output_dir: Union[str, Path],
    atm_lut: Union[str, Path],
    cpu_cores: Optional[int] = 1,
    grain_shape: Optional[int] = 1,
    lap_type: Optional[str] = "dust_skiles_size3",
    lap_min: Optional[float] = 0.0,
    lap_max: Optional[float] = 4000.0,
    lap_points: Optional[int] = 10,
    altitude_min: Optional[float] = 0.0,
    altitude_max: Optional[float] = 7.0,
    altitude_points: Optional[int] = 7,
    illumination_angle_min: Optional[float] = 0.0,
    illumination_angle_max: Optional[float] = 89.0,
    illumination_angle_points: Optional[int] = 10,
    toa_sza_min: Optional[float] = 0.0,
    toa_sza_max: Optional[float] = 70.0,
    toa_sza_points: Optional[int] = 7,
    svf_min: Optional[float] = 1e-3,
    svf_max: Optional[float] = 1.0,
    svf_points: Optional[int] = 5,
) -> None:
    """
    Generate SPIReS broadband blue-sky albedo and spectral reflectance lookup tables (LUTs)
    using the DISORT snow surface model and prebuilt atmospheric lookup table.

    This assumes the product has undergone BRDF normalization (similar to most VIIRS-like products).
    And that VZA within DISORT can be treated as 0deg. Therefore, RAA also is assumed to be a default value.

    The generated blue-sky reflectance LUT is at the full resolution (10 nm , across the entire VSWIR spectrum).
    The resulting LUT from this must be then passed to the method to convolve to sensor wavelengths.

    Parameters
    ----------
    output_dir : Union[str, Path]
        Directory path where the output NetCDF LUT files will be saved.
    atm_lut : Union[str, Path]
        Path to MODTRAN-like atmospheric NetCDF LUT file.
    cpu_cores : Optional[int], optional
        Number of CPU cores to use for parallel processing (default is 1).
    grain_shape : Optional[int], optional
        Snow grain shape: 1=sphere, 2=spheroid, 3=hexagonal plate, 4=koch snowflake (default is 1).
    lap_type : Optional[str], optional
        Light-absorbing particle (LAP) lap type name (default is "dust_skiles_size3"). For now only dust is supported.
    lap_min : Optional[float], optional
        Minimum LAP concentration value in ppm (default is 0.0).
    lap_max : Optional[float], optional
        Maximum LAP concentration value in ppm (default is 4000.0).
    lap_points : Optional[int], optional
        Number of LAP concentration points (default is 10).
    altitude_min : Optional[float], optional
        Minimum altitude in km (default is 0.0).
    altitude_max : Optional[float], optional
        Maximum altitude in km (default is 7.0).
    altitude_points : Optional[int], optional
        Number of altitude grid points (default is 7).
    illumination_angle_min : Optional[float], optional
        Minimum illumination angle in degrees (default is 0.0).
    illumination_angle_max : Optional[float], optional
        Maximum illumination angle in degrees (default is 89.0).
    illumination_angle_points : Optional[int], optional
        Number of illumination angle points (default is 10).
    toa_sza_min : Optional[float], optional
        Minimum TOA solar zenith angle in degrees (default is 0.0).
    toa_sza_max : Optional[float], optional
        Maximum TOA solar zenith angle in degrees (default is 70.0).
    toa_sza_points : Optional[int], optional
        Number of TOA solar zenith angle grid points (default is 7).
    svf_min : Optional[float], optional
        Minimum sky view factor (default is 1e-3).
    svf_max : Optional[float], optional
        Maximum sky view factor (default is 1.0).
    svf_points : Optional[int], optional
        Number of sky view factor points (default is 5).

    Returns
    -------
    None
        Saves 'spires_albedo_lut.nc' and 'spires_reflectance_lut.nc' to the specified output directory.
    """

    # Create output if doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run dummy run to get wavelength and check OP data is downloaded
    _, _, _, _, wavelength = run_point()

    # Load MODTRAN-like data and create interpolator object
    m = xr.load_dataset(atm_lut)
    solar_irr = m.solar_irr.values
    wl = m.wl.values
    transm_down_dir = (
        m["transm_down_dir"]
        .transpose("AOT550", "solar_zenith", "surface_elevation_km", "wl")
        .values
    )
    transm_down_dif = (
        m["transm_down_dif"]
        .transpose("AOT550", "solar_zenith", "surface_elevation_km", "wl")
        .values
    )
    grid_modtran = [
        m["AOT550"].values,
        m["solar_zenith"].values,
        m["surface_elevation_km"].values,
    ]
    wl_mask = (wl >= 280.0) & (wl <= 4000.0)
    wl, solar_irr = wl[wl_mask], solar_irr[wl_mask]
    transm_down_dir, transm_down_dif = (
        transm_down_dir[..., wl_mask],
        transm_down_dif[..., wl_mask],
    )
    g_t_down_dir = VectorInterpolator(
        grid_input=grid_modtran, data_input=transm_down_dir, version="mlg"
    )
    g_t_down_dif = VectorInterpolator(
        grid_input=grid_modtran, data_input=transm_down_dif, version="mlg"
    )

    # Grid resolution
    altitude = np.linspace(
        altitude_min, altitude_max + 0.001, altitude_points, dtype=np.float32
    )
    illumination_angle = np.linspace(
        illumination_angle_min,
        illumination_angle_max + 0.001,
        illumination_angle_points,
        dtype=np.float32,
    )
    sqrt_grain_radius = np.sqrt(np.arange(30.0, 1500.001, 10.0, dtype=np.float32))
    dust_conc = np.linspace(lap_min, lap_max + 0.001, lap_points, dtype=np.float32)
    toa_sza = np.linspace(
        toa_sza_min, toa_sza_max + 0.001, toa_sza_points, dtype=np.float32
    )
    svf = np.linspace(svf_min, svf_max + 0.001, svf_points, dtype=np.float32)

    # Assumed fixed, clear atmosphere
    AOT = 0.05

    # Set up shape of the luts
    shape_alb = (
        len(toa_sza),
        len(illumination_angle),
        len(dust_conc),
        len(sqrt_grain_radius),
        len(svf),
        len(altitude),
    )

    shape_refl = (
        len(wavelength),
        len(illumination_angle),
        len(dust_conc),
        len(sqrt_grain_radius),
    )

    # Define output filenames
    albedo_file = output_path / "spires_albedo_lut.nc"
    reflectance_file = output_path / "spires_reflectance_lut.nc"

    # Create the outputs
    xr.Dataset(
        {
            c.ALBEDO_LUT_VARIABLE: (
                [
                    "solar_zenith",
                    "illumination_angle",
                    "lap_concentration",
                    "sqrt_grain_radius",
                    "skyview",
                    "altitude",
                ],
                np.zeros(shape_alb, dtype=np.float32),
            )
        },
        coords={
            "solar_zenith": xr.DataArray(
                toa_sza,
                dims=("solar_zenith",),
                attrs={"units": c.LUT_AXIS_UNITS["solar_zenith"]},
            ),
            "illumination_angle": xr.DataArray(
                illumination_angle,
                dims=("illumination_angle",),
                attrs={"units": c.LUT_AXIS_UNITS["illumination_angle"]},
            ),
            "lap_concentration": xr.DataArray(
                dust_conc,
                dims=("lap_concentration",),
                attrs={
                    "units": c.LUT_AXIS_UNITS["lap_concentration"],
                    "lap_type": "dust",
                },
            ),
            "sqrt_grain_radius": xr.DataArray(
                sqrt_grain_radius,
                dims=("sqrt_grain_radius",),
                attrs={"units": c.LUT_AXIS_UNITS["sqrt_grain_radius"]},
            ),
            "altitude": xr.DataArray(
                altitude,
                dims=("altitude",),
                attrs={"units": c.LUT_AXIS_UNITS["altitude"]},
            ),
            "skyview": xr.DataArray(
                svf, dims=("skyview",), attrs={"units": c.LUT_AXIS_UNITS["skyview"]}
            ),
        },
    ).to_netcdf(albedo_file)

    xr.Dataset(
        {
            c.REFLECTANCE_LUT_VARIABLE: (
                ["band", "solar_angle", "lap_concentration", "sqrt_grain_radius"],
                np.zeros(shape_refl, dtype=np.float32),
            )
        },
        coords={
            "band": xr.DataArray(wavelength, dims=("band",), attrs={"units": "nm"}),
            "solar_angle": xr.DataArray(
                illumination_angle,
                dims=("solar_angle",),
                attrs={"units": c.LUT_AXIS_UNITS["solar_angle"]},
            ),
            "lap_concentration": xr.DataArray(
                dust_conc,
                dims=("lap_concentration",),
                attrs={
                    "units": c.LUT_AXIS_UNITS["lap_concentration"],
                    "lap_type": "dust",
                },
            ),
            "sqrt_grain_radius": xr.DataArray(
                sqrt_grain_radius,
                dims=("sqrt_grain_radius",),
                attrs={"units": c.LUT_AXIS_UNITS["sqrt_grain_radius"]},
            ),
        },
    ).to_netcdf(reflectance_file)

    # Group by params where rfl vary (DISORT sims)
    disort_surface_grid = list(
        product(
            enumerate(illumination_angle),
            enumerate(sqrt_grain_radius),
            enumerate(dust_conc),
        )
    )

    # Group all parameters from the albedo table
    full_grid = list(
        product(
            enumerate(altitude),
            enumerate(sqrt_grain_radius),
            enumerate(dust_conc),
            enumerate(toa_sza),
            enumerate(illumination_angle),
            enumerate(svf),
        )
    )
    total_sims = len(full_grid)

    # then we batch them, so that disort rfl can be reused in the portions of the lut
    batch_size = 50
    surface_batches = [
        disort_surface_grid[i : i + batch_size]
        for i in range(0, len(disort_surface_grid), batch_size)
    ]

    print(f"Computing {total_sims} points across {cpu_cores} cores...")

    def process_surface_batch(batch):
        batch_output = []
        for (ill_idx, ill_angle_val), (grain_idx, sq_r), (dust_idx, d) in batch:

            # Calls DISORT
            # NOTE here we are using the rho_dir_dif and rho_dif_dif - this was a choice..
            # This is essentially the black and white sky albedos.
            # We could alternatively use the dir_dir and dif_dir for more directional-like quantities.
            # Both have pros/cons. And depends on how much you trust the BRDF model vs surface roughness assumptions.

            # We could also choose to save all 4 of these raw rfl defnitions, but at least for the moment for SPIReS it
            # doesn't seem very useful
            _, _, rho_dir_dif, rho_dif_dif, _ = run_point(
                sza=ill_angle_val,
                vza=0.0,  # VZA assumed to be zero
                raa=0.0,  # RAA assumed to not matter
                grain_radius=[np.round(sq_r**2)],
                lap_concentration=[d],
                algae_concentration=[0.0],  # no algae
                lwc=[0.0],  # no liquid water
                lap_type=lap_type.lower(),
                grain_shape=grain_shape,
                snow_density=[300.0],  # kg /m3
                dz=[100.0],  # 100 m -> optically deep
                n_layers=1,  # 1 = single layer snowpack
            )

            # Evaluate all atmospheric variations (altitude, toa_sza, svf) for this point
            for alt_idx, alt in enumerate(altitude):
                for sza_idx, sza_deg in enumerate(toa_sza):
                    for svf_idx, sv in enumerate(svf):

                        # convert to cosine for radiance calcs
                        cos_theta = np.cos(np.radians(sza_deg))
                        cosi = np.cos(np.radians(ill_angle_val))

                        # Compute downward flux
                        lut_array = np.array([AOT, sza_deg, alt])
                        L_down_dir = (
                            g_t_down_dir(lut_array) * solar_irr * cos_theta / np.pi
                        )
                        L_down_dif = (
                            g_t_down_dif(lut_array) * solar_irr * cos_theta / np.pi
                        )

                        L_dif = L_down_dif * sv
                        L_dir = L_down_dir / cos_theta * cosi

                        L_total = np.interp(wavelength, wl, L_dir + L_dif)

                        # Diffuse fraction (k)
                        # interpolating over noisy water features
                        k = L_dif / (L_dif + L_dir + 1e-12)
                        k = np.interp(wavelength, wl, k)

                        k_bad_mask = ((wavelength >= 1330) & (wavelength <= 1450)) | (
                            (wavelength >= 1800) & (wavelength <= 1980)
                        )
                        if np.any(k_bad_mask):
                            k_valid_indices = np.where(~k_bad_mask)[0]
                            k_bad_indices = np.where(k_bad_mask)[0]
                            if len(k_valid_indices) > 0 and len(k_bad_indices) > 0:
                                k[k_bad_indices] = np.interp(
                                    wavelength[k_bad_indices],
                                    wavelength[k_valid_indices],
                                    k[k_valid_indices],
                                )
                        # assuming diffuse fraction at longer wavelengths to be near zero
                        k[wavelength > 2500.0] = 0.0

                        # Compute bluesky albedo
                        rfl_blue = (1 - k) * rho_dir_dif + k * rho_dif_dif
                        albedo = np.trapezoid(rfl_blue * L_total, dx=1) / np.trapezoid(
                            L_total + 1e-12, dx=1
                        )

                        rfl = (
                            rfl_blue
                            if (alt_idx == 0 and sza_idx == 0 and svf_idx == 0)
                            else None
                        )
                        batch_output.append(
                            (
                                (
                                    alt_idx,
                                    grain_idx,
                                    dust_idx,
                                    sza_idx,
                                    ill_idx,
                                    svf_idx,
                                ),
                                albedo,
                                rfl,
                            )
                        )

        return batch_output

    with (
        xr.open_dataset(albedo_file, mode="a") as ds_alb,
        xr.open_dataset(reflectance_file, mode="a") as ds_rfl,
    ):
        results = Parallel(n_jobs=cpu_cores)(
            delayed(process_surface_batch)(batch) for batch in surface_batches
        )

        for batch_res in results:
            for indices, albedo, rfl in batch_res:
                alt_idx, grain_idx, dust_idx, sza_idx, ill_idx, svf_idx = indices
                ds_alb["albedo"][
                    sza_idx, ill_idx, dust_idx, grain_idx, svf_idx, alt_idx
                ] = albedo.astype(np.float32)
                if rfl is not None:
                    ds_rfl["reflectance"][:, ill_idx, dust_idx, grain_idx] = rfl.astype(
                        np.float32
                    )

        ds_alb.to_netcdf(albedo_file, mode="a")
        ds_rfl.to_netcdf(reflectance_file, mode="a")

    return


def run_point(
    sza: float = 45.0,
    vza: float = 15.0,
    raa: float = 0.0,
    n_layers: int = 3,
    grain_shape: int = 1,
    grain_radius: List[float] = [300.0, 300.0, 300.0],
    lap_concentration: List[float] = [1000.0, 0.0, 0.0],
    algae_concentration: List[float] = [1e6, 0.0, 0.0],
    lwc: List[float] = [10.0, 10.0, 10.0],
    snow_density: List[float] = [150.0, 300.0, 300.0],
    dz: List[float] = [0.01, 0.1, 5.0],
    lap_type: str = "dust_skiles_size3",
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """

    Entry point for the DISORT model for testing and/or for LUT building.

    e.g., see, create_spires_lut() - although there are many ways you can choose to grid/interpret this model.

    sza:             solar zenith angle (degrees)
    vza:             view zenith angle (degrees)
    raa:             relative azimuth angle (degrees)
    nbr_lyr:         number of horizontal snow surface layers
    rho_snw:         density of each layer (unit = kg m-3), list must have len(nbr_lyr)
    dz:              thickness of each layer (unit = m), list must have len(nbr_lyr)
    lap_type:        file name of lap optical properties
    lap_concentration:     lap in micro g per g of snow
    algae_concentration: snow algae concentration of each layer in units of cells/mL, list must have len(nbr_lyr)
    grain_radius:         snow grain radius of each layer, list must have len(nbr_lyr)
    lwc:         liquid water fraction
    grain_shape:     1=sphere; 2=spheroid; 3=hexagonal plate; 4=koch snowflake
    """

    # Create a separation between algae and lap to allow for two types if needed
    if "dust" not in lap_type:
        raise ValueError(
            "For now, only support we only support dust for lap_type. Check spires-contract."
        )

    # Set base config (direct illum first)
    cfg = {
        "sza": sza,
        "vza": vza,
        "vaa": raa,
        "saa": 0.0,
        "rds_snw": grain_radius,
        "mss_cnc_lap": lap_concentration,
        "mss_cnc_snw_alg": algae_concentration,
        "lw_frac": lwc,
        "lap_type": lap_type.lower(),
        "grain_shape": grain_shape,
        "rho_snw": snow_density,
        "dz": dz,
        "nbr_lyr": n_layers,
        "e_dir": 1.0,
        "e_dif": 0.0,
    }

    # Run black sky
    model = DISORT(**cfg)
    rho_dir_dir, rho_dir_dif, wvl = model.run()

    # Update config and run white sky
    cfg["e_dir"] = 0.0
    cfg["e_dif"] = 1.0
    model = DISORT(**cfg)
    rho_dif_dir, rho_dif_dif, wvl = model.run()

    return rho_dir_dir, rho_dif_dir, rho_dir_dif, rho_dif_dif, wvl * 1000.0
