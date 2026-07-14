import numpy as np
import pooch
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed
import zarr
from typing import List, Tuple, Optional, Union
import numpy.typing as npt

from spires_lut.spires_disort import DISORT


def get_biosnicar_data() -> None:
    # Download biosnicar data if not already present
    # TODO we could also create our own OPs and host them at some point in the future
    pooch.create(
        path=Path.home() / ".spires" / "data",
        base_url="https://github.com/jmcook1186/biosnicar-py/archive/refs/tags/",
        registry={"3.0.zip": "sha256:1a25a07a57d622614635ff650ccf53a6a81b1534dc53b17ba1acd3a57609838e"},
        env="LUT_DATA_DIR").fetch("3.0.zip", processor=pooch.Unzip())
    return


def generate_lut(
    wavelength_file: Union[str, Path],
    output_dir: Union[str, Path],
    cpu_cores: int = 1,
    liquid_water: bool = False,
    black_sky: bool = True,
    snow_algae: bool = False,
    grain_shape: int = 1,
    impurity_type: str = "dust_skiles_size3",
    impurity_max_value: Optional[float] = None,
    output_quantity: List[str] = ["albedo"],
    relative_azimuth: bool = False,
) -> None:
    """
    This is sort of a place holder for now. I can imagine the main config could hold these kind of options as well?


    % Snow grain shape option
    % 1=sphere; 2=spheroid; 3=hexagonal plate; 4=koch snowflake

    The amount of options are limited here

    """

    # This allows us to sample more of the space depending on what the lap type is
    if impurity_max_value is None:
        if "dust" in impurity_type.lower():
            max_lap_value = 4000.0
        elif "ash" in impurity_type.lower():
            max_lap_value = 7000.0
        else:
            max_lap_value = 50.0
    else:
        max_lap_value = impurity_max_value

    get_biosnicar_data()

    # Load input wavelength file from user
    # NOTE this assumes nanometers
    data = np.loadtxt(wavelength_file, skiprows=0)
    wvl = data[:, 0]

    # Setup LUT grid - TODO in future this can be a config
    dim_map = {
        "solar_zenith": np.arange(0, 89.9001, 5.0),
        "observer_zenith": np.arange(0, 60.0001, 20.0),
        "relative_azimuth": (
            np.arange(0, 180.0001, 13) if relative_azimuth else np.array([0.0])
        ),
        "dust_concentration": np.linspace(0, max_lap_value, 5),
        "grain_size": np.arange(30, 1500.0001, 15),
    }
    if snow_algae:
        dim_map["algae"] = np.arange(0, 6e-5 + 0.01, 1e-5)
    if liquid_water:
        dim_map["lwc"] = np.arange(0, 40.0001, 10)

    grid = list(product(*dim_map.values()))
    dim_names = list(dim_map.keys())
    shape = tuple(len(g) for g in dim_map.values()) + (len(wvl),)

    # Setup zarr store
    store_path = Path(output_dir) / "spires_lut.zarr"
    root = zarr.open_group(store=store_path, mode="w")

    # Save dimension coords in the zarr group
    for name, g in dim_map.items():
        root.create_array(name, data=g)
    root.create_array("wvl", data=wvl)

    # Pre-allocate for each output (potentialy can have albedo + rfl)
    for qty in output_quantity:
        root.create_array(
            qty, shape=shape, chunks=(*([1] * len(dim_names)), len(wvl)), dtype="f4"
        )

    # Chunking can be tweaked/tested here
    chunk_size = 1000
    total_sims = len(grid)

    dim_idx_map = {name: idx for idx, name in enumerate(dim_names)}

    for i in range(0, total_sims, chunk_size):
        chunk = grid[i : i + chunk_size]
        tasks = []
        for vals in chunk:
            task = {
                "sza": vals[dim_idx_map["solar_zenith"]],
                "vza": vals[dim_idx_map["observer_zenith"]],
                "vaa": vals[dim_idx_map["relative_azimuth"]],
                "saa": 0.0,
                "rds_snw": [vals[dim_idx_map["grain_size"]]],
                "mss_cnc_lap": [vals[dim_idx_map["dust_concentration"]]],
                "mss_cnc_snw_alg": ([vals[dim_idx_map["algae"]]] if snow_algae else [0.0]),
                "lw_frac": [vals[dim_idx_map["lwc"]]] if liquid_water else [0.0],
                "lap_type": impurity_type.lower(),
                "grain_shape": grain_shape,
                "rho_snw": [300.0],
                "dz": [100.0],
                "wvl": wvl,
                "fwhm": data[:, 1],
                "nbr_lyr": 1,
                "e_dir": np.ones(len(wvl)) * (1.0 if black_sky else 0.0),
                "e_dif": np.ones(len(wvl)) * (0.0 if black_sky else 1.0),
            }
            tasks.append(task)

        # Run Parallel
        results = Parallel(n_jobs=cpu_cores)(
            delayed(lambda p: DISORT(**p).run())(t) for t in tasks
        )

        # Flush to Zarr
        for vals, res in zip(chunk, results):
            idx = tuple(
                np.searchsorted(dim_map[dim_names[k]], vals[k])
                for k in range(len(dim_names))
            )
            for q_idx, qty in enumerate(output_quantity):
                root[qty][idx] = res[q_idx]

        # Display LUT progress to terminal
        completed = min(i + chunk_size, total_sims)
        percent = (completed / total_sims) * 100
        print(
            f"Simulation progress: {completed}/{total_sims} ({percent:.1f}%)",
            flush=True,
        )

    return


def single_spectra(
    wavelength_file: Union[str, Path],
    sza: float = 45.0,
    vza: float = 0.0,
    raa: float = 0.0,
    grain_size: List[float] = [800.0, 800.0, 800.0],
    impurity_concentration: List[float] = [3000.0, 0.0, 0.0],
    algae_concentration: List[float] = [0.0, 0.0, 0.0],
    lwc: List[float] = [0.0, 0.0, 0.0],
    n_layers: int = 3,
    snow_density: List[float] = [150.0, 300.0, 300.0],
    dz: List[float] = [0.01, 0.1, 5.0],
    black_sky: bool = True,
    grain_shape: int = 1,
    impurity_type: str = "dust_skiles_size3",
) -> Tuple[Optional[npt.NDArray], Optional[npt.NDArray], npt.NDArray]:
    """
    Single run

    % Snow grain shape option
    % 1=sphere; 2=spheroid; 3=hexagonal plate; 4=koch snowflake

    Right now this is sort of seperate from the SPIReS generate lut thing.. but can be brought together.

    """

    # Check to make sure it is still there
    get_biosnicar_data()

    # Load input wavelength file
    data = np.loadtxt(wavelength_file, skiprows=0)
    wvl = data[:, 0]
    fwhm = data[:, 1]

    # Build configuration dictionary
    cfg = {
        "sza": sza,
        "vza": vza,
        "vaa": raa,
        "saa": 0.0,
        "rds_snw": grain_size,
        "mss_cnc_lap": impurity_concentration,
        "mss_cnc_snw_alg": algae_concentration,
        "lw_frac": lwc,
        "lap_type": impurity_type.lower(),
        "grain_shape": grain_shape,
        "rho_snw": snow_density,
        "dz": dz,
        "wvl": wvl,
        "fwhm": fwhm,
        "nbr_lyr": n_layers,
        "e_dir": np.ones(len(wvl)) * (1.0 if black_sky else 0.0),
        "e_dif": np.ones(len(wvl)) * (0.0 if black_sky else 1.0),
    }

    # Run simulation
    model = DISORT(**cfg)
    rfl, albedo, _ = model.run()

    return rfl, albedo, wvl

