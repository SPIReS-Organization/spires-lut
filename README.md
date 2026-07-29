# spires-lut

Reflectance lookup tables (LUTs) for the
[SPIReS](https://github.com/SPIReS-Organization) package family: create, read, and
write the Mie-scattering reflectance LUTs the inversion interpolates.

LUTs conform to the LUT boundary defined in
[`spires-contract`](https://github.com/SPIReS-Organization/spires-contract) —
dims `(band, solar_angle, dust_concentration, grain_size)`.


## DISORT

While the main objective of spires-lut is to generate the data for SPIReS inversion. One may also use it to interface with the DISORT code for a single spectra (code example below). Please note the Fortran code (as well as pyrt) do not originate from this repo and instead are compiled here for convenience to the user.

```python 

from spires_lut.disort_wrapper import DISORT

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
rfl, albedo, wvl = DISORT(**cfg).run()

```