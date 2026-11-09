# spires-lut

Reflectance lookup tables (LUTs) for the
[SPIReS](https://github.com/SPIReS-Organization) package family: create, read, and
write the Mie-scattering reflectance LUTs the inversion interpolates.

LUTs conform to the LUT boundary defined in
[`spires-contract`](https://github.com/SPIReS-Organization/spires-contract) —
dims `(band, solar_angle, lap_concentration, sqrt_grain_radius)`.


## Sensor LUTs

`create_spires_lut` writes a hyperspectral (10 nm, 255–4995 nm) reflectance LUT.
`spires_lut.lut.build_sensor_lut` band-averages it onto a sensor's spectral response
functions and validates the result against `spires-contract`:

```python
from spires_lut.io import read_lut, write_lut
from spires_lut.lut import build_sensor_lut, sensor_lut_filename
from spires_lut.sensors import get_sensor, list_sensors

hs = read_lut("spires_reflectance_lut.nc")
oli = build_sensor_lut(hs, get_sensor("OLI"))
write_lut(oli, sensor_lut_filename("OLI", oli.attrs["lap_type"]))
print(list_sensors())
# ['HLSL30', 'HLSS30', 'HLSS30-msi', 'LANDSAT10', 'MODIS', 'MSI', 'MSI-S2B', 'OLI', 'OLI2']
```

The `band` coordinate of a sensor LUT holds the band names `spires-io` emits
(`B1..B9` for OLI, `B01..B12`/`B8A` for MSI and HLS, `1..7` for MODIS). Measured SRFs
for OLI, OLI-2, MSI and MODIS are bundled under `spires_lut/sensors/data/` (regenerate
with `tools/fetch_srfs.py`); Landsat 10 uses Gaussian bands from the published
centre/bandwidth table until SRFs exist. `examples/make_sensor_luts.py` runs the whole
chain. Band metadata and provenance: `spires_lut/sensors/data/bands.toml`.


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