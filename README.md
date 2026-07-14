# spires-lut

Reflectance lookup tables (LUTs) for the
[SPIReS](https://github.com/SPIReS-Organization) package family: create, read, and
write the Mie-scattering reflectance LUTs the inversion interpolates.

LUTs conform to the LUT boundary defined in
[`spires-contract`](https://github.com/SPIReS-Organization/spires-contract) —
dims `(band, solar_angle, dust_concentration, grain_size)`.

> **Status:** scaffolding only. The package layout and dependency on
> `spires-contract` are in place; the implementation is to be filled in.


pyRT and DISORT fortran code is from https://github.com/mjwolff/pyRT_DISORT and https://github.com/mjwolff/pyDISORT .

The optical properties are from https://github.com/jmcook1186/biosnicar-py, and all citations they suggest should be used in any works.