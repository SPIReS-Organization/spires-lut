# spires-lut

Reflectance lookup tables for the SPIReS package family: create, read, and write
the Mie-scattering LUTs the inversion interpolates against
(dimensions: band, solar angle, LAP concentration, grain size).

This package is part of the [SPIReS family](https://spires.readthedocs.io/).

```{note}
**Status: scaffolding.** `spires-lut` is an early scaffold — the public API is
not implemented yet, so this site is a placeholder. It will grow an API
reference (autodoc) as the LUT create/read/write code lands. Track progress in
the [repository](https://github.com/SPIReS-Organization/spires-lut).
```

## Planned scope

- Read/write the Mie-scattering reflectance LUTs consumed by `spires-inversion`.
- Formalize the `spires_contract.lut` boundary
  (band, solar_angle, lap_concentration, grain_size).
