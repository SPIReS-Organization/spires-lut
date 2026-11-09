# tools/

One-shot derivation scripts. Not part of the installed package.

- `fetch_srfs.py` — downloads USGS (OLI, OLI-2), ESA (Sentinel-2 MSI) and NASA MCST
  (MODIS Terra) spectral response workbooks and writes the bundled 1 nm CSVs plus
  the generated region of `spires_lut/sensors/data/bands.toml`. Needs
  `pip install spires-lut[tools]` (openpyxl). Hand-written entries (Landsat 10,
  HLS aliases) sit below the `# --- END GENERATED ---` marker and are preserved.
