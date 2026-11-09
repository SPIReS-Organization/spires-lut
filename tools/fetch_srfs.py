#!/usr/bin/env python
"""Download agency spectral response workbooks and write spires_lut/sensors/data/.

Re-runnable derivation of the bundled SRF CSVs and of the generated region of
bands.toml. Requires ``openpyxl`` (``pip install spires-lut[tools]``). Network
access to usgs.gov, copernicus.eu and gsfc.nasa.gov is needed; usgs.gov rejects
requests without a browser User-Agent.

Output CSVs are long format, ``band, wavelength_nm, response``, on the agencies'
native 1 nm grid, keeping only rows with response > 1e-4, each prefixed with a
``#`` provenance header. Band centres written to bands.toml are the
response-weighted mean wavelength of each curve; FWHM is the width at half of the
curve's peak.

Usage: python tools/fetch_srfs.py [--out spires_lut/sensors/data] [--keep-downloads DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (spires-lut fetch_srfs)"}

SOURCES = {
    "oli": {
        "url": "https://landsat.usgs.gov/landsat/spectral_viewer/bands/L8_OLI_RSR.xlsx",
        "doc": "USGS Landsat 8 OLI relative spectral response (Spectral Characteristics Viewer)",
        "platform": "Landsat 8",
        "sheets": {  # sheet -> (band code, long name)
            "CoastalAerosol": ("B1", "coastal aerosol"), "Blue": ("B2", "blue"), "Green": ("B3", "green"),
            "Red": ("B4", "red"), "NIR": ("B5", "near infrared"), "SWIR1": ("B6", "shortwave infrared 1"),
            "SWIR2": ("B7", "shortwave infrared 2"), "Pan": ("B8", "panchromatic"), "Cirrus": ("B9", "cirrus"),
        },
        "wl_col": 0, "resp_col": 1,
    },
    "oli2": {
        "url": "https://landsat.usgs.gov/landsat/spectral_viewer/bands/L9_OLI2_RSR.xlsx",
        "doc": "USGS Landsat 9 OLI-2 relative spectral response (GLAMR-derived; Spectral Characteristics Viewer)",
        "platform": "Landsat 9",
        "sheets": {
            "CoastalAerosol": ("B1", "coastal aerosol"), "Blue": ("B2", "blue"), "Green": ("B3", "green"),
            "Red": ("B4", "red"), "NIR": ("B5", "near infrared"), "SWIR1": ("B6", "shortwave infrared 1"),
            "SWIR2": ("B7", "shortwave infrared 2"), "Pan": ("B8", "panchromatic"), "Cirrus": ("B9", "cirrus"),
        },
        "wl_col": 0, "resp_col": 1,
    },
}

S2_URL = (
    "https://sentiwiki.copernicus.eu/__attachments/"
    "a_ece6183b6698587e7ecd9804974bece3d82e39e151aa5aeca9d8fd95e1f1558c/"
    "COPE-GSEG-EOPG-TN-15-0007%20-%20Sentinel-2%20Spectral%20Response%20Functions%202024%20-%204.0.xlsx"
    "?cb=05a866b9653a4d2bc442df394ea6cc7a"
)
S2_DOC = "ESA COPE-GSEG-EOPG-TN-15-0007 Sentinel-2 Spectral Response Functions, v4.0 (2024-06-04)"
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
S2_LONG = {
    "B1": "coastal aerosol", "B2": "blue", "B3": "green", "B4": "red", "B5": "red edge 1",
    "B6": "red edge 2", "B7": "red edge 3", "B8": "near infrared", "B8A": "narrow near infrared",
    "B9": "water vapour", "B10": "cirrus", "B11": "shortwave infrared 1", "B12": "shortwave infrared 2",
}
# spires-io / HLS zero-padded codes
S2_CODE = {b: (b if b == "B8A" else f"B{int(b[1:]):02d}") for b in S2_BANDS}

MODIS_URL = "https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/Terra_RSR_in-band.xlsx"
MODIS_DOC = "NASA MCST Terra MODIS (PFM) in-band relative spectral response"
MODIS_LONG = {1: "red", 2: "near infrared", 3: "blue", 4: "green", 5: "1240 nm", 6: "1640 nm", 7: "2130 nm"}


def fetch(url: str, keep: Path | None) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if keep is not None:
        keep.mkdir(parents=True, exist_ok=True)
        (keep / url.split("/")[-1].split("?")[0]).write_bytes(data)
    return data


def normalise(wl: np.ndarray, resp: np.ndarray) -> pd.DataFrame:
    ok = np.isfinite(wl) & np.isfinite(resp)
    wl, resp = wl[ok], resp[ok]
    order = np.argsort(wl)
    wl, resp = wl[order], resp[order]
    resp = resp / resp.max()
    keep = resp > 1e-4
    return pd.DataFrame({"wavelength_nm": wl[keep], "response": resp[keep]})


def centre_fwhm(df: pd.DataFrame) -> tuple[float, float]:
    wl, r = df["wavelength_nm"].to_numpy(), df["response"].to_numpy()
    centre = float(np.sum(wl * r) / np.sum(r))
    above = wl[r >= 0.5 * r.max()]
    return centre, float(above.max() - above.min())


def write_csv(path: Path, frames: dict[str, pd.DataFrame], url: str, doc: str, today: str) -> None:
    header = f"# source: {url}\n# document: {doc}\n# retrieved: {today}\n# columns: band, wavelength_nm, response (peak-normalised, 1 nm)\n"
    out = pd.concat([f.assign(band=b)[["band", "wavelength_nm", "response"]] for b, f in frames.items()])
    with path.open("w", newline="") as fh:
        fh.write(header)
        out.to_csv(fh, index=False, float_format="%.6g")


def toml_sensor(name, platform, csv, url, doc, today, rows) -> str:
    lines = [f"[sensors.{name}]", f'platform = "{platform}"',
             f'srf_source = "{url} ({doc}; retrieved {today})"', "bands = ["]
    for code, long_name, centre, fwhm, col in rows:
        lines.append(
            f'  {{ name = "{code}", long_name = "{long_name}", center_nm = {centre:.1f}, fwhm_nm = {fwhm:.1f}, '
            f'response = "srf", srf_file = "{csv}", srf_band = "{col}" }},'
        )
    lines.append("]\n")
    return "\n".join(lines)


def do_landsat(key, cfg, out, keep, today) -> str:
    import openpyxl  # noqa: F401  (pandas engine)
    wb = pd.read_excel(io.BytesIO(fetch(cfg["url"], keep)), sheet_name=None, header=0)
    frames, rows = {}, []
    for sheet, (code, long_name) in cfg["sheets"].items():
        df = wb[sheet]
        f = normalise(df.iloc[:, cfg["wl_col"]].to_numpy(float), df.iloc[:, cfg["resp_col"]].to_numpy(float))
        frames[code] = f
        c, w = centre_fwhm(f)
        rows.append((code, long_name, c, w, code))
    write_csv(out / f"{key}.csv", frames, cfg["url"], cfg["doc"], today)
    return toml_sensor(key.upper(), cfg["platform"], f"{key}.csv", cfg["url"], cfg["doc"], today, rows)


def do_s2(out, keep, today) -> str:
    raw = fetch(S2_URL, keep)
    chunks = []
    for sat, csv, name in (("S2A", "msi_s2a.csv", "MSI"), ("S2B", "msi_s2b.csv", "MSI-S2B")):
        df = pd.read_excel(io.BytesIO(raw), sheet_name=f"Spectral Responses ({sat})")
        wl = df["SR_WL"].to_numpy(float)
        frames, rows = {}, []
        for b in S2_BANDS:
            f = normalise(wl, df[f"{sat}_SR_AV_{b}"].to_numpy(float))
            code = S2_CODE[b]
            frames[code] = f
            c, w = centre_fwhm(f)
            rows.append((code, S2_LONG[b], c, w, code))
        write_csv(out / csv, frames, S2_URL, S2_DOC, today)
        chunks.append(toml_sensor(name, f"Sentinel-2{sat[-1]}", csv, S2_URL, S2_DOC, today, rows))
    return "\n".join(chunks)


def do_modis(out, keep, today) -> str:
    raw = fetch(MODIS_URL, keep)
    frames, rows = {}, []
    for b in range(1, 8):
        df = pd.read_excel(io.BytesIO(raw), sheet_name=f"band_{b}")
        # several detector channels per band: average the RSR per wavelength
        g = df.groupby("Wavelength")["RSR"].mean()
        f = normalise(g.index.to_numpy(float), g.to_numpy(float))
        code = str(b)
        frames[code] = f
        c, w = centre_fwhm(f)
        rows.append((code, MODIS_LONG[b], c, w, code))
    write_csv(out / "modis_terra.csv", frames, MODIS_URL, MODIS_DOC, today)
    return toml_sensor("MODIS", "Terra", "modis_terra.csv", MODIS_URL, MODIS_DOC, today, rows)


BEGIN, END = "# --- BEGIN GENERATED (tools/fetch_srfs.py) ---", "# --- END GENERATED ---"


def splice_toml(path: Path, generated: str) -> None:
    text = path.read_text() if path.exists() else f"{BEGIN}\n{END}\n"
    if BEGIN not in text or END not in text:
        sys.exit(f"{path}: generated-region markers missing; refusing to overwrite")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    path.write_text(f"{head}{BEGIN}\n{generated}\n{END}{tail}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("spires_lut/sensors/data"))
    ap.add_argument("--keep-downloads", type=Path, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    parts = [do_landsat(k, c, a.out, a.keep_downloads, today) for k, c in SOURCES.items()]
    parts.append(do_s2(a.out, a.keep_downloads, today))
    parts.append(do_modis(a.out, a.keep_downloads, today))
    splice_toml(a.out / "bands.toml", "\n".join(parts))
    print(f"wrote CSVs and generated TOML region to {a.out}")


if __name__ == "__main__":
    main()
