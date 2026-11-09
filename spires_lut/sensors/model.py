"""Sensor and band descriptions.

A ``Band`` says *how* its spectral response is defined (``srf``: a measured curve
bundled as package data; ``tophat``: flat between two edges; ``gaussian``: centre and
FWHM). It knows nothing about snow or about the wavelength grid it will be evaluated
on; that is ``convolve.py``'s job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

RESPONSE_KINDS = ("srf", "tophat", "gaussian")


@dataclass(frozen=True)
class Band:
    name: str
    center_nm: float
    fwhm_nm: float
    response: str
    lower_nm: Optional[float] = None
    upper_nm: Optional[float] = None
    srf_file: Optional[str] = None
    srf_band: Optional[str] = None
    long_name: str = ""

    def __post_init__(self):
        if self.response not in RESPONSE_KINDS:
            raise ValueError(
                f"band {self.name!r}: response must be one of {RESPONSE_KINDS}, got {self.response!r}"
            )
        if self.response == "tophat" and (self.lower_nm is None or self.upper_nm is None):
            raise ValueError(f"band {self.name!r}: tophat response needs lower_nm and upper_nm")
        if self.response == "srf" and (self.srf_file is None or self.srf_band is None):
            raise ValueError(f"band {self.name!r}: srf response needs srf_file and srf_band")
        if self.fwhm_nm <= 0:
            raise ValueError(f"band {self.name!r}: fwhm_nm must be positive")


@dataclass(frozen=True)
class Sensor:
    name: str
    platform: str
    bands: Tuple[Band, ...]
    srf_source: str

    def __post_init__(self):
        names = [b.name for b in self.bands]
        if len(set(names)) != len(names):
            raise ValueError(f"sensor {self.name!r}: duplicate band names in {names}")

    @property
    def band_names(self) -> Tuple[str, ...]:
        return tuple(b.name for b in self.bands)
