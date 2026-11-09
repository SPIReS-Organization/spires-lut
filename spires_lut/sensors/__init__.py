"""Sensor registry.

``get_sensor(name)`` returns a frozen :class:`Sensor` built from
``spires_lut/sensors/data/bands.toml``. Entries either list ``bands`` directly or alias
another entry via ``curves_from`` (optionally with a ``rename`` map that also selects
and orders the bands kept). Every band is checked at load time for overlap with the
DISORT wavelength grid, so a mis-specified band fails here and never reaches a LUT.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from importlib import resources
from typing import Dict, List

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from spires_lut.sensors.convolve import DISORT_GRID_NM, sensor_weights
from spires_lut.sensors.model import Band, Sensor

__all__ = ["Band", "Sensor", "get_sensor", "list_sensors"]


@lru_cache(maxsize=1)
def _table() -> Dict[str, dict]:
    text = resources.files("spires_lut.sensors.data").joinpath("bands.toml").read_text()
    return tomllib.loads(text)["sensors"]


def list_sensors() -> List[str]:
    return sorted(_table())


def _build(name: str, seen: tuple = ()) -> Sensor:
    table = _table()
    if name not in table:
        raise KeyError(f"unknown sensor {name!r}; available: {list_sensors()}")
    if name in seen:
        raise ValueError(f"alias cycle in bands.toml: {seen + (name,)}")
    entry = table[name]
    if "curves_from" in entry:
        source = _build(entry["curves_from"], seen + (name,))
        rename = entry.get("rename")
        if rename is None:
            bands = source.bands
        else:
            by_name = {b.name: b for b in source.bands}
            missing = [k for k in rename if k not in by_name]
            if missing:
                raise ValueError(f"sensor {name!r}: rename refers to unknown {source.name} bands {missing}")
            bands = tuple(
                Band(**{**by_name[old].__dict__, "name": new}) for old, new in rename.items()
            )
        return Sensor(name=name, platform=entry["platform"], bands=bands,
                      srf_source=entry.get("srf_source", source.srf_source))
    bands = tuple(Band(**row) for row in entry["bands"])
    return Sensor(name=name, platform=entry["platform"], bands=bands, srf_source=entry["srf_source"])


@lru_cache(maxsize=None)
def get_sensor(name: str) -> Sensor:
    sensor = _build(name)
    sensor_weights(sensor, DISORT_GRID_NM)   # coverage check: raises ValueError naming the band
    return sensor
