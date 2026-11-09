import numpy as np
import pytest

from spires_lut.sensors import get_sensor, list_sensors
from spires_lut.sensors.convolve import DISORT_GRID_NM, band_weights

EXPECTED_COUNTS = {
    "OLI": 9, "OLI2": 9, "MSI": 13, "MSI-S2B": 13, "MODIS": 7,
    "HLSL30": 8, "HLSS30": 7, "HLSS30-msi": 13, "LANDSAT10": 21,
}


def test_list_sensors_matches_expected_registry():
    assert set(list_sensors()) == set(EXPECTED_COUNTS)


@pytest.mark.parametrize("name,count", sorted(EXPECTED_COUNTS.items()))
def test_every_sensor_loads_with_expected_band_count(name, count):
    s = get_sensor(name)
    assert s.name == name
    assert len(s.bands) == count


def test_unknown_sensor_lists_available():
    with pytest.raises(KeyError, match="OLI"):
        get_sensor("ETM+")


def test_hls_aliases_share_oli_curves():
    oli = {b.name: b for b in get_sensor("OLI").bands}
    l30 = get_sensor("HLSL30")
    assert l30.band_names == ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09")
    assert l30.bands[0].srf_file == oli["B1"].srf_file == "oli.csv"
    assert l30.bands[0].srf_band == "B1"
    s30 = get_sensor("HLSS30")
    assert s30.band_names == ("B01", "B02", "B03", "B04", "B8A", "B11", "B12")
    assert s30.bands[4].srf_band == "B5"          # OLI NIR under the S30 code B8A
    assert get_sensor("HLSS30-msi").bands[8].srf_file == "msi_s2a.csv"


def test_spires_io_band_codes():
    assert get_sensor("MSI").band_names == (
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
    assert get_sensor("MODIS").band_names == ("1", "2", "3", "4", "5", "6", "7")


@pytest.mark.parametrize("name", ["OLI", "OLI2", "MSI", "MSI-S2B", "MODIS"])
def test_real_srf_weights_sum_to_one_and_centre_near_published(name):
    s = get_sensor(name)
    for b in s.bands:
        w = band_weights(b, DISORT_GRID_NM)
        assert w.sum() == pytest.approx(1.0, abs=1e-9), b.name
        weighted_centre = float(np.sum(w * DISORT_GRID_NM))
        # catches nm/um unit mistakes and column mix-ups; 10 nm bins + asymmetric curves -> allow 8 nm
        assert abs(weighted_centre - b.center_nm) < 8.0, (name, b.name, weighted_centre, b.center_nm)


def test_landsat10_excludes_tir_and_covers_snow_bands():
    names = {b.long_name for b in get_sensor("LANDSAT10").bands}
    assert "snow/ice 1" in names and "snow/ice 2" in names
    assert not any(n.startswith("TIR") for n in names)
