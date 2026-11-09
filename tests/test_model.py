import pytest

from spires_lut.sensors.model import Band, Sensor


def test_tophat_requires_edges():
    with pytest.raises(ValueError, match="B1"):
        Band(name="B1", center_nm=443.0, fwhm_nm=16.0, response="tophat")


def test_srf_requires_file_and_column():
    with pytest.raises(ValueError, match="srf_file"):
        Band(name="B1", center_nm=443.0, fwhm_nm=16.0, response="srf", srf_file="oli.csv")


def test_unknown_response_kind_rejected():
    with pytest.raises(ValueError, match="response"):
        Band(name="B1", center_nm=443.0, fwhm_nm=16.0, response="boxcar")


def test_sensor_band_names_in_order():
    bands = (
        Band(name="B2", center_nm=482.0, fwhm_nm=60.0, response="gaussian"),
        Band(name="B3", center_nm=561.0, fwhm_nm=57.0, response="gaussian"),
    )
    s = Sensor(name="X", platform="test", bands=bands, srf_source="n/a")
    assert s.band_names == ("B2", "B3")


def test_sensor_rejects_duplicate_band_names():
    b = Band(name="B2", center_nm=482.0, fwhm_nm=60.0, response="gaussian")
    with pytest.raises(ValueError, match="duplicate"):
        Sensor(name="X", platform="test", bands=(b, b), srf_source="n/a")
