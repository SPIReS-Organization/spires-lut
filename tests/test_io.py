import numpy as np
import pytest
from spires_contract import ContractError

from spires_lut.io import read_lut, write_lut
from spires_lut.lut import build_sensor_lut
from spires_lut.sensors import get_sensor


def test_round_trip_preserves_values_dtype_bands_and_attrs(tmp_path, hyperspectral_lut):
    ds = build_sensor_lut(hyperspectral_lut, get_sensor("MSI"))
    path = write_lut(ds, tmp_path / "msi.nc")
    back = read_lut(path)
    assert back["reflectance"].dtype == np.float32
    np.testing.assert_array_equal(back["reflectance"].values, ds["reflectance"].values)
    assert list(back["band"].values) == list(ds["band"].values)
    assert back["lap_concentration"].attrs["lap_type"] == "dust"
    assert back.attrs["sensor"] == "MSI"
    np.testing.assert_allclose(back["center_wavelength_nm"].values, ds["center_wavelength_nm"].values)


def test_hyperspectral_lut_round_trips_too(tmp_path, hyperspectral_lut):
    path = write_lut(hyperspectral_lut, tmp_path / "hs.nc")
    back = read_lut(path)
    assert back["band"].dtype.kind == "f"
    assert back["band"].size == 475


def test_write_refuses_invalid(tmp_path, hyperspectral_lut):
    bad = hyperspectral_lut.copy()
    bad["reflectance"] = bad["reflectance"].astype(np.float64)
    with pytest.raises(ContractError):
        write_lut(bad, tmp_path / "bad.nc")
    assert not (tmp_path / "bad.nc").exists()


def test_read_refuses_invalid_file(tmp_path, hyperspectral_lut):
    bad = hyperspectral_lut.copy()
    bad["lap_concentration"].attrs.pop("lap_type")
    bad.to_netcdf(tmp_path / "bad.nc")
    with pytest.raises(ContractError):
        read_lut(tmp_path / "bad.nc")
