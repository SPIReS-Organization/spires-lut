import numpy as np
import pytest
from spires_contract import ContractError
from spires_contract.lut import validate_reflectance_lut

from spires_lut.lut import build_sensor_lut, sensor_lut_filename
from spires_lut.sensors import get_sensor, list_sensors


@pytest.mark.parametrize("name", list_sensors())
def test_every_sensor_builds_a_valid_float32_lut(hyperspectral_lut, name):
    out = build_sensor_lut(hyperspectral_lut, get_sensor(name))
    validate_reflectance_lut(out, expected_lap_type="dust")   # raises on violation
    r = out["reflectance"]
    assert r.dtype == np.float32
    assert r.dims == ("band", "solar_angle", "lap_concentration", "sqrt_grain_radius")
    assert list(out["band"].values) == list(get_sensor(name).band_names)
    assert out.attrs["sensor"] == name
    assert out["lap_concentration"].attrs["lap_type"] == "dust"


def test_band_values_lie_between_spectrum_extremes(hyperspectral_lut):
    out = build_sensor_lut(hyperspectral_lut, get_sensor("OLI"))
    hs = hyperspectral_lut["reflectance"]
    assert float(out["reflectance"].max()) <= float(hs.max()) + 1e-6
    assert float(out["reflectance"].min()) >= float(hs.min()) - 1e-6


def test_oli_nir_matches_direct_weighted_mean(hyperspectral_lut):
    from spires_lut.sensors.convolve import DISORT_GRID_NM, band_weights
    oli = get_sensor("OLI")
    out = build_sensor_lut(hyperspectral_lut, oli)
    b5 = next(b for b in oli.bands if b.name == "B5")
    w = band_weights(b5, DISORT_GRID_NM)
    expected = np.tensordot(w, hyperspectral_lut["reflectance"].values, axes=([0], [0]))
    np.testing.assert_allclose(out["reflectance"].sel(band="B5").values, expected, rtol=1e-6)


def test_attrs_record_provenance(hyperspectral_lut):
    out = build_sensor_lut(hyperspectral_lut, get_sensor("LANDSAT10"), source="synthetic.nc")
    for key in ("sensor", "platform", "srf_source", "srf_response_kind", "lap_type",
                "spires_lut_version", "created", "source_hyperspectral_lut"):
        assert key in out.attrs, key
    assert out.attrs["srf_response_kind"] == "gaussian"
    assert out.attrs["source_hyperspectral_lut"] == "synthetic.nc"
    assert out.attrs["surface_model"] == "unknown"   # fixture carries no engine stamp


def test_attrs_propagate_surface_model(hyperspectral_lut):
    stamped = hyperspectral_lut.assign_attrs(surface_model="delta_eddington")
    out = build_sensor_lut(stamped, get_sensor("OLI"), source="x.nc")
    assert out.attrs["surface_model"] == "delta_eddington"


def test_rejects_dataset_without_wavelength_band(hyperspectral_lut):
    bad = hyperspectral_lut.rename({"band": "wavelength"})
    with pytest.raises(ValueError, match="band"):
        build_sensor_lut(bad, get_sensor("OLI"))


def test_rejects_wrong_dim_order(hyperspectral_lut):
    bad = hyperspectral_lut.transpose("solar_angle", "band", "lap_concentration", "sqrt_grain_radius")
    with pytest.raises(ValueError, match="dims"):
        build_sensor_lut(bad, get_sensor("OLI"))


def test_rejects_string_band_input(hyperspectral_lut):
    already = build_sensor_lut(hyperspectral_lut, get_sensor("OLI"))
    with pytest.raises(ValueError, match="numeric"):
        build_sensor_lut(already, get_sensor("OLI"))


def test_contract_violation_propagates(hyperspectral_lut):
    bad = hyperspectral_lut.copy()
    bad["lap_concentration"].attrs.pop("lap_type")
    with pytest.raises(ContractError):
        build_sensor_lut(bad, get_sensor("OLI"))


def test_filename_convention():
    assert sensor_lut_filename("HLSS30-msi", "dust") == "spires_reflectance_lut_hlss30-msi_dust.nc"
