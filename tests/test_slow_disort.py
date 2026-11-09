"""End-to-end: DISORT on a tiny grid -> OLI LUT. Skipped unless the compiled extension
and an atmospheric LUT (env SPIRES_ATM_LUT) are available."""
import os
from pathlib import Path

import pytest

disort = pytest.importorskip("disort")
ATM = os.environ.get("SPIRES_ATM_LUT")
pytestmark = pytest.mark.skipif(ATM is None, reason="set SPIRES_ATM_LUT to run the DISORT smoke test")


def test_disort_tiny_grid_to_oli(tmp_path):
    from spires_contract.lut import validate_reflectance_lut
    from spires_lut.io import read_lut
    from spires_lut.lut import build_sensor_lut
    from spires_lut.sensors import get_sensor
    from spires_lut.simulate import create_spires_lut

    create_spires_lut(
        output_dir=tmp_path, atm_lut=Path(ATM), cpu_cores=1,
        lap_points=2, illumination_angle_points=2, toa_sza_points=2, altitude_points=2, svf_points=2,
    )
    hs = read_lut(tmp_path / "spires_reflectance_lut.nc")
    out = build_sensor_lut(hs, get_sensor("OLI"), source="smoke")
    validate_reflectance_lut(out, expected_lap_type="dust")
    assert out.sizes["band"] == 9
    assert 0.0 <= float(out["reflectance"].min()) and float(out["reflectance"].max()) <= 1.0
